#!/usr/bin/env python3
"""
GeoJSON to COCO Dataset Converter with Negative Examples

Converts a GeoJSON file with polygon annotations into COCO detection and panoptic segmentation formats.
Also includes negative examples from cleaned annotation spreadsheet data.

Features:
- Supports multiple date formats (YYMMMDD, YYYY-MM-DD, YYMMMDDhhmmss)
- Supports both M2AS and PANSHARP image file types
- Image normalization to 0-10000 using min/max linear stretch (preserves all data, no clipping)
- Intelligent image selection for negative examples:
  - Uses specific date from annotation if available
  - Falls back to most recent 'good' image if no specific date
  - Selects best image when multiple exist for same date (most valid pixels)
- Handles multiple images with close timestamps by selecting the one with most valid data
- Processes both explicit no-fields CSV and main annotations CSV for negative examples
"""

import os
import json
import time
import math
import warnings
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, NamedTuple, Set
import multiprocessing as mp
import re

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.features import rasterize
import PIL.Image as Image
from tqdm import tqdm
import shutil
import pycocotools.mask as mask_util

class InstanceInfo(NamedTuple):
    """Store information about a single instance (field)"""
    id: int
    mask: np.ndarray
    bbox: List[float]
    area: float
    color: Optional[np.ndarray] = None

@dataclass
class CategoryInfo:
    """Store category information for COCO format"""
    id: int
    name: str
    supercategory: str
    isthing: bool = True
    color: Optional[List[int]] = None

def rgb2id(color: np.ndarray) -> int:
    """Convert RGB color to unique ID using the COCO panopticapi formula"""
    if len(color) == 4:  # RGBA
        return int(color[0]) + 256 * int(color[1]) + 256 * 256 * int(color[2])
    return int(color[0]) + 256 * int(color[1]) + 256 * 256 * int(color[2])

class GeoJSONToCOCOConverter:
    """Convert GeoJSON annotations to COCO format with negative examples support"""
    
    def __init__(
        self,
        geojson_files: List[str],
        image_dir: str,
        output_dir: str,
        annotations_csv: str = None,
        no_fields_csv: str = None,
        split: str = "train",
        min_area: int = 10,
        num_workers: int = None,
        background_id: int = 4988569,
        field_category_id: int = 1,
        background_category_id: int = 2,
        field_color: List[int] = [100, 204, 25], # Green
        background_color: List[int] = [153, 30, 76], # Dark red
        sample_fraction: float = 1.0,
        random_seed: int = 42,
        verbose: bool = True,
        labeler_priority: List[str] = None
    ):
        """Initialize the converter."""

        self.geojson_files = [Path(f) for f in geojson_files]
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.annotations_csv = Path(annotations_csv) if annotations_csv else None
        self.no_fields_csv = Path(no_fields_csv) if no_fields_csv else None
        self.split = split
        self.min_area = min_area
        self.num_workers = num_workers if num_workers is not None else mp.cpu_count()
        self.background_id = background_id
        self.field_category_id = field_category_id
        self.background_category_id = background_category_id
        self.sample_fraction = sample_fraction
        self.random_seed = random_seed
        self.verbose = verbose

        # Labeler priority order (ZF=zhanpei, KR=kyle, IF=ian, AB=aspen, CS=caspian, EV=evelyn, CD=clark, GC=chen)
        self.labeler_priority = labeler_priority or ['ZF', 'KR', 'IF', 'AB', 'CS', 'EV', 'CD', 'GC']

        np.random.seed(self.random_seed)
        
        self.field_color = field_color 
        self.background_color = background_color  

        self.categories = [
            CategoryInfo(
                id=field_category_id, name="ag_field", supercategory="landcover", 
                isthing=True, color=self.field_color
            ),
            CategoryInfo(
                id=background_category_id, name="background", supercategory="background", 
                isthing=False, color=self.background_color
            )
        ]

        self._setup_directories()

        self.image_id_counter = 0
        
        # Load annotation data
        self.annotations_df = None
        self.no_fields_df = None
        self._load_annotation_data()
        
        # Tracking for date-matching issues
        self.date_matching_issues = []
        self.alternative_date_used = []
        self.fallback_images_used = []
        
        if self.verbose:
            print(f"Initialized GeoJSON to COCO converter with {len(self.geojson_files)} GeoJSON files.")

    def _setup_directories(self):
        """Create necessary output directories"""
        self.output_dir.mkdir(exist_ok=True, parents=True)
        (self.output_dir / "annotations").mkdir(exist_ok=True)
        
        (self.output_dir / self.split).mkdir(exist_ok=True)
        (self.output_dir / f"panoptic_{self.split}").mkdir(exist_ok=True)
        (self.output_dir / f"panoptic_semseg_{self.split}").mkdir(exist_ok=True)
            
    def _get_next_image_id(self) -> int:
        image_id = self.image_id_counter
        self.image_id_counter += 1
        return image_id

    def _load_annotation_data(self):
        """Load the cleaned annotation CSV data"""
        if self.annotations_csv and self.annotations_csv.exists():
            if self.verbose:
                print(f"Loading annotation data from {self.annotations_csv}")
            self.annotations_df = pd.read_csv(self.annotations_csv)
            
        if self.no_fields_csv and self.no_fields_csv.exists():
            if self.verbose:
                print(f"Loading no-fields data from {self.no_fields_csv}")
            self.no_fields_df = pd.read_csv(self.no_fields_csv)
        
    def _extract_tile_id_from_filename(self, filename: str) -> Optional[str]:
        """Extract tile_id from image filename using the pattern from geojson_to_coco.py"""
        # Pattern: {tile_id}_{date}*-M2AS-*.tif or {tile_id}_{date}*-PANSHARP-*.tif
        # Example: 35_20MAY03081809-M2AS-507761552070_01_P002.tif
        # Example: 35_20MAY03081809-PANSHARP-507761552070_01_P002.tif
        match = re.match(r'^(\d+)_\d{2}[A-Z]{3}\d{2}.*\.tif$', filename)
        if match:
            return match.group(1)
        return None

    def _convert_date_format(self, date_str: str) -> Optional[str]:
        """
        Convert various date formats to the format used in image filenames.
        
        Args:
            date_str: Date in various formats (e.g., '19MAY15', '2019-05-15', '20MAY23082821')
            
        Returns:
            Date string in format YYMMMDD or YYMMMDDhhmmss, or None if conversion fails
        """
        if pd.isna(date_str) or not date_str:
            return None
            
        date_str = str(date_str).strip()
        
        # If already in the target format (YYMMMDD or YYMMMDDhhmmss)
        if re.match(r'^\d{2}[A-Z]{3}\d{2}(\d{6})?$', date_str):
            # Handle ambiguous DDMMMYY vs YYMMMDD format for basic format
            if len(date_str) == 7:  # Basic format without time
                resolved = self._resolve_ambiguous_date_format(date_str)
                if resolved:
                    return self._validate_date_string(resolved)
                # Try swapped year/day even if validation failed above
                alt = self._get_alternative_date_interpretation(date_str)
                return self._validate_date_string(alt) if alt else None
            return self._validate_date_string(date_str)
        
        # Handle ISO format (YYYY-MM-DD)
        if re.match(r'^\d{4}-\d{2}-\d{2}', date_str):
            try:
                dt = pd.to_datetime(date_str)
                day = f"{dt.day:02d}"
                month_map = {
                    1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN',
                    7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'
                }
                month = month_map[dt.month]
                year = f"{dt.year % 100:02d}"
                candidate = f"{year}{month}{day}"
                return self._validate_date_string(candidate)
            except:
                return None
        
        # Handle other formats by trying to parse as datetime
        try:
            dt = pd.to_datetime(date_str)
            day = f"{dt.day:02d}"
            month_map = {
                1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN',
                7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'
            }
            month = month_map[dt.month]
            year = f"{dt.year % 100:02d}"
            candidate = f"{year}{month}{day}"
            return self._validate_date_string(candidate)
        except:
            return None

    def _resolve_ambiguous_date_format(self, date_str: str) -> Optional[str]:
        """
        Resolve ambiguous date formats like '11OCT21' which could be DDMMMYY or YYMMMDD.
        
        Args:
            date_str: Date string in format like '11OCT21' or '21OCT11'
            
        Returns:
            Resolved date string in YYMMMDD format, or None if resolution fails
        """
        if not re.match(r'^\d{2}[A-Z]{3}\d{2}$', date_str):
            return date_str  # Not in the ambiguous format, return as-is
        
        # Extract components
        first_two = int(date_str[:2])
        month_str = date_str[2:5]
        last_two = int(date_str[5:])
        
        # Month mapping
        month_map = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
            'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }
        
        if month_str not in month_map:
            return None
        
        month_num = month_map[month_str]
        
        # Try interpretation 1: YYMMMDD (first_two = year, last_two = day)
        year1 = first_two
        day1 = last_two
        
        # Try interpretation 2: DDMMMYY (first_two = day, last_two = year)
        day2 = first_two
        year2 = last_two
        
        # Validate both interpretations based on expected ranges
        valid_interpretations = []
        
        # Check interpretation 1: YYMMMDD
        if (15 <= year1 <= 23 and 1 <= day1 <= 31):
            valid_interpretations.append(('YYMMMDD', year1, month_num, day1))
        
        # Check interpretation 2: DDMMMYY
        if (15 <= year2 <= 23 and 1 <= day2 <= 31):
            valid_interpretations.append(('DDMMMYY', year2, month_num, day2))
        
        if not valid_interpretations:
            if self.verbose:
                print(f"Could not resolve ambiguous date '{date_str}' - no valid interpretation found")
            return None
        
        if len(valid_interpretations) > 1:
            if self.verbose:
                print(f"Multiple valid interpretations for '{date_str}': {valid_interpretations}")
            # Prefer YYMMMDD interpretation if both are valid
            for interpretation, year, month, day in valid_interpretations:
                if interpretation == 'YYMMMDD':
                    return f"{year:02d}{month_str}{day:02d}"
        
        # Use the single valid interpretation
        interpretation, year, month, day = valid_interpretations[0]
        if interpretation == 'YYMMMDD':
            return f"{year:02d}{month_str}{day:02d}"
        else:  # DDMMMYY
            return f"{year:02d}{month_str}{day:02d}"

    def _validate_date_string(self, date_str: str) -> Optional[str]:
        """
        Ensure date is in YYMMMDD or YYMMMDDhhmmss with year 15-23 and day 1-31.
        """
        match = re.match(r'^(\d{2})([A-Z]{3})(\d{2})(\d{6})?$', date_str)
        if not match:
            return None

        year, month_str, day = int(match.group(1)), match.group(2), int(match.group(3))
        month_map = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
            'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }

        if month_str not in month_map:
            return None

        if 15 <= year <= 23 and 1 <= day <= 31:
            return date_str

        return None

    def _find_best_image_for_date(self, tile_id: str, date: str, is_negative_example: bool = False) -> Optional[Path]:
        """
        Find the best image file for a given tile_id and date.
        
        Args:
            tile_id: The tile/AOI ID
            date: Date in format YYMMMDD or YYMMMDDhhmmss
            
        Returns:
            Path to the best image, or None if no image found
        """
        search_path = self.image_dir / tile_id
        if not search_path.exists():
            return None
        
        # Try the original date format first
        matching_files = self._try_date_pattern(search_path, tile_id, date)
        
        # If no files found and date is in ambiguous format, try the alternative interpretation
        # Only do this for negative examples (spreadsheet dates), not for GeoJSON dates
        if not matching_files and is_negative_example and len(date) == 7 and re.match(r'^\d{2}[A-Z]{3}\d{2}$', date):
            alternative_date = self._get_alternative_date_interpretation(date)
            if alternative_date and alternative_date != date:
                if self.verbose:
                    print(f"Trying alternative date interpretation for {date}: {alternative_date}")
                matching_files = self._try_date_pattern(search_path, tile_id, alternative_date)
                if matching_files:
                    # Track that alternative date was used
                    self.alternative_date_used.append({
                        'tile_id': tile_id,
                        'original_date': date,
                        'alternative_date': alternative_date,
                        'found_image': matching_files[0].name
                    })
        
        if not matching_files:
            if self.verbose:
                print(f"No files found matching pattern for tile_id {tile_id} and date {date}, checking other tiles for same date")
            # If no image found for this specific tile, check other tiles for the same date
            # (tiles may spatially overlap, so another tile might have the M2AS image)
            matching_files = self._find_image_in_other_tiles(date)
            if matching_files:
                # Prefer M2AS over PANSHARP
                m2as_files = [f for f in matching_files if 'M2AS' in f.name]
                if m2as_files:
                    matching_files = m2as_files
                
                if self.verbose:
                    found_tile_id = matching_files[0].parent.name
                    print(f"Found image for date {date} in overlapping tile {found_tile_id}: {matching_files[0].name}")
                # Track that we used an overlapping tile's image
                self.date_matching_issues.append({
                    'tile_id': tile_id,
                    'date': date,
                    'issue': f'Used image from overlapping tile {matching_files[0].parent.name}'
                })
            else:
                # Track the date matching issue
                self.date_matching_issues.append({
                    'tile_id': tile_id,
                    'date': date,
                    'issue': 'No matching image found in any tile'
                })
                return None
        
        if len(matching_files) == 1:
            return matching_files[0]
        
        # Multiple files found - select the one with the most valid data
        best_image_path = None
        max_valid_pixels = 0
        
        for file_path in matching_files:
            try:
                with rio.open(file_path) as src:
                    img_data = src.read()
                    # Count valid pixels (non-zero, non-nan)
                    valid_pixels = np.count_nonzero((img_data != 0) & ~np.isnan(img_data))
                    
                    if valid_pixels > max_valid_pixels:
                        max_valid_pixels = valid_pixels
                        best_image_path = file_path
                        
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not read {file_path}: {e}")
                continue
        
        return best_image_path if best_image_path else matching_files[0]

    def _find_image_in_other_tiles(self, date: str) -> List[Path]:
        """
        Search all tile directories for an M2AS or PANSHARP image with the given date.
        Used when the requested tile doesn't have an image for that date, but an overlapping tile might.
        
        Args:
            date: Date in format YYMMMDD or YYMMMDDhhmmss
            
        Returns:
            List of matching file paths (empty if none found)
        """
        matching_files = []
        
        # Search all tile subdirectories
        if not self.image_dir.exists():
            return matching_files
        
        # Create glob pattern to match the date (without tile_id prefix)
        if len(date) == 7:  # Basic format: YYMMMDD
            glob_pattern_m2as = f"*_{date}*-M2AS-*.tif"
            glob_pattern_pansharp = f"*_{date}*-PANSHARP-*.tif"
        else:  # Extended format: YYMMMDDhhmmss
            glob_pattern_m2as = f"*_{date}*-M2AS-*.tif"
            glob_pattern_pansharp = f"*_{date}*-PANSHARP-*.tif"
        
        # Search in all subdirectories (each tile has its own folder)
        for tile_dir in self.image_dir.iterdir():
            if tile_dir.is_dir():
                m2as_files = list(tile_dir.glob(glob_pattern_m2as))
                pansharp_files = list(tile_dir.glob(glob_pattern_pansharp))
                matching_files.extend(m2as_files)
                matching_files.extend(pansharp_files)
        
        return matching_files

    def _try_date_pattern(self, search_path: Path, tile_id: str, date: str) -> List[Path]:
        """Try to find files matching a specific date pattern."""
        # Create glob pattern to match the date
        # Handle both basic format (YYMMMDD) and extended format (YYMMMDDhhmmss)
        # Support both M2AS and PANSHARP file types (P2AS is panchromatic, skip it)
        if len(date) == 7:  # Basic format: YYMMMDD
            glob_pattern_m2as = f"{tile_id}_{date}*-M2AS-*.tif"
            glob_pattern_pansharp = f"{tile_id}_{date}*-PANSHARP-*.tif"
        else:  # Extended format: YYMMMDDhhmmss
            glob_pattern_m2as = f"{tile_id}_{date}*-M2AS-*.tif"
            glob_pattern_pansharp = f"{tile_id}_{date}*-PANSHARP-*.tif"
        
        # Try both patterns and combine results
        m2as_files = list(search_path.glob(glob_pattern_m2as))
        pansharp_files = list(search_path.glob(glob_pattern_pansharp))
        
        return m2as_files + pansharp_files

    def _extract_date_from_image_filename(self, filename: str) -> Optional[str]:
        """
        Extract date from image filename.
        
        Args:
            filename: Image filename like '2297_20MAY14081253-M2AS-507761556040_01_P001.tif'
                      or '2297_20MAY14081253-PANSHARP-507761556040_01_P001.tif'
            
        Returns:
            Date string in format like '20MAY14' or None if not found
        """
        # Pattern: {tile_id}_{date}*-M2AS-*.tif or {tile_id}_{date}*-PANSHARP-*.tif
        # Extract the date part from the filename
        import re
        match = re.match(r'^\d+_(\d{2}[A-Z]{3}\d{2})\d{6}-(M2AS|PANSHARP)-.*\.tif$', filename)
        if match:
            date_str = match.group(1)
            return self._validate_date_string(date_str)
        return None

    def _get_alternative_date_interpretation(self, date_str: str) -> Optional[str]:
        """
        Get the alternative interpretation of an ambiguous date.
        If date_str is '11OCT21', return '21OCT11' and vice versa.
        
        Args:
            date_str: Date string in format like '11OCT21'
            
        Returns:
            Alternative interpretation or None if not ambiguous
        """
        if not re.match(r'^\d{2}[A-Z]{3}\d{2}$', date_str):
            return None
        
        first_two = date_str[:2]
        month_str = date_str[2:5]
        last_two = date_str[5:]
        
        # Swap first and last two digits
        alt = f"{last_two}{month_str}{first_two}"
        return self._validate_date_string(alt)

    def _find_most_recent_good_image(self, tile_id: str) -> Optional[Path]:
        """
        Find the most recent 'good' image for a tile_id when no specific date is available.
        
        Args:
            tile_id: The tile/AOI ID
            
        Returns:
            Path to the best image, or None if no image found
        """
        search_path = self.image_dir / tile_id
        if not search_path.exists():
            return None
        
        # Get all M2AS and PANSHARP images for this tile (P2AS is panchromatic, skip it)
        m2as_images = list(search_path.glob(f"{tile_id}_*-M2AS-*.tif"))
        pansharp_images = list(search_path.glob(f"{tile_id}_*-PANSHARP-*.tif"))
        all_images = m2as_images + pansharp_images
        
        if not all_images:
            return None
        
        # Sort by filename (which includes date) to get most recent
        all_images.sort(key=lambda x: x.name, reverse=True)
        
        # Find the best image among the most recent ones
        best_image_path = None
        max_valid_pixels = 0
        
        # Check the 5 most recent images
        for file_path in all_images[:5]:
            try:
                with rio.open(file_path) as src:
                    img_data = src.read()
                    # Count valid pixels (non-zero, non-nan)
                    valid_pixels = np.count_nonzero((img_data != 0) & ~np.isnan(img_data))
                    
                    if valid_pixels > max_valid_pixels:
                        max_valid_pixels = valid_pixels
                        best_image_path = file_path
                        
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not read {file_path}: {e}")
                continue
        
        return best_image_path if best_image_path else all_images[0]

    def _find_image_by_date_matching(self, tile_id: str, date: str, is_negative_example: bool = False) -> Optional[Path]:
        """Find image file by tile_id and date matching, similar to geojson_to_coco.py logic"""
        # First try to find image for the specific date
        image_path = self._find_best_image_for_date(tile_id, date, is_negative_example)
        if image_path:
            if self.verbose:
                print(f"Found exact date match for tile_id {tile_id}, date {date}: {image_path.name}")
            return image_path
        
        # If no image found for the specific date, try to find the most recent good image
        if self.verbose:
            print(f"No exact date match found for tile_id {tile_id} and date {date}, trying most recent good image")
        
        fallback_image = self._find_most_recent_good_image(tile_id)
        if fallback_image:
            if self.verbose:
                print(f"Using fallback image for tile_id {tile_id}: {fallback_image.name}")
            # Track fallback image usage
            self.fallback_images_used.append({
                'tile_id': tile_id,
                'original_date': date,
                'fallback_image': fallback_image.name
            })
        return fallback_image

    def _is_negative_example_from_csv(self, tile_id: str, date: str) -> bool:
        """Check if an image should be considered a negative example based on CSV data"""
        # First check no_fields_df
        if self.no_fields_df is not None:
            # Match by aoi_id (which could be tile_id) and date
            # Convert the date to the format used in the CSV for comparison
            csv_date = self._convert_date_format(date)
            if csv_date:
                no_fields_match = self.no_fields_df[
                    (self.no_fields_df.get('aoi_id', '').astype(str) == str(tile_id)) & 
                    (self.no_fields_df.get('image_acquisition_date', '') == csv_date)
                ]
                if len(no_fields_match) > 0:
                    return True
        
        # Check main annotations_df for 'No ag'/'No fields' annotations
        if self.annotations_df is not None:
            # Find matching rows by date (assuming tile_id might be encoded in the row index or other column)
            csv_date = self._convert_date_format(date)
            if csv_date:
                date_matches = self.annotations_df[
                    self.annotations_df.get('image_acquisition_date', '') == csv_date
                ]
                
                if len(date_matches) > 0:
                    # Check notes columns for no-fields patterns
                    notes_cols = ['L1 notes', 'L2 notes', 'L3 notes', 'Expert notes']
                    no_fields_patterns = [
                        r'no\s+field', r'no\s+ag', r'no\s+agriculture', 
                        r'no\s+visible', r'nothing', r'empty', r'none'
                    ]
                    
                    for _, row in date_matches.iterrows():
                        for col in notes_cols:
                            if col in row and pd.notna(row[col]):
                                note_text = str(row[col]).lower()
                                for pattern in no_fields_patterns:
                                    if re.search(pattern, note_text, re.IGNORECASE):
                                        return True
        
        return False

    def _get_prioritized_labeler(self, row: pd.Series) -> str:
        """Get the highest priority labeler for a given annotation row"""
        labeler_cols = ['Expert initial', 'L3 initials', 'L2 initials', 'L1 initials']
        
        for col in labeler_cols:
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                labeler = str(row[col]).strip().upper()
                if labeler in self.labeler_priority:
                    return labeler
                    
        return 'UNKNOWN'
        
    def _load_annotations(self) -> Tuple[gpd.GeoDataFrame, List[str], Set[Tuple[str, str]]]:
        """Load annotations from GeoJSON files and identify all tile_ids with annotations"""
        all_gdfs = []
        all_tile_ids = set()
        annotated_pairs = set()  # (tile_id, date) pairs that have polygon annotations
        
        for geojson_file in self.geojson_files:
            if self.verbose:
                print(f"Loading annotations from {geojson_file}")
            
            try:
                gdf = gpd.read_file(geojson_file)
                if not gdf.empty:
                    all_gdfs.append(gdf)
                    # Track which (tile_id, date) pairs have annotations
                    for _, row in gdf.iterrows():
                        tile_id = str(row.get('tile_id', ''))
                        date = str(row.get('date', ''))
                        if tile_id and date:
                            annotated_pairs.add((tile_id, date))
                            all_tile_ids.add(tile_id)
            except Exception as e:
                warnings.warn(f"Error loading {geojson_file}: {e}")
                continue
        
        # Combine all GeoDataFrames
        if all_gdfs:
            combined_gdf = pd.concat(all_gdfs, ignore_index=True)
        else:
            combined_gdf = gpd.GeoDataFrame()
        
        # Sample if needed
        if self.sample_fraction < 1.0 and not combined_gdf.empty:
            unique_tile_ids = combined_gdf["tile_id"].unique()
            sample_size = int(len(unique_tile_ids) * self.sample_fraction)
            sampled_ids = np.random.choice(unique_tile_ids, sample_size, replace=False)
            combined_gdf = combined_gdf[combined_gdf["tile_id"].isin(sampled_ids)]
            all_tile_ids = set(sampled_ids)
            # Update annotated_pairs
            annotated_pairs = {(tid, date) for tid, date in annotated_pairs if tid in sampled_ids}

        tile_ids = list(all_tile_ids)
        
        if self.verbose:
            print(f"Found {len(tile_ids)} unique tile IDs with annotations to process.")
            
        return combined_gdf, tile_ids, annotated_pairs

    def _identify_negative_examples(self, annotated_pairs: Set[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Identify negative examples from the annotation CSV that are not in the GeoJSON annotations"""
        negative_examples = []
        
        if self.annotations_df is None and self.no_fields_df is None:
            if self.verbose:
                print("No annotations CSV provided, skipping negative example identification.")
            return negative_examples
        
        # Process no_fields_df first (these are explicitly marked as no-fields)
        if self.no_fields_df is not None:
            if self.verbose:
                print(f"Processing {len(self.no_fields_df)} no-fields entries from CSV")
            
            for _, row in self.no_fields_df.iterrows():
                # Get tile_id from aoi_id column
                tile_id = row.get('aoi_id', '')
                if pd.isna(tile_id) or not tile_id:
                    continue
                
                tile_id = str(tile_id).strip()
                
                # Get the specific date from the annotation if available
                specific_date = None
                date_cols = ['L1 image date', 'L2 date', 'L3 date', 'Expert date']
                for col in date_cols:
                    if col in row and pd.notna(row[col]):
                        specific_date = self._convert_date_format(row[col])
                        if specific_date:
                            break
                
                # If no specific date found, use image_acquisition_date
                if not specific_date:
                    specific_date = self._convert_date_format(row.get('image_acquisition_date', ''))
                
                if not specific_date:
                    # Skip if we can't determine a date
                    continue
                
                # Check if this (tile_id, date) pair has polygon annotations
                if (tile_id, specific_date) not in annotated_pairs:
                    # Try to find the image for this specific date first
                    image_path = self._find_best_image_for_date(tile_id, specific_date, True)
                    
                    # If no image found for specific date, try most recent good image
                    if not image_path:
                        image_path = self._find_most_recent_good_image(tile_id)
                    
                    if image_path and image_path.exists():
                        negative_examples.append((tile_id, specific_date))
                        if self.verbose:
                            print(f"Added negative example: tile_id={tile_id}, date={specific_date}, image={image_path.name}")
        
        # Process main annotations_df for 'No ag'/'No fields' annotations
        if self.annotations_df is not None:
            # Get all images that have been labeled (have at least one labeler initial)
            labeler_cols = ['L1 initials', 'L2 initials', 'L3 initials', 'Expert initial']
            
            labeled_mask = self.annotations_df[labeler_cols].notna().any(axis=1)
            labeled_images = self.annotations_df[labeled_mask]
            
            if self.verbose:
                print(f"Processing {len(labeled_images)} labeled images from main annotations CSV")
                print("Looking for rows with explicit 'No ag' or 'No fields' notes...")
            
            for _, row in labeled_images.iterrows():
                # Try to extract tile_id from the row
                potential_tile_id = None
                
                # First, try the unnamed first column (might be tile_id)
                if ' ' in row.index and pd.notna(row[' ']):
                    potential_tile_id = str(row[' '])
                
                if not potential_tile_id:
                    # Skip if we can't determine tile_id
                    continue
                    
                tile_id = potential_tile_id.strip()
                
                # Check if this row explicitly has "No ag" or "No fields" notes
                notes_cols = ['L1 notes', 'L2 notes', 'L3 notes', 'Expert notes']
                no_fields_patterns = [
                    r'no\s+field', r'no\s+ag', r'no\s+agriculture', 
                    r'no\s+visible', r'nothing', r'empty', r'none'
                ]
                
                has_no_fields_note = False
                for col in notes_cols:
                    if col in row and pd.notna(row[col]):
                        note_text = str(row[col]).lower()
                        for pattern in no_fields_patterns:
                            if re.search(pattern, note_text, re.IGNORECASE):
                                has_no_fields_note = True
                                break
                    if has_no_fields_note:
                        break
                
                # Only proceed if this row explicitly has no-fields notes
                if not has_no_fields_note:
                    continue
                
                if self.verbose:
                    print(f"Found no-fields note for tile_id={tile_id}")
                
                # Get the specific date from the annotation if available
                specific_date = None
                date_cols = ['L1 image date', 'L2 date', 'L3 date', 'Expert date']
                for col in date_cols:
                    if col in row and pd.notna(row[col]):
                        specific_date = self._convert_date_format(row[col])
                        if specific_date:
                            break
                
                # If no specific date found, use image_acquisition_date
                if not specific_date:
                    specific_date = self._convert_date_format(row.get('image_acquisition_date', ''))
                
                if not specific_date:
                    # Skip if we can't determine a date
                    continue
                
                # Check if this (tile_id, date) pair has polygon annotations
                if (tile_id, specific_date) not in annotated_pairs:
                    # Try to find the image for this specific date first
                    image_path = self._find_best_image_for_date(tile_id, specific_date, True)
                    
                    # If no image found for specific date, try most recent good image
                    if not image_path:
                        image_path = self._find_most_recent_good_image(tile_id)
                    
                    if image_path and image_path.exists():
                        negative_examples.append((tile_id, specific_date))
                        if self.verbose:
                            print(f"Added negative example: tile_id={tile_id}, date={specific_date}, image={image_path.name}")
        
        if self.verbose:
            print(f"Identified {len(negative_examples)} potential negative examples")
        
        return negative_examples

    def _generate_instance_color(self) -> np.ndarray:
        """Generate a unique color for instance segmentation"""
        while True:
            base_color = np.array(self.field_color)
            jitter = np.random.randint(-30, 31, size=3)
            color = np.clip(base_color + jitter, 0, 255)
            if rgb2id(color) != self.background_id:
                return color

    def _create_rle_from_binary_mask(self, binary_mask: np.ndarray) -> Dict:
        """Create RLE encoded mask from binary mask"""
        binary_mask = binary_mask.astype(np.uint8)
        rle = mask_util.encode(np.asfortranarray(binary_mask))
        rle['counts'] = rle['counts'].decode('utf-8')
        return rle

    def _create_instance(
        self, geom, transform: rio.transform.Affine, height: int, width: int,
        instance_id: int, generate_color: bool = True
    ) -> Optional[InstanceInfo]:
        """Create instance from geometry"""
        try:
            mask = rasterize(
                [(geom, 1)], out_shape=(height, width), transform=transform,
                dtype=np.uint8, all_touched=True
            )
            
            if mask is None or not mask.any(): return None
                
            area = float(mask.sum())
            if area < self.min_area: return None
                
            rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
            if not rows.any() or not cols.any(): return None
                
            y_indices, x_indices = np.where(rows)[0], np.where(cols)[0]
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
            
            bbox = [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]
            color = self._generate_instance_color() if generate_color else None
            
            return InstanceInfo(id=instance_id, mask=mask, bbox=bbox, area=area, color=color)
            
        except Exception as e:
            warnings.warn(f"Error creating instance: {str(e)}")
            return None

    def _process_single_image(
        self, tile_id: str, date: str, annotations_gdf: gpd.GeoDataFrame, image_id: int, 
        generate_panoptic: bool = True, is_negative_example: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Process a single image and its annotations."""
        try:
            # Find the image file
            image_path = self._find_image_by_date_matching(tile_id, date, is_negative_example)
            
            if not image_path or not image_path.exists():
                warnings.warn(f"No image found for tile_id: {tile_id} and date: {date}")
                return None

            # Get image properties
            try:
                with rio.open(image_path) as src:
                    height, width = src.height, src.width
                    transform = src.transform
                    crs = src.crs
            except Exception as e:
                warnings.warn(f"Error reading image {image_path}: {e}")
                return None

            # Extract the actual date from the image filename for consistent naming
            actual_date = self._extract_date_from_image_filename(image_path.name)
            if not actual_date:
                # Fallback to original date if we can't extract from filename
                actual_date = date

            instances, instance_anns, panoptic_segments = [], [], []

            # Process annotations only if not a negative example
            if not is_negative_example and not annotations_gdf.empty:
                # Filter annotations for this tile_id and date
                tile_annotations = annotations_gdf[
                    (annotations_gdf['tile_id'].astype(str) == str(tile_id)) &
                    (annotations_gdf['date'].astype(str) == str(date))
                ]

                if not tile_annotations.empty:
                    # Reproject if necessary
                    if tile_annotations.crs != crs:
                        tile_annotations = tile_annotations.to_crs(crs)

                    for ann_idx, (_, row) in enumerate(tile_annotations.iterrows()):
                        annotation_id = image_id * 100000 + ann_idx
                        instance = self._create_instance(
                            row.geometry, transform, height, width, annotation_id, generate_panoptic
                        )
                        if instance is None: 
                            continue
                        
                        instances.append(instance)
                        
                        rle_mask = self._create_rle_from_binary_mask(instance.mask)

                        instance_ann = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": self.field_category_id,
                            "area": instance.area,
                            "bbox": instance.bbox,
                            "segmentation": rle_mask,
                            "iscrowd": 0
                        }
                        instance_anns.append(instance_ann)
                        
                        if generate_panoptic:
                            color_id = rgb2id(instance.color) if instance.color is not None else annotation_id
                            panoptic_segments.append({
                                "id": color_id,
                                "category_id": self.field_category_id,
                                "area": instance.area,
                                "bbox": instance.bbox,
                                "iscrowd": 0
                            })
            
            # Create standardized output filename: {tile_id}_{YYMMMDD}.tif
            # This ensures consistency between image files and panoptic/semseg masks
            standardized_filename = f"{tile_id}_{actual_date}.tif"

            # Create image info
            image_info = {
                "id": image_id,
                "file_name": standardized_filename,  # Use standardized name, not original
                "height": height,
                "width": width,
                "tile_id": tile_id,
                "date": date,
                "source_file": image_path.name,  # Keep original filename for reference
                "is_negative_example": is_negative_example or not instances,
                "sem_seg_file_name": f"panoptic_semseg_{self.split}/{tile_id}_{actual_date}.png",
                "pan_seg_file_name": f"panoptic_{self.split}/{tile_id}_{actual_date}.png"
            }
            
            result = {
                "image_info": image_info, 
                "image_path": str(image_path),
                "instances": instances, 
                "instance_anns": instance_anns, 
                "is_negative_example": is_negative_example or not instances
            }
            
            # Add panoptic annotation
            if generate_panoptic:
                background_mask = np.ones((height, width), dtype=np.uint8)
                if instances:
                    for instance in instances:
                        background_mask[instance.mask == 1] = 0
                
                panoptic_segments.append({
                    "id": self.background_id, 
                    "category_id": self.background_category_id,
                    "area": float(background_mask.sum()), 
                    "bbox": [0, 0, width, height], 
                    "iscrowd": 0
                })
                
                panoptic_ann = {
                    "image_id": image_id, 
                    "file_name": f"{tile_id}_{actual_date}.png", 
                    "segments_info": panoptic_segments
                }
                result.update({"panoptic_ann": panoptic_ann, "background_mask": background_mask})
                
            return result
            
        except Exception as e:
            raise type(e)(f"Error processing {tile_id}_{date}: {str(e)}") from e

    def _generate_and_save_images(self, result: Dict[str, Any]):
        """Generate and save panoptic/semantic images and copy source image with normalization."""
        try:
            image_info = result["image_info"]
            image_path = Path(result["image_path"])
            tile_id = image_info["tile_id"]

            # Extract actual date from image filename for consistent naming
            actual_date = self._extract_date_from_image_filename(image_path.name)
            if not actual_date:
                # Fallback to date from image_info if we can't extract from filename
                actual_date = image_info["date"]

            # Use standardized filename: {tile_id}_{YYMMMDD}.tif
            # This matches the panoptic/semseg naming convention
            standardized_filename = f"{tile_id}_{actual_date}.tif"
            img_output_path = self.output_dir / self.split / standardized_filename
            panoptic_output_path = self.output_dir / f"panoptic_{self.split}" / f"{tile_id}_{actual_date}.png"
            semseg_output_path = self.output_dir / f"panoptic_semseg_{self.split}" / f"{tile_id}_{actual_date}.png"
            
            # Load and normalize the satellite imagery to 0-10000
            # Using min/max linear stretch to preserve ALL data (no clipping)
            with rio.open(image_path) as src:
                # Read all bands
                raw = src.read()
                
                nodata_val = src.nodata if src.nodata is not None else 0
                
                normalized = np.empty_like(raw, dtype=np.uint16)
                for b in range(raw.shape[0]):
                    band = raw[b].astype(np.float32)
                    mask = (band == nodata_val) | np.isnan(band)
                    valid = band[~mask]
                    
                    if valid.size == 0:
                        normalized[b] = np.zeros_like(band, dtype=np.uint16)
                        continue
                    
                    # Simple linear stretch using actual min/max (no clipping)
                    # This preserves ALL data including bright features (roofs, tree crowns)
                    vmin, vmax = valid.min(), valid.max()
                    if vmax - vmin < 1e-6:
                        # Handle edge case where all values are the same
                        norm = np.clip(band - vmin, 0, None)
                    else:
                        # Linear stretch from min to max
                        norm = (band - vmin) / (vmax - vmin)
                    
                    norm = np.clip(norm, 0, 1) * 10000.0
                    norm = norm.astype(np.uint16)
                    norm[mask] = 0
                    normalized[b] = norm
                
                # Save normalized image
                with rio.open(
                    img_output_path,
                    'w',
                    driver='GTiff',
                    height=src.height,
                    width=src.width,
                    count=src.count,
                    dtype=np.uint16,
                    crs=src.crs,
                    transform=src.transform,
                    nodata=0
                ) as dst:
                    dst.write(normalized)

            if "panoptic_ann" in result:
                height, width = image_info["height"], image_info["width"]
                instances = result["instances"]
                
                panoptic_seg = np.zeros((height, width, 3), dtype=np.uint8)
                semantic_seg = np.zeros((height, width), dtype=np.uint8)

                # Fill with background color and semantic ID
                semantic_seg.fill(1)  # Background class ID
                for i in range(3):
                    panoptic_seg[:, :, i] = self.background_color[i]

                # Add instances
                if instances:
                    for instance in instances:
                        mask = instance.mask == 1
                        semantic_seg[mask] = 0  # Field class ID
                        if instance.color is not None:
                            panoptic_seg[mask] = instance.color
                
                Image.fromarray(panoptic_seg).save(panoptic_output_path)
                Image.fromarray(semantic_seg).save(semseg_output_path)
                
        except Exception as e:
            warnings.warn(f"Error generating images for {image_info['tile_id']}_{image_info['date']}: {str(e)}")

    def _process_task_wrapper(self, args):
        """Wrapper function for processing a single task (must be a method for pickling)."""
        tile_id, date, annotations_gdf, image_id, generate_panoptic, is_negative = args
        try:
            result = self._process_single_image(tile_id, date, annotations_gdf, image_id, generate_panoptic, is_negative)
            if result:
                self._generate_and_save_images(result)
                return result
        except Exception as e:
            warnings.warn(f"Error processing tile_id {tile_id}, date {date}: {e}")
        return None

    def convert_dataset(self, generate_panoptic: bool = True):
        """Convert the entire dataset including negative examples."""
        start_time = time.time()
        
        # Load annotations and identify annotated pairs
        all_annotations_gdf, tile_ids_with_annotations, annotated_pairs = self._load_annotations()
        
        # Identify negative examples
        negative_examples = self._identify_negative_examples(annotated_pairs)
        
        # Combine positive and negative examples
        all_tasks = []
        
        # Add positive examples (images with annotations)
        for tile_id in tile_ids_with_annotations:
            tile_annotations = all_annotations_gdf[all_annotations_gdf['tile_id'].astype(str) == str(tile_id)]
            for date in tile_annotations['date'].unique():
                date_str = str(date).strip()
                all_tasks.append((tile_id, date_str, tile_annotations, False))  # False = not negative
        
        # Add negative examples
        for tile_id, date in negative_examples:
            all_tasks.append((tile_id, date, gpd.GeoDataFrame(), True))  # True = is negative
        
        if self.verbose:
            print(f"Processing {len(all_tasks)} total images ({len(tile_ids_with_annotations)} with annotations, {len(negative_examples)} negative examples)")
        
        all_image_infos, all_instance_anns, all_panoptic_anns = [], [], []
        positive_count = 0
        negative_count = 0

        # Process all tasks
        tasks_with_ids = [(tile_id, date, annotations_gdf, self._get_next_image_id(), generate_panoptic, is_negative) 
                         for tile_id, date, annotations_gdf, is_negative in all_tasks]
        
        # Use single-threaded processing to avoid pickle issues
        for task in tqdm(tasks_with_ids, desc=f"Processing {self.split} split"):
            try:
                result = self._process_task_wrapper(task)
                if result:
                    all_image_infos.append(result["image_info"])
                    all_instance_anns.extend(result["instance_anns"])
                    if generate_panoptic and "panoptic_ann" in result:
                        all_panoptic_anns.append(result["panoptic_ann"])
                    
                    if result.get("is_negative_example", False):
                        negative_count += 1
                    else:
                        positive_count += 1
            except Exception as e:
                warnings.warn(f"A task failed: {e}")

        if self.verbose:
            print(f"Successfully processed {len(all_image_infos)} images ({positive_count} positive, {negative_count} negative)")

        # Save instance annotations
        instances_dict = {
            "images": all_image_infos, 
            "annotations": all_instance_anns,
            "categories": [
                {
                    "id": cat.id,
                    "name": cat.name,
                    "supercategory": cat.supercategory,
                    "isthing": 1 if cat.isthing else 0,
                    "color": cat.color
                }
                for cat in self.categories if cat.isthing
            ]
        }
        instance_path = self.output_dir / "annotations" / f"instances_{self.split}.json"
        with open(instance_path, "w") as f:
            json.dump(instances_dict, f, indent=2)
        if self.verbose:
            print(f"Saved {len(all_image_infos)} images and {len(all_instance_anns)} instances to {instance_path}")
            
        # Save panoptic annotations
        if generate_panoptic:
            panoptic_dict = {
                "images": all_image_infos, 
                "annotations": all_panoptic_anns,
                "categories": [
                    {
                        "id": cat.id,
                        "name": cat.name,
                        "supercategory": cat.supercategory,
                        "isthing": 1 if cat.isthing else 0,
                        "color": cat.color
                    }
                    for cat in self.categories
                ]
            }
            panoptic_path = self.output_dir / "annotations" / f"panoptic_{self.split}.json"
            with open(panoptic_path, "w") as f:
                json.dump(panoptic_dict, f, indent=2)
            if self.verbose:
                print(f"Saved {len(all_panoptic_anns)} panoptic annotations to {panoptic_path}")
                
        if self.verbose:
            print(f"\nConversion completed in {time.time() - start_time:.2f} seconds")
            
        # Generate summary report of date-matching issues
        self._generate_date_matching_summary()

    def _generate_date_matching_summary(self):
        """Generate a summary report of date-matching issues encountered during conversion."""
        if not self.verbose:
            return
            
        print("\n" + "="*80)
        print("DATE-MATCHING ISSUES SUMMARY")
        print("="*80)
        
        # Summary statistics
        total_issues = len(self.date_matching_issues)
        total_alternative_dates = len(self.alternative_date_used)
        total_fallbacks = len(self.fallback_images_used)
        
        print(f"Total images with date-matching issues: {total_issues}")
        print(f"Images resolved with alternative date interpretation: {total_alternative_dates}")
        print(f"Images using fallback (most recent) images: {total_fallbacks}")
        
        if total_alternative_dates > 0:
            print(f"\nAlternative date interpretations used:")
            print("-" * 60)
            for issue in self.alternative_date_used:
                print(f"  Tile {issue['tile_id']}: {issue['original_date']} -> {issue['alternative_date']} ({issue['found_image']})")
        
        if total_fallbacks > 0:
            print(f"\nFallback images used:")
            print("-" * 60)
            for issue in self.fallback_images_used:
                print(f"  Tile {issue['tile_id']}: {issue['original_date']} -> {issue['fallback_image']}")
        
        if total_issues > 0:
            print(f"\nUnresolved date-matching issues:")
            print("-" * 60)
            for issue in self.date_matching_issues:
                print(f"  Tile {issue['tile_id']}: {issue['date']} - {issue['issue']}")
        
        print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GeoJSON annotations to COCO format with negative examples.")
    parser.add_argument("--geojson_files", required=True, nargs='+', help="Paths to input GeoJSON files.")
    parser.add_argument("--image_dir", required=True, help="Directory containing the source images.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the COCO-formatted dataset.")
    parser.add_argument("--annotations_csv", help="Path to cleaned annotations CSV file.")
    parser.add_argument("--no_fields_csv", help="Path to no-fields CSV file.")
    parser.add_argument("--split", default="train", help="Name of the dataset split (e.g., train, val).")
    parser.add_argument("--min_area", type=int, default=10, help="Minimum area of instances to include (pixels).")
    parser.add_argument("--num_workers", type=int, help="Number of worker processes.")
    parser.add_argument("--no_panoptic", action="store_true", help="Skip panoptic segmentation generation.")
    parser.add_argument("--sample", type=float, default=1.0, help="Fraction of images to process (for debugging).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output.")
    
    args = parser.parse_args()
    
    converter = GeoJSONToCOCOConverter(
        geojson_files=args.geojson_files,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        annotations_csv=args.annotations_csv,
        no_fields_csv=args.no_fields_csv,
        split=args.split,
        min_area=args.min_area,
        num_workers=args.num_workers,
        sample_fraction=args.sample,
        random_seed=args.seed,
        verbose=not args.quiet,
    )
    
    converter.convert_dataset(generate_panoptic=not args.no_panoptic)