"""
Semantic 3-class to vector and geographic postprocessing for Uganda UNet inference.

Output schema matches Mask2Former pipeline (segment_id, confidence, score, category_id, area)
so ag-seg downstream tools (aggregate_cell_predictions, compute_field_statistics, etc.) work unchanged.
No dependency on ag-seg.
"""

import logging
import math
from typing import Optional, Tuple

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import box, shape
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


def _polygon_area_m2(geom, crs: Optional[CRS]) -> float:
    """Return geometry area in square meters (convert from deg² if geographic CRS)."""
    area = geom.area
    if crs is None:
        return area
    is_geographic = (
        str(crs).startswith("EPSG:4326")
        or (hasattr(crs, "is_geographic") and crs.is_geographic)
    )
    if not is_geographic:
        return area
    center_lat = (geom.bounds[1] + geom.bounds[3]) / 2.0
    meters_per_degree_lat = 111320.0
    meters_per_degree_lon = 111320.0 * math.cos(math.radians(abs(center_lat)))
    return area * (meters_per_degree_lat * meters_per_degree_lon)


def semantic_3class_to_geodataframe(
    seg: np.ndarray,
    transform: rasterio.Affine,
    crs: CRS,
    min_area: float = 100.0,
    max_area: Optional[float] = 90000.0,
    merge_boundary_into_field: bool = True,
) -> gpd.GeoDataFrame:
    """
    Convert 3-class semantic map (0=bg, 1=field, 2=boundary) to GeoDataFrame of field polygons.

    Uses connected components on the field class; each component becomes one polygon.
    Property schema matches Mask2Former: segment_id, confidence, score, category_id, area.
    """
    # Field mask: class 1, or 1+2 if merging boundary
    if merge_boundary_into_field:
        field_mask = (seg == 1) | (seg == 2)
    else:
        field_mask = seg == 1
    field_mask = field_mask.astype(np.uint8)

    labeled, num_features = ndimage.label(field_mask)
    if num_features == 0:
        return gpd.GeoDataFrame(columns=["geometry"], crs=crs)

    features = []
    for seg_id in range(1, num_features + 1):
        mask = (labeled == seg_id).astype(np.uint8)
        for geom_dict, _ in shapes(mask, transform=transform):
            geom = shape(geom_dict)
            if geom.is_empty or geom.area == 0:
                continue
            area_m2 = _polygon_area_m2(geom, crs)
            if area_m2 < min_area:
                continue
            if max_area is not None and area_m2 > max_area:
                continue
            if not geom.is_valid:
                geom = geom.buffer(0)
            if geom.is_empty:
                continue
            features.append({
                "geometry": geom,
                "segment_id": seg_id,
                "category_id": 0,
                "confidence": 1.0,
                "score": 1.0,
                "mask_score": 1.0,
                "isthing": True,
                "area": area_m2,
                "touches_edge": False,
            })

    if not features:
        return gpd.GeoDataFrame(columns=["geometry"], crs=crs)
    gdf = gpd.GeoDataFrame(features, crs=crs)
    return gdf


def filter_edge_polygons_geo(
    polygons_gdf: gpd.GeoDataFrame,
    chip_bounds: box,
    edge_buffer: float = 5.0,
) -> gpd.GeoDataFrame:
    """Filter out polygons that are too close to chip edges (geographic coordinates)."""
    if polygons_gdf.empty:
        return polygons_gdf
    crs = polygons_gdf.crs
    is_geographic = crs is not None and (
        str(crs).startswith("EPSG:4326")
        or (hasattr(crs, "is_geographic") and crs.is_geographic)
    )
    if is_geographic:
        center_lat = (chip_bounds.bounds[1] + chip_bounds.bounds[3]) / 2.0
        meters_per_degree_lat = 111320.0
        meters_per_degree_lon = 111320.0 * math.cos(math.radians(center_lat))
        avg_meters_per_degree = (meters_per_degree_lat + meters_per_degree_lon) / 2.0
        edge_buffer_degrees = edge_buffer / avg_meters_per_degree
        inner_boundary = chip_bounds.buffer(-edge_buffer_degrees)
    else:
        inner_boundary = chip_bounds.buffer(-edge_buffer)
    within_mask = polygons_gdf.geometry.within(inner_boundary)
    return polygons_gdf[within_mask].copy()


def resolve_overlaps(
    polygons_gdf: gpd.GeoDataFrame,
    overlap_threshold: float = 0.7,
    keep_higher_confidence: bool = True,
) -> gpd.GeoDataFrame:
    """Resolve overlapping polygons: keep higher-confidence when IoU > threshold."""
    if polygons_gdf.empty or len(polygons_gdf) == 1:
        return polygons_gdf
    to_remove = set()
    spatial_index = polygons_gdf.sindex
    for idx, row in polygons_gdf.iterrows():
        if idx in to_remove:
            continue
        bounds = row.geometry.bounds
        possible_matches = list(spatial_index.intersection(bounds))
        current_confidence = row.get("confidence", 0.0)
        if isinstance(current_confidence, dict):
            current_confidence = current_confidence.get("confidence", 0.0)
        n = len(polygons_gdf)
        for match_idx in possible_matches:
            if match_idx >= n:
                continue
            match_label = polygons_gdf.index[match_idx]
            if match_label == idx or match_label in to_remove:
                continue
            match_row = polygons_gdf.iloc[match_idx]
            intersection = row.geometry.intersection(match_row.geometry)
            union = row.geometry.union(match_row.geometry)
            if union.area > 0:
                iou = intersection.area / union.area
                if iou > overlap_threshold:
                    match_confidence = match_row.get("confidence", 0.0)
                    if isinstance(match_confidence, dict):
                        match_confidence = match_confidence.get("confidence", 0.0)
                    if keep_higher_confidence:
                        if match_confidence > current_confidence:
                            to_remove.add(idx)
                            break
                        else:
                            to_remove.add(match_label)
                    else:
                        to_remove.add(match_label)
    if to_remove:
        return polygons_gdf.drop(index=list(to_remove)).copy()
    return polygons_gdf


def merge_overlapping_fields(
    polygons_gdf: gpd.GeoDataFrame,
    iou_threshold: float = 0.7,
) -> gpd.GeoDataFrame:
    """Merge overlapping fields (union geometry, average confidence)."""
    if polygons_gdf.empty or len(polygons_gdf) == 1:
        return polygons_gdf
    processed = set()
    merged_features = []
    spatial_index = polygons_gdf.sindex
    for idx, row in polygons_gdf.iterrows():
        if idx in processed:
            continue
        bounds = row.geometry.bounds
        possible_matches = list(spatial_index.intersection(bounds))
        overlapping_geometries = []
        overlapping_confidences = []
        n = len(polygons_gdf)
        for match_idx in possible_matches:
            if match_idx >= n:
                continue
            match_label = polygons_gdf.index[match_idx]
            if match_label in processed:
                continue
            match_row = polygons_gdf.iloc[match_idx]
            intersection = row.geometry.intersection(match_row.geometry)
            union = row.geometry.union(match_row.geometry)
            if union.area > 0:
                iou = intersection.area / union.area
                if iou > iou_threshold:
                    overlapping_geometries.append(match_row.geometry)
                    match_conf = match_row.get("confidence", 0.0)
                    if isinstance(match_conf, dict):
                        match_conf = match_conf.get("confidence", 0.0)
                    overlapping_confidences.append(match_conf)
                    processed.add(match_label)
        if overlapping_geometries:
            merged_geometry = unary_union([row.geometry] + overlapping_geometries)
            current_conf = row.get("confidence", 0.0)
            if isinstance(current_conf, dict):
                current_conf = current_conf.get("confidence", 0.0)
            avg_confidence = np.mean([current_conf] + overlapping_confidences)
            merged_feature = row.copy()
            merged_feature.geometry = merged_geometry
            merged_feature["confidence"] = avg_confidence
            merged_feature["area"] = _polygon_area_m2(merged_geometry, polygons_gdf.crs)
            merged_feature["merged_count"] = len(overlapping_geometries) + 1
            merged_features.append(merged_feature)
        else:
            merged_features.append(row)
    if not merged_features:
        return polygons_gdf
    return gpd.GeoDataFrame(merged_features, crs=polygons_gdf.crs)
