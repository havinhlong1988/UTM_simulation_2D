#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export path footprint outputs:
  CSV
  XYZ
  GeoJSON
  KML
  summary JSON
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def export_path_outputs(
    model: pd.DataFrame,
    path_indices: list[int],
    output_dir: Path,
    path_name: str,
    algorithm_name: str,
    result: dict,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path_df = model.loc[path_indices].copy()
    path_df = path_df.reset_index().rename(columns={"node_index": "original_node_index"})

    if "index" in path_df.columns and "original_node_index" not in path_df.columns:
        path_df = path_df.rename(columns={"index": "original_node_index"})

    path_df.insert(0, "path_step", np.arange(len(path_df), dtype=int))

    path_df["algorithm"] = algorithm_name
    # Add simple cumulative distance along exported path.
    path_df = add_cumulative_path_columns(path_df)

    csv_file = output_dir / f"{path_name}.csv"
    xyz_file = output_dir / f"{path_name}.xyz"
    geojson_file = output_dir / f"{path_name}.geojson"
    kml_file = output_dir / f"{path_name}.kml"
    summary_file = output_dir / f"{path_name}_summary.json"

    export_csv(path_df, csv_file)
    export_xyz(path_df, xyz_file)
    export_geojson(path_df, geojson_file, path_name, result)
    export_kml(path_df, kml_file, path_name)
    export_summary(path_df, summary_file, path_name, algorithm_name, result)

    return {
        "csv": csv_file,
        "xyz": xyz_file,
        "geojson": geojson_file,
        "kml": kml_file,
        "summary": summary_file,
    }

def add_cumulative_path_columns(path_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add approximate cumulative distance in coordinate units.

    Note:
    This is a simple lon/lat-aware distance approximation.
    It does not include FLZ cost. Detailed traveltime is in summary JSON.
    """

    out = path_df.copy()

    x = out["x"].astype(float).values
    y = out["y"].astype(float).values
    z = out["z"].astype(float).values

    is_lonlat = (
        -180 <= np.nanmin(x) <= 180
        and -180 <= np.nanmax(x) <= 180
        and -90 <= np.nanmin(y) <= 90
        and -90 <= np.nanmax(y) <= 90
    )

    if is_lonlat:
        lon0 = float(np.nanmean(x))
        lat0 = float(np.nanmean(y))
        lat0_rad = np.deg2rad(lat0)

        meters_per_deg_lat = 111_320.0
        meters_per_deg_lon = 111_320.0 * np.cos(lat0_rad)

        xm = (x - lon0) * meters_per_deg_lon
        ym = (y - lat0) * meters_per_deg_lat
        zm = z
    else:
        xm = x
        ym = y
        zm = z

    dist = np.zeros(len(out), dtype=float)

    for i in range(1, len(out)):
        dx = xm[i] - xm[i - 1]
        dy = ym[i] - ym[i - 1]
        dz = zm[i] - zm[i - 1]
        dist[i] = np.sqrt(dx * dx + dy * dy + dz * dz)

    out["segment_distance_m"] = dist
    out["cumulative_distance_m"] = np.cumsum(dist)
    out["cumulative_distance_km"] = out["cumulative_distance_m"] / 1000.0

    return out

def export_csv(path_df: pd.DataFrame, csv_file: Path):
    keep_cols = [
        "path_step",
        "original_node_index",
        "x",
        "y",
        "z",
        "slowness",
        "label",
        "label_prefix",
        "segment_distance_m",
        "cumulative_distance_m",
        "cumulative_distance_km",
        "algorithm",
    ]

    cols = [c for c in keep_cols if c in path_df.columns]
    path_df[cols].to_csv(csv_file, index=False)


def export_xyz(path_df: pd.DataFrame, xyz_file: Path):
    """
    XYZ format:
      x y z slowness label path_step
    """
    with open(xyz_file, "w", encoding="utf-8") as f:
        f.write("# x y z slowness label path_step\n")

        for _, row in path_df.iterrows():
            f.write(
                f"{row['x']:.10f} "
                f"{row['y']:.10f} "
                f"{row['z']:.6f} "
                f"{row['slowness']:.10g} "
                f"{row['label']} "
                f"{int(row['path_step'])}\n"
            )


def export_geojson(path_df: pd.DataFrame, geojson_file: Path, path_name: str, result: dict):
    """
    GeoJSON LineString.

    Coordinates are written as:
      [x, y, z]

    If x/y are lon/lat, this can be opened directly in QGIS.
    """
    coordinates = [
        [
            float(row["x"]),
            float(row["y"]),
            float(row["z"]),
        ]
        for _, row in path_df.iterrows()
    ]

    feature = {
        "type": "Feature",
        "properties": {
            "name": path_name,
            "algorithm": result.get("algorithm", None),
            "success": result.get("success", None),
            "total_cost": result.get("total_cost", None),
            "path_nodes": int(len(path_df)),
            "expanded_nodes": result.get("expanded_nodes", None),
            "runtime_seconds": result.get("runtime_seconds", None),
        },
        "geometry": {
            "type": "LineString",
            "coordinates": coordinates,
        },
    }

    data = {
        "type": "FeatureCollection",
        "features": [feature],
    }

    geojson_file.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )


def export_kml(path_df: pd.DataFrame, kml_file: Path, path_name: str):
    """
    KML LineString.

    KML coordinate order:
      lon,lat,alt

    This assumes x=longitude and y=latitude.
    If your model x/y are UTM meters, use GeoJSON/CSV instead,
    or convert to lon/lat before KML export.
    """
    coords_text = "\n".join(
        f"{float(row['x']):.10f},{float(row['y']):.10f},{float(row['z']):.3f}"
        for _, row in path_df.iterrows()
    )

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{path_name}</name>

    <Style id="path_style">
      <LineStyle>
        <color>ff0000ff</color>
        <width>4</width>
      </LineStyle>
    </Style>

    <Placemark>
      <name>{path_name}</name>
      <styleUrl>#path_style</styleUrl>
      <LineString>
        <tessellate>1</tessellate>
        <altitudeMode>absolute</altitudeMode>
        <coordinates>
{coords_text}
        </coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>
"""

    kml_file.write_text(kml, encoding="utf-8")


def export_summary(
    path_df: pd.DataFrame,
    summary_file: Path,
    path_name: str,
    algorithm_name: str,
    result: dict,
):
    summary = {
        "path_name": path_name,
        "algorithm": algorithm_name,
        "success": result.get("success", None),
        "message": result.get("message", None),
        "path_nodes": int(len(path_df)),
        "total_cost": result.get("total_cost", None),
        "expanded_nodes": result.get("expanded_nodes", None),
        "visited_nodes": result.get("visited_nodes", None),
        "runtime_seconds": result.get("runtime_seconds", None),
        "start": {
            "x": float(path_df.iloc[0]["x"]),
            "y": float(path_df.iloc[0]["y"]),
            "z": float(path_df.iloc[0]["z"]),
            "label": str(path_df.iloc[0]["label"]),
        },
        "end": {
            "x": float(path_df.iloc[-1]["x"]),
            "y": float(path_df.iloc[-1]["y"]),
            "z": float(path_df.iloc[-1]["z"]),
            "label": str(path_df.iloc[-1]["label"]),
        },
        "algorithm_path_distance_m": result.get("algorithm_path_distance_m", None),
        "algorithm_path_distance_km": result.get("algorithm_path_distance_km", None),
        "algorithm_estimated_traveltime_s": result.get("algorithm_estimated_traveltime_s", None),
        "algorithm_estimated_traveltime_min": result.get("algorithm_estimated_traveltime_min", None),

        "output_path_distance_m": result.get("output_path_distance_m", None),
        "output_path_distance_km": result.get("output_path_distance_km", None),
        "output_estimated_traveltime_s": result.get("output_estimated_traveltime_s", None),
        "output_estimated_traveltime_min": result.get("output_estimated_traveltime_min", None),
    }

    summary_file.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )