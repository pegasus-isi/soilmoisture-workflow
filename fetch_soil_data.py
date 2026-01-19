#!/usr/bin/env python3

"""
Fetch soil moisture data from Open-Meteo (ERA5/ERA5-Land).

This script fetches hourly soil moisture and temperature data for
polygon centroids and outputs a CSV compatible with the workflow.

Usage:
    ./fetch_soil_data.py --polygons-file polygons.json \
        --polygon-id field1 --start-date 2024-01-01 \
        --end-date 2024-01-31 --output soil_data.csv
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/era5"


def ensure_closed_polygon(coords: List[List[float]]) -> List[List[float]]:
    """Ensure polygon coordinates are closed (first point equals last)."""
    if not coords:
        return coords
    if coords[0] != coords[-1]:
        return coords + [coords[0]]
    return coords


def polygon_centroid(coords: List[List[float]]) -> Tuple[float, float]:
    """Approximate centroid as mean of coordinates."""
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return sum(lons) / len(lons), sum(lats) / len(lats)


def load_polygons_file(path: str) -> List[Dict]:
    """Load polygon definitions from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        polygons = data.get("polygons", [])
    else:
        polygons = data
    if not isinstance(polygons, list):
        raise ValueError("Invalid polygons file format")
    for polygon in polygons:
        coords = polygon.get("coordinates", [])
        polygon["coordinates"] = ensure_closed_polygon(coords)
    return polygons


def fetch_open_meteo_hourly(lat: float, lon: float, start_date: datetime,
                            end_date: datetime, endpoint: str) -> Dict:
    """Fetch hourly soil data from Open-Meteo."""
    hourly_vars = [
        "soil_moisture_0_to_7cm",
        "soil_temperature_0_to_7cm",
        "soil_temperature_7_to_28cm",
    ]
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(hourly_vars),
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "timezone": "UTC",
    }
    response = requests.get(endpoint, params=params, timeout=60)
    if response.status_code != 200:
        print(f"Error fetching Open-Meteo data: {response.status_code}")
        print(response.text)
        return {}
    return response.json()


def fetch_open_meteo_data(polygon: Dict, start_date: datetime,
                          end_date: datetime, endpoint: str) -> pd.DataFrame:
    """Fetch Open-Meteo soil data for a polygon (centroid sampling)."""
    coords = polygon.get("coordinates", [])
    coords = ensure_closed_polygon(coords)
    if not coords:
        return pd.DataFrame()

    center_lon, center_lat = polygon_centroid(coords)
    data = fetch_open_meteo_hourly(center_lat, center_lon, start_date, end_date, endpoint)
    hourly = data.get("hourly", {})

    times = hourly.get("time", [])
    moisture = hourly.get("soil_moisture_0_to_7cm", [])
    temp_surface = hourly.get("soil_temperature_0_to_7cm", [])
    temp_7_28 = hourly.get("soil_temperature_7_to_28cm", [])

    if not times:
        return pd.DataFrame()

    records = []
    for idx, ts in enumerate(times):
        try:
            timestamp = datetime.fromisoformat(ts)
        except ValueError:
            continue
        records.append({
            "timestamp": timestamp,
            "polygon_id": polygon.get("id", "polygon"),
            "polygon_name": polygon.get("name", ""),
            "latitude": center_lat,
            "longitude": center_lon,
            "soil_moisture": moisture[idx] if idx < len(moisture) else None,
            "soil_temp_surface": temp_surface[idx] if idx < len(temp_surface) else None,
            "soil_temp_10cm": temp_7_28[idx] if idx < len(temp_7_28) else None,
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("timestamp").drop_duplicates()
    return df


def save_data(df: pd.DataFrame, output_file: str):
    """Save data to CSV."""
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} records to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch soil moisture data from Open-Meteo"
    )

    # Actions
    parser.add_argument('--fetch', action='store_true',
                        help='Fetch soil data for polygon(s)')

    # Data fetching
    parser.add_argument('--polygon-id', type=str, help='Polygon ID')
    parser.add_argument('--polygon-ids', type=str, nargs='+',
                        help='Multiple polygon IDs')
    parser.add_argument('--polygons-file', type=str, required=True,
                        help='JSON file with polygon definitions')
    parser.add_argument('--start-date', type=str,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, default='soil_data.csv',
                        help='Output CSV file')
    parser.add_argument('--open-meteo-endpoint', type=str,
                        default=OPEN_METEO_URL,
                        help='Open-Meteo API endpoint URL')

    args = parser.parse_args()

    if args.fetch or args.polygon_id or args.polygon_ids or args.polygons_file:
        polygon_ids = args.polygon_ids or ([args.polygon_id] if args.polygon_id else [])

        # Parse dates
        if args.start_date:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        else:
            start_date = datetime.now() - timedelta(days=30)

        if args.end_date:
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        else:
            end_date = datetime.now()

        if end_date.date() > datetime.now().date():
            print("Error: end-date must be in the past for Open-Meteo archive data")
            sys.exit(1)

        print(f"Fetching data from {start_date.date()} to {end_date.date()}")

        polygons = load_polygons_file(args.polygons_file)
        polygons_by_id = {p.get("id"): p for p in polygons}
        targets = polygon_ids or list(polygons_by_id.keys())
        if not targets:
            print("Error: no polygon IDs found in polygons file")
            sys.exit(1)

        all_data = []
        for pid in targets:
            polygon = polygons_by_id.get(pid)
            if not polygon:
                print(f"Polygon {pid} not found in {args.polygons_file}")
                continue
            print(f"  Fetching polygon: {pid}")
            df = fetch_open_meteo_data(polygon, start_date, end_date, args.open_meteo_endpoint)
            all_data.append(df)

        combined_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
        save_data(combined_df, args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
