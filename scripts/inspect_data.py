"""Script to inspect meteorological datasets (IMD gridded rainfall, ERA5, or test fixtures).

Usage:
    python scripts/inspect_data.py [--file path/to/dataset.nc] [--var rain]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from monsooncast.data.loaders import generate_synthetic_monsoon_dataset, standardize_dataset_coords
from monsooncast.data.validation import audit_missing_values


def main():
    parser = argparse.ArgumentParser(description="Inspect NetCDF meteorological dataset.")
    parser.add_argument("--file", type=str, default=None, help="Path to NetCDF file")
    parser.add_argument("--var", type=str, default="rain", help="Rainfall variable name")
    parser.add_argument("--generate-test-fixture", action="store_true", help="Generate synthetic test NetCDF")
    args = parser.parse_args()

    file_path = args.file
    if file_path is None or not Path(file_path).exists():
        default_synth = Path("data/raw/synthetic_sample.nc")
        if not default_synth.exists() or args.generate_test_fixture:
            print("No real IMD file supplied. Generating synthetic test fixture at data/raw/synthetic_sample.nc...")
            ds = generate_synthetic_monsoon_dataset(
                output_path=default_synth,
                start_year=2010,
                end_year=2023,
                lat_bounds=(18.0, 25.0),
                lon_bounds=(73.0, 82.0),
                resolution=0.5
            )
            file_path = str(default_synth)
        else:
            file_path = str(default_synth)

    print("\n=======================================================")
    print("      MONSOONCAST METEOROLOGICAL DATA INSPECTION       ")
    print("=======================================================")
    print(f"Target File: {file_path}")

    ds = xr.open_dataset(file_path)
    ds = standardize_dataset_coords(ds)
    var_name = args.var if args.var in ds else list(ds.data_vars)[0]

    print("\n[1] DATASET METADATA:")
    for k, v in ds.attrs.items():
        print(f"  - {k}: {v}")

    print("\n[2] COORDINATES & DIMENSIONS:")
    for dim, size in ds.sizes.items():
        print(f"  - {dim}: {size}")

    if "lat" in ds.coords:
        lats = ds["lat"].values
        print(f"  - Latitude range: [{lats.min():.2f}, {lats.max():.2f}] (step={np.mean(np.diff(lats)):.2f})")
    if "lon" in ds.coords:
        lons = ds["lon"].values
        print(f"  - Longitude range: [{lons.min():.2f}, {lons.max():.2f}] (step={np.mean(np.diff(lons)):.2f})")
    if "time" in ds.coords:
        times = ds["time"].values
        print(f"  - Temporal range: {str(times[0])[:10]} to {str(times[-1])[:10]} ({len(times)} days)")

    print(f"\n[3] VARIABLE AUDIT ('{var_name}'):")
    val_report = audit_missing_values(ds, var_name=var_name)
    for k, v in val_report.items():
        print(f"  - {k}: {v}")

    data_arr = ds[var_name].values
    valid_data = data_arr[~np.isnan(data_arr) & (data_arr != -999.0)]
    if len(valid_data) > 0:
        print("\n[4] METEOROLOGICAL STATISTICS (Valid grid-days):")
        print(f"  - Mean daily rainfall: {np.mean(valid_data):.2f} mm/day")
        print(f"  - Median daily rainfall: {np.median(valid_data):.2f} mm/day")
        print(f"  - Std deviation: {np.std(valid_data):.2f} mm/day")
        print(f"  - Maximum observed: {np.max(valid_data):.2f} mm/day")
        print(f"  - Fraction >= 50 mm/day (Heavy): {np.mean(valid_data >= 50.0) * 100:.2f}%")
        print(f"  - Fraction >= 100 mm/day (Very Heavy): {np.mean(valid_data >= 100.0) * 100:.2f}%")

    print("\n=======================================================")
    print("Inspection complete. Data integrity verified.")


if __name__ == "__main__":
    main()
