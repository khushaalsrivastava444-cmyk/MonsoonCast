# MonsoonCast Data Governance and Directory Structure

This directory manages meteorological observation datasets, reanalysis products, and processed experimental tensors.

## Directory Layout

```
data/
├── README.md               # Data governance and acquisition documentation
├── raw/                    # Untouched, immutable source files
│   ├── imd/                # IMD daily gridded rainfall NetCDF files
│   ├── era5/               # ERA5 single-level and pressure-level NetCDFs (optional)
│   └── synthetic_sample.nc # Small synthetic sample strictly for test verification
├── interim/                # Cleaned, coordinate-aligned, seasonally filtered data
└── processed/              # Feature matrices (lagged predictors, extreme event labels)
```

## Data Sources

### 1. IMD 0.25° x 0.25° Daily Gridded Rainfall (Dataset A)
- **Provider:** India Meteorological Department (IMD), Pune, National Climate Center (NCC).
- **Source:** [IMD Pune Gridded NetCDF Archive](https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html)
- **Spatial Domain:** 6.5°N – 38.5°N, 66.5°E – 100.5°E (129 x 135 grid points at 0.25° resolution).
- **Temporal Resolution:** Daily accumulated rainfall (08:30 IST to 08:30 IST next day).
- **Units:** mm/day.
- **Missing Value Flag:** `-999.0` (indicates oceanic grid points or unmonitored border areas).
- **Licensing & Attribution:** For educational and non-commercial research use; proper citation of IMD NCC Pune required.

### 2. ERA5 Atmospheric Reanalysis (Dataset B)
- **Provider:** European Centre for Medium-Range Weather Forecasts (ECMWF) via Copernicus Climate Data Store (CDS).
- **Variables:** Geopotential height (500 hPa), U/V zonal & meridional wind (850 hPa & 200 hPa), Specific Humidity (700 hPa & 850 hPa), Total Column Water Vapour (TCWV), Mean Sea Level Pressure (MSLP).
- **Role:** Atmospheric state predictors for retrospective association and multi-variate diagnostic experiments.

### 3. Numerical Weather Prediction Ensemble Forecasts (Dataset C - Extension)
- **Provider:** TIGGE / ECMWF ensemble forecast archives.
- **Temporal Alignment Constraint:** Forecasts must be indexed by issue time ($t_{init}$), lead time ($\Delta t$), and valid time ($t_{valid} = t_{init} + \Delta t$).

## Software Verification Policy: Synthetic vs Real Data

To guarantee end-to-end automated testing without requiring an immediate multi-gigabyte download, the pipeline includes a deterministic synthetic generator (`generate_synthetic_monsoon_dataset`).
- **Synthetic data are strictly marked with `is_synthetic = True` metadata.**
- Synthetic datasets simulate orographic enhancement (Western Ghats), monsoon seasonality (JJAS peak), and zero-inflated Gamma distributions.
- **Scientific findings and publication metrics must NEVER report synthetic data as empirical observations.**
