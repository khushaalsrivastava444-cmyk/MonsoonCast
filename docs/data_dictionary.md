# MonsoonCast Data Dictionary

## 1. Grid Coordinates & Dimensions

| Dimension | Description | Standard Values / Range | Coordinate Name |
| :--- | :--- | :--- | :--- |
| `time` | Observation timestamp (UTC / IST 08:30) | Daily frequency (e.g. 2010-06-01 to 2023-09-30) | `time` |
| `lat` | Latitude (degrees north) | 6.5°N to 38.5°N (0.25° spacing) | `lat` or `latitude` |
| `lon` | Longitude (degrees east) | 66.5°E to 100.5°E (0.25° spacing) | `lon` or `longitude` |

## 2. Variables & Physical Predictors

### Target Variables
- `rainfall`: 24-hour accumulated rainfall (mm/day). Range: $[0.0, 1000.0+]$.
- `extreme_event`: Binary indicator $\mathbb{I}(\text{rainfall}_{t+1} \ge \tau)$.
  - Values: `0` (Non-extreme), `1` (Extreme rainfall event).

### Engineered Antecedent Features
- `rain_lag1`: 1-day lagged rainfall $R(s, t)$ in mm/day.
- `rain_lag2`: 2-day lagged rainfall $R(s, t-1)$ in mm/day.
- `rain_lag3`: 3-day lagged rainfall $R(s, t-2)$ in mm/day.
- `rain_lag7`: 7-day lagged rainfall $R(s, t-6)$ in mm/day.
- `rain_roll_sum3`: Trailing 3-day cumulative rainfall $\sum_{k=0}^2 R(s, t-k)$ in mm.
- `rain_roll_sum7`: Trailing 7-day cumulative rainfall $\sum_{k=0}^6 R(s, t-k)$ in mm.
- `rain_roll_max7`: Trailing 7-day maximum daily rainfall in mm/day.
- `rain_roll_mean7`: Trailing 7-day mean daily rainfall in mm/day.
- `rain_spatial_mean3x3`: Mean rainfall of 8 neighboring grid cells on day $t$ (mm/day).
- `rain_spatial_max3x3`: Maximum rainfall among 8 neighboring grid cells on day $t$ (mm/day).
- `climatology_mean`: Historical daily rainfall expectation for given (lat, lon, DOY) computed on train set.
- `doy_sin`: Sine component of seasonal day of year $\sin(2\pi \cdot \text{DOY} / 365.25)$.
- `doy_cos`: Cosine component of seasonal day of year $\cos(2\pi \cdot \text{DOY} / 365.25)$.

### Atmospheric Reanalysis Variables (ERA5 Mode)
- `z500`: Geopotential height at 500 hPa ($m^2/s^2$ or geopotential meters).
- `u850`, `v850`: Zonal and meridional winds at 850 hPa (m/s) capturing low-level monsoon jet.
- `u200`, `v200`: Zonal and meridional winds at 200 hPa (m/s) capturing tropical easterly jet.
- `q850`: Specific humidity at 850 hPa (kg/kg).
- `tcwv`: Total column water vapour ($kg/m^2$ or mm).
- `msl`: Mean sea level pressure (Pa).

## 3. Missing Value Conventions
- Missing / Non-monitored flag in IMD NetCDF: `-999.0`
- Handled via masking and imputation during preprocessing.
- Ocean grid points outside IMD land boundary are masked out to prevent distorting empirical distributions.
