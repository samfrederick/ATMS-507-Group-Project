#!/usr/bin/env python3

import numpy as np
import pandas as pd
import xarray as xr

# ============================================================
# File paths
# ============================================================
FLUX_FILE = "/scratch/sjacker2/project_data/merra2_rad_downloads/merra2_toa_upwelling_lw_sw_combined.nc"
MONITOR_FILE = "/scratch/sjacker2/project_data/monthly_pm25_eastern_us_1999_2025.csv"

OUTPUT_NC = "/scratch/sjacker2/project_data/merra2_flux_monitor_cells_eastern_us_annual.nc"
OUTPUT_CSV = "/scratch/sjacker2/project_data/merra2_monitor_cell_fluxes_eastern_us_annual.csv"

# ============================================================
# Eastern US bounding box
# ============================================================
minlat = 25
maxlat = 50
minlon = -90
maxlon = -65

YEAR_START = 1997
YEAR_END = 2025

FLUX_VARS = ["TOA_LW_UP", "TOA_SW_UP"]

# ============================================================
# Helper: get grid-cell edges from grid-cell centers
# ============================================================
def get_edges_from_centers(centers):
    """
    Given 1D coordinate centers, return grid-cell edges.
    Works for regular or nearly regular grids.
    """
    centers = np.asarray(centers)
    edges = np.zeros(len(centers) + 1)

    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])

    return edges


# ============================================================
# Helper: assign monitors to grid cells
# ============================================================
def assign_monitors_to_grid(monitors, lats, lons):
    """
    Assign each monitor to a grid cell using lat/lon grid-cell edges.
    Returns dataframe with lat_idx and lon_idx.
    """
    lat_edges = get_edges_from_centers(lats)
    lon_edges = get_edges_from_centers(lons)

    lat_idx = np.digitize(monitors["Latitude"].values, lat_edges) - 1
    lon_idx = np.digitize(monitors["Longitude"].values, lon_edges) - 1

    monitors = monitors.copy()
    monitors["lat_idx"] = lat_idx
    monitors["lon_idx"] = lon_idx

    monitors = monitors[
        (monitors["lat_idx"] >= 0) &
        (monitors["lat_idx"] < len(lats)) &
        (monitors["lon_idx"] >= 0) &
        (monitors["lon_idx"] < len(lons))
    ].copy()

    monitors["grid_lat"] = lats[monitors["lat_idx"].values]
    monitors["grid_lon"] = lons[monitors["lon_idx"].values]

    return monitors


# ============================================================
# Helper: fit trend
# ============================================================
def fit_trend(years, values):
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)

    mask = np.isfinite(years) & np.isfinite(values)

    if mask.sum() < 2:
        return np.array([np.nan, np.nan]), np.full_like(years, np.nan, dtype=float)

    coeffs = np.polyfit(years[mask], values[mask], 1)
    trendline = np.polyval(coeffs, years)

    return coeffs, trendline


# ============================================================
# Read monitor data and get unique monitor locations
# ============================================================
mon = pd.read_csv(MONITOR_FILE)
mon.columns = mon.columns.str.strip()

for col in ["Latitude", "Longitude", "year", "month", "monthly_pm25_mean"]:
    if col in mon.columns:
        mon[col] = pd.to_numeric(mon[col], errors="coerce")

mon = mon.dropna(subset=["Latitude", "Longitude"]).copy()

# Keep only monitors inside eastern US bounding box
mon = mon[
    (mon["Latitude"] >= minlat) &
    (mon["Latitude"] <= maxlat) &
    (mon["Longitude"] >= minlon) &
    (mon["Longitude"] <= maxlon)
].copy()

# Create monitor ID if needed
if "monitor_id" not in mon.columns:
    mon["monitor_id"] = (
        mon["State Code"].astype(str).str.zfill(2) + "-" +
        mon["County Code"].astype(str).str.zfill(3) + "-" +
        mon["Site Num"].astype(str).str.zfill(4) + "-" +
        mon["POC"].astype(str)
    )

monitor_locations = (
    mon[["monitor_id", "Latitude", "Longitude"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

print(f"Unique AQS monitors in eastern US box: {len(monitor_locations):,}")


# ============================================================
# Open flux fields
# ============================================================
ds_flux = xr.open_dataset(FLUX_FILE)[FLUX_VARS]

ds_flux = ds_flux.sel(
    time=slice(f"{YEAR_START}-01-01", f"{YEAR_END}-12-31")
)

print("\nOpened flux dataset:")
print(ds_flux)

lats = ds_flux["lat"].values
lons = ds_flux["lon"].values

monitor_locations_for_grid = monitor_locations.copy()

# If MERRA-2 longitudes are 0–360, convert monitor longitudes too
if np.nanmax(lons) > 180:
    monitor_locations_for_grid["Longitude"] = monitor_locations_for_grid["Longitude"] % 360


# ============================================================
# Assign monitors to flux grid cells
# ============================================================
monitor_grid = assign_monitors_to_grid(
    monitor_locations_for_grid,
    lats,
    lons
)

# Keep original longitude for reference
monitor_grid["monitor_lon_original"] = monitor_locations.loc[
    monitor_grid.index, "Longitude"
].values

print(f"\nMonitors successfully assigned to flux grid cells: {len(monitor_grid):,}")

unique_cells = (
    monitor_grid[["lat_idx", "lon_idx", "grid_lat", "grid_lon"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

print(f"Unique grid cells containing monitors: {len(unique_cells):,}")

cell_counts = (
    monitor_grid
    .groupby(["lat_idx", "lon_idx", "grid_lat", "grid_lon"], as_index=False)
    .agg(n_monitors=("monitor_id", "nunique"))
)

print("\nFirst few monitor-cell counts:")
print(cell_counts.head())


# ============================================================
# Create a mask for cells that contain monitors
# ============================================================
cell_mask = xr.zeros_like(ds_flux["TOA_SW_UP"].isel(time=0), dtype=bool)

for _, row in unique_cells.iterrows():
    cell_mask.values[
        int(row["lat_idx"]),
        int(row["lon_idx"])
    ] = True

toa_sw_monitor_cells = ds_flux["TOA_SW_UP"].where(cell_mask)
toa_lw_monitor_cells = ds_flux["TOA_LW_UP"].where(cell_mask)


# ============================================================
# Area-weighted average over monitor-containing cells only
# ============================================================
weights = np.cos(np.deg2rad(ds_flux["lat"]))
weights.name = "weights"

sw_monitor_cell_mean = (
    toa_sw_monitor_cells
    .weighted(weights)
    .mean(dim=("lat", "lon"), skipna=True)
)

lw_monitor_cell_mean = (
    toa_lw_monitor_cells
    .weighted(weights)
    .mean(dim=("lat", "lon"), skipna=True)
)

sw_monitor_cell_mean.name = "TOA_SW_UP_mean"
sw_monitor_cell_mean.attrs["units"] = ds_flux["TOA_SW_UP"].attrs.get("units", "W m-2")
sw_monitor_cell_mean.attrs["description"] = (
    "Area-weighted mean TOA upwelling shortwave flux using only eastern US grid cells "
    "that contain at least one AQS monitor"
)

lw_monitor_cell_mean.name = "TOA_LW_UP_mean"
lw_monitor_cell_mean.attrs["units"] = ds_flux["TOA_LW_UP"].attrs.get("units", "W m-2")
lw_monitor_cell_mean.attrs["description"] = (
    "Area-weighted mean TOA upwelling longwave flux using only eastern US grid cells "
    "that contain at least one AQS monitor"
)


# ============================================================
# Convert to annual means
# ============================================================
sw_monitor_cell_annual = (
    sw_monitor_cell_mean
    .groupby("time.year")
    .mean("time", skipna=True)
)

lw_monitor_cell_annual = (
    lw_monitor_cell_mean
    .groupby("time.year")
    .mean("time", skipna=True)
)

out_ds = xr.Dataset({
    "TOA_SW_UP_mean": sw_monitor_cell_annual,
    "TOA_LW_UP_mean": lw_monitor_cell_annual,
})


# ============================================================
# Add metadata
# ============================================================
out_ds["TOA_SW_UP_mean"].attrs["units"] = ds_flux["TOA_SW_UP"].attrs.get("units", "W m-2")
out_ds["TOA_SW_UP_mean"].attrs["description"] = (
    "Annual area-weighted mean TOA upwelling shortwave flux using only eastern US "
    "grid cells that contain at least one AQS monitor"
)

out_ds["TOA_LW_UP_mean"].attrs["units"] = ds_flux["TOA_LW_UP"].attrs.get("units", "W m-2")
out_ds["TOA_LW_UP_mean"].attrs["description"] = (
    "Annual area-weighted mean TOA upwelling longwave flux using only eastern US "
    "grid cells that contain at least one AQS monitor"
)

out_ds.attrs["spatial_sampling"] = (
    "Only flux grid cells containing at least one AQS monitor in the eastern US bounding box are averaged"
)
out_ds.attrs["temporal_resolution"] = "Annual"
out_ds.attrs["area_weighting"] = "cos(latitude)"
out_ds.attrs["n_unique_monitor_cells"] = int(len(unique_cells))
out_ds.attrs["n_unique_monitors"] = int(len(monitor_locations))
out_ds.attrs["lat_bounds"] = f"{minlat} to {maxlat}"
out_ds.attrs["lon_bounds"] = f"{minlon} to {maxlon}"
out_ds.attrs["year_start"] = int(YEAR_START)
out_ds.attrs["year_end"] = int(YEAR_END)

# Save NetCDF
out_ds.to_netcdf(OUTPUT_NC)

print("\nSaved monitor-cell-sampled annual flux means to:")
print(OUTPUT_NC)


# ============================================================
# Save CSV version
# ============================================================
out_df = out_ds.to_dataframe().reset_index()
out_df.to_csv(OUTPUT_CSV, index=False)

print("\nSaved CSV to:")
print(OUTPUT_CSV)


# ============================================================
# Print summary and trends
# ============================================================
print("\nAnnual output:")
print(out_df.head())
print(out_df.tail())

print("\nSummary:")
print(out_df[["TOA_SW_UP_mean", "TOA_LW_UP_mean"]].describe())

years = out_df["year"].values

sw_coeffs, sw_trendline = fit_trend(years, out_df["TOA_SW_UP_mean"].values)
lw_coeffs, lw_trendline = fit_trend(years, out_df["TOA_LW_UP_mean"].values)

print("\nAnnual area-weighted eastern US EPA-monitor-cell trends")
print("-------------------------------------------------------")
print(f"Years used: {int(np.nanmin(years))}-{int(np.nanmax(years))}")
print(f"Number of years: {len(years)}")
print()
print(f"TOA_SW_UP: {sw_coeffs[0]:.6f} W m^-2/year")
print(f"TOA_SW_UP: {sw_coeffs[0] * 10:.6f} W m^-2/decade")
print()
print(f"TOA_LW_UP: {lw_coeffs[0]:.6f} W m^-2/year")
print(f"TOA_LW_UP: {lw_coeffs[0] * 10:.6f} W m^-2/decade")