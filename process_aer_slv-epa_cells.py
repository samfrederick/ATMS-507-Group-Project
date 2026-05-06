import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# ============================================================
# File paths
# ============================================================
AER_FILE = "/scratch/sjacker2/project_data/merra2_aer_eastern_us_subset_combined.nc"
SLV_FILE = "/scratch/sjacker2/project_data/merra2_slv_eastern_us_subset_combined.nc"
MONITOR_FILE = "/scratch/sjacker2/project_data/monthly_pm25_eastern_us_1999_2025.csv"

OUTPUT_NC = "/scratch/sjacker2/project_data/merra2_pm25_t2m_monitor_cells_eastern_us_monthly.nc"
OUTPUT_CSV = "/scratch/sjacker2/project_data/merra2_monitor_cell_pm25_t2m_eastern_us_monthly.csv"

# ============================================================
# Eastern US bounding box
# ============================================================
minlat = 25
maxlat = 50
minlon = -90
maxlon = -65

YEAR_START = 1997
YEAR_END = 2025

AER_VARS = ["DUSMASS25", "OCSMASS", "BCSMASS", "SSSMASS25", "SO4SMASS"]
SLV_VARS = ["T2M"]

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
# Helper: assign monitors to MERRA-2 grid cells
# ============================================================
def assign_monitors_to_grid(monitors, lats, lons):
    """
    Assign each monitor to a MERRA-2 grid cell using lat/lon grid-cell edges.
    Returns dataframe with merra_lat_idx and merra_lon_idx.
    """
    lat_edges = get_edges_from_centers(lats)
    lon_edges = get_edges_from_centers(lons)

    lat_idx = np.digitize(monitors["Latitude"].values, lat_edges) - 1
    lon_idx = np.digitize(monitors["Longitude"].values, lon_edges) - 1

    monitors = monitors.copy()
    monitors["merra_lat_idx"] = lat_idx
    monitors["merra_lon_idx"] = lon_idx

    monitors = monitors[
        (monitors["merra_lat_idx"] >= 0) &
        (monitors["merra_lat_idx"] < len(lats)) &
        (monitors["merra_lon_idx"] >= 0) &
        (monitors["merra_lon_idx"] < len(lons))
    ].copy()

    monitors["merra_lat"] = lats[monitors["merra_lat_idx"].values]
    monitors["merra_lon"] = lons[monitors["merra_lon_idx"].values]

    return monitors


# ============================================================
# Read monitor data and get unique monitor locations
# ============================================================
mon = pd.read_csv(MONITOR_FILE)
mon.columns = mon.columns.str.strip()

for col in ["Latitude", "Longitude", "year", "month", "monthly_pm25_mean"]:
    if col in mon.columns:
        mon[col] = pd.to_numeric(mon[col], errors="coerce")

mon = mon.dropna(subset=["Latitude", "Longitude"]).copy()

mon = mon[
    (mon["Latitude"] >= minlat) &
    (mon["Latitude"] <= maxlat) &
    (mon["Longitude"] >= minlon) &
    (mon["Longitude"] <= maxlon)
].copy()

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
# Open MERRA-2 aerosol and temperature fields
# ============================================================
ds_aer = xr.open_dataset(AER_FILE)[AER_VARS]
ds_slv = xr.open_dataset(SLV_FILE)[SLV_VARS]

ds_aer = ds_aer.sel(time=slice(f"{YEAR_START}-01-01", f"{YEAR_END}-12-31"))
ds_slv = ds_slv.sel(time=slice(f"{YEAR_START}-01-01", f"{YEAR_END}-12-31"))

print(ds_aer)
print(ds_slv)

# Assume same lat/lon grid between aerosol and SLV files
lats = ds_aer["lat"].values
lons = ds_aer["lon"].values

monitor_locations_for_grid = monitor_locations.copy()

# If MERRA-2 longitudes are 0 to 360, convert monitor longitudes to 0 to 360
if np.nanmax(lons) > 180:
    monitor_locations_for_grid["Longitude"] = monitor_locations_for_grid["Longitude"] % 360


# ============================================================
# Assign monitors to MERRA-2 cells
# ============================================================
monitor_grid = assign_monitors_to_grid(
    monitor_locations_for_grid,
    lats,
    lons
)

monitor_grid["monitor_lon_original"] = monitor_locations.loc[
    monitor_grid.index, "Longitude"
].values

print(f"Monitors successfully assigned to MERRA-2 cells: {len(monitor_grid):,}")

unique_cells = (
    monitor_grid[["merra_lat_idx", "merra_lon_idx", "merra_lat", "merra_lon"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

print(f"Unique MERRA-2 cells containing monitors: {len(unique_cells):,}")

cell_counts = (
    monitor_grid
    .groupby(["merra_lat_idx", "merra_lon_idx", "merra_lat", "merra_lon"], as_index=False)
    .agg(n_monitors=("monitor_id", "nunique"))
)

print(cell_counts.head())


# ============================================================
# Compute dry MERRA-2 PM2.5
# ============================================================
pm25_dry = (
    ds_aer["DUSMASS25"]
    + ds_aer["SSSMASS25"]
    + ds_aer["BCSMASS"]
    + ds_aer["OCSMASS"]
    + ((132.14 / 96.06) * ds_aer["SO4SMASS"])
) * 1e9

pm25_dry.name = "PM25_dry"
pm25_dry.attrs["units"] = "ug m-3"
pm25_dry.attrs["long_name"] = "Dry MERRA-2 PM2.5 at grid cells containing AQS monitors"

t2m = ds_slv["T2M"].copy()
t2m.name = "T2M"
t2m.attrs["units"] = ds_slv["T2M"].attrs.get("units", "K")
t2m.attrs["long_name"] = "2-m air temperature at grid cells containing AQS monitors"


# ============================================================
# Create a mask for MERRA-2 cells that contain monitors
# ============================================================
cell_mask = xr.zeros_like(pm25_dry.isel(time=0), dtype=bool)

for _, row in unique_cells.iterrows():
    cell_mask.values[
        int(row["merra_lat_idx"]),
        int(row["merra_lon_idx"])
    ] = True

pm25_monitor_cells = pm25_dry.where(cell_mask)
t2m_monitor_cells = t2m.where(cell_mask)


# ============================================================
# Average only over MERRA-2 cells with monitors
# ============================================================
weights = np.cos(np.deg2rad(pm25_monitor_cells["lat"]))
weights.name = "weights"

merra_monitor_cell_pm25_mean = (
    pm25_monitor_cells
    .weighted(weights)
    .mean(dim=("lat", "lon"))
)

merra_monitor_cell_t2m_mean = (
    t2m_monitor_cells
    .weighted(weights)
    .mean(dim=("lat", "lon"))
)

merra_monitor_cell_pm25_mean.name = "PM25_mean"
merra_monitor_cell_pm25_mean.attrs["units"] = "ug m-3"
merra_monitor_cell_pm25_mean.attrs["description"] = (
    "Area-weighted eastern US mean using only MERRA-2 grid cells that contain at least one AQS monitor"
)

merra_monitor_cell_t2m_mean.name = "T2M_mean"
merra_monitor_cell_t2m_mean.attrs["units"] = ds_slv["T2M"].attrs.get("units", "K")
merra_monitor_cell_t2m_mean.attrs["description"] = (
    "Area-weighted eastern US mean using only MERRA-2 grid cells that contain at least one AQS monitor"
)

# Resample to monthly start if needed
merra_monitor_cell_pm25_monthly = merra_monitor_cell_pm25_mean.resample(time="MS").mean()
merra_monitor_cell_t2m_monthly = merra_monitor_cell_t2m_mean.resample(time="MS").mean()

out_ds = xr.Dataset({
    "PM25_mean": merra_monitor_cell_pm25_monthly,
    "T2M_mean": merra_monitor_cell_t2m_monthly,
})

out_ds.attrs["pm25_formula"] = (
    "PM2.5_dry = DUSMASS25 + SSSMASS25 + BCSMASS + OCSMASS + (132.14/96.06)*SO4SMASS"
)
out_ds.attrs["spatial_sampling"] = (
    "Only MERRA-2 grid cells containing at least one AQS monitor in the eastern US bounding box are averaged"
)
out_ds.attrs["n_unique_monitor_cells"] = int(len(unique_cells))
out_ds.attrs["n_unique_monitors"] = int(len(monitor_locations))

out_ds.to_netcdf(OUTPUT_NC)

print(f"\nSaved monitor-cell-sampled MERRA-2 monthly means to:")
print(OUTPUT_NC)


# ============================================================
# Save CSV version
# ============================================================
out_df = out_ds.to_dataframe().reset_index()

out_df["year"] = pd.to_datetime(out_df["time"]).dt.year
out_df["month"] = pd.to_datetime(out_df["time"]).dt.month

out_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved CSV to:")
print(OUTPUT_CSV)

print("\nSummary:")
print(out_df[["PM25_mean", "T2M_mean"]].describe())


# ============================================================
# Quick plot
# ============================================================
fig, ax1 = plt.subplots(figsize=(13, 5))

l1 = ax1.plot(
    out_df["time"],
    out_df["PM25_mean"],
    linewidth=2,
    label="MERRA-2 PM2.5, monitor-containing cells"
)
ax1.set_xlabel("Date", fontsize=14)
ax1.set_ylabel("PM2.5 (µg m$^{-3}$)", fontsize=14)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
l2 = ax2.plot(
    out_df["time"],
    out_df["T2M_mean"],
    linewidth=2,
    label="MERRA-2 T2M, monitor-containing cells"
)
ax2.set_ylabel("2-m Temperature (K)", fontsize=14)

lines = l1 + l2
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, fontsize=12, loc="upper left")

plt.title("Eastern US MERRA-2 PM2.5 and T2M Averaged Only Over Grid Cells with AQS Monitors", fontsize=15)
plt.tight_layout()
plt.show()

ds_aer.close()
ds_slv.close()
out_ds.close()