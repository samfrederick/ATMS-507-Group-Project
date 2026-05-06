#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd

# =========================
# USER SETTINGS
# =========================
DATA_DIR = Path("/scratch/sjacker2/project_data/merra2_rad_downloads")
FILE_GLOB = "MERRA2_*.tavgM_2d_rad_Nx.*.nc4"

OUTPUT_NC = DATA_DIR / "merra2_eastern_us_toa_upwelling_lw_sw_combined.nc"

# Variable names expected in your files
VAR_LW_TOA_UP = "LWTUP"   # TOA upwelling longwave
VAR_SW_TOA_DN = "SWTDN"   # TOA incoming shortwave
VAR_SW_TOA_NET = "SWTNT"  # TOA net downward shortwave

# Output variable names
OUT_TOA_LW_UP = "TOA_LW_UP"
OUT_TOA_SW_UP = "TOA_SW_UP"

# =========================
# FUNCTIONS
# =========================
def get_files(data_dir: Path, pattern: str):
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {data_dir} matching {pattern}")
    return files


def preprocess_one(ds: xr.Dataset) -> xr.Dataset:
    """
    For one monthly file, keep:
      TOA_LW_UP = LWTUP
      TOA_SW_UP = SWTDN - SWTNT
    """
    needed = [
        VAR_LW_TOA_UP,
        VAR_SW_TOA_DN,
        VAR_SW_TOA_NET,
    ]
    missing = [v for v in needed if v not in ds.variables]
    if missing:
        raise KeyError(f"Missing variables: {missing}")

    toa_lw_up = ds[VAR_LW_TOA_UP]
    toa_sw_up = ds[VAR_SW_TOA_DN] - ds[VAR_SW_TOA_NET]

    toa_lw_up = toa_lw_up.rename(OUT_TOA_LW_UP)
    toa_sw_up = toa_sw_up.rename(OUT_TOA_SW_UP)

    toa_lw_up.attrs["long_name"] = "TOA upwelling longwave flux"
    toa_lw_up.attrs["units"] = ds[VAR_LW_TOA_UP].attrs.get("units", "")
    toa_lw_up.attrs["source_variable"] = VAR_LW_TOA_UP

    toa_sw_up.attrs["long_name"] = "TOA upwelling shortwave flux"
    toa_sw_up.attrs["units"] = ds[VAR_SW_TOA_DN].attrs.get("units", "")
    toa_sw_up.attrs["formula"] = f"{VAR_SW_TOA_DN} - {VAR_SW_TOA_NET}"

    out = xr.Dataset({
        OUT_TOA_LW_UP: toa_lw_up,
        OUT_TOA_SW_UP: toa_sw_up,
    })

    for coord in ["time", "lat", "lon"]:
        if coord in out.coords:
            out = out.sortby(coord)

    return out


def area_weighted_mean(da: xr.DataArray) -> xr.DataArray:
    weights = np.cos(np.deg2rad(da["lat"]))
    weights.name = "weights"
    return da.weighted(weights).mean(dim=("lat", "lon"))


def linear_trend_per_decade(da_time_series: xr.DataArray) -> float:
    if "time" not in da_time_series.dims:
        raise ValueError("Input DataArray must have a time dimension.")

    y = da_time_series.values
    t = pd.to_datetime(da_time_series["time"].values)

    mask = np.isfinite(y)
    y = y[mask]
    t = t[mask]

    if len(y) < 2:
        return np.nan

    x = np.array([
        ti.year + (ti.dayofyear - 1) / (366 if ti.is_leap_year else 365)
        for ti in t
    ])

    slope_per_year, _ = np.polyfit(x, y, 1)
    return slope_per_year * 10.0


# =========================
# MAIN
# =========================
def main():
    files = get_files(DATA_DIR, FILE_GLOB)
    print(f"Found {len(files)} files.")

    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        preprocess=preprocess_one,
        parallel=False,
        chunks=None,
        decode_times=True,
        engine="netcdf4",
    )

    ds = ds.sortby("time")

    ds.attrs["title"] = "MERRA-2 Eastern US TOA upwelling longwave and shortwave fluxes"
    ds.attrs["source_directory"] = str(DATA_DIR)
    ds.attrs["note"] = (
        f"{OUT_TOA_LW_UP} = {VAR_LW_TOA_UP}; "
        f"{OUT_TOA_SW_UP} = {VAR_SW_TOA_DN} - {VAR_SW_TOA_NET}. "
        "Data already clipped to eastern US."
    )

    encoding = {
        OUT_TOA_LW_UP: {"zlib": True, "complevel": 4},
        OUT_TOA_SW_UP: {"zlib": True, "complevel": 4},
    }

    ds.to_netcdf(OUTPUT_NC, encoding=encoding)
    print(f"Saved combined NetCDF to:\n{OUTPUT_NC}")

    # Area-weighted regional monthly means
    toa_lw_up_mean = area_weighted_mean(ds[OUT_TOA_LW_UP])
    toa_sw_up_mean = area_weighted_mean(ds[OUT_TOA_SW_UP])

    # Compute trends per decade
    toa_lw_up_trend_decade = linear_trend_per_decade(toa_lw_up_mean)
    toa_sw_up_trend_decade = linear_trend_per_decade(toa_sw_up_mean)

    print("\nArea-weighted eastern US trends over full time period:")
    print(f"{OUT_TOA_LW_UP}: {toa_lw_up_trend_decade:.6f} {ds[OUT_TOA_LW_UP].attrs.get('units', '')}/decade")
    print(f"{OUT_TOA_SW_UP}: {toa_sw_up_trend_decade:.6f} {ds[OUT_TOA_SW_UP].attrs.get('units', '')}/decade")


if __name__ == "__main__":
    main()