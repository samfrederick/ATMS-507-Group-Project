#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd

#settings
DATA_DIR = Path("/scratch/sjacker2/project_data")
FILE_GLOB = "MERRA2_*.tavgM_2d_rad_Nx.*.nc4.nc4"

OUTPUT_NC = DATA_DIR / "merra2_eastern_us_fluxes_ts_combined.nc"

#variable names
VAR_LW_SFC_NET = "LWGNTCLR"   # surface net longwave (clear-sky)
VAR_LW_TOA_UP  = "LWTUPCLR"   # TOA upwelling longwave (clear-sky)
VAR_SW_SFC_NET = "SWGNTCLR"   # surface net shortwave (clear-sky)
VAR_SW_TOA_NET = "SWTNTCLR"   # TOA net shortwave (clear-sky)
VAR_TS         = "TS"         # skin surface temperature (not using anymore)

#output variable names
OUT_NET_TOA = "NET_TOA_CLR"
OUT_NET_SFC = "NET_SFC_CLR"
OUT_TS      = "TS"


def get_files(data_dir: Path, pattern: str):
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {data_dir} matching {pattern}")
    return files


def preprocess_one(ds: xr.Dataset) -> xr.Dataset:
    """
    For one monthly file, compute:
      NET_TOA = SWTNTCLR - LWTUPCLR
      NET_SFC = SWGNTCLR + LWGNTCLR
    and keep TS.
    """
    needed = [
        VAR_LW_SFC_NET,
        VAR_LW_TOA_UP,
        VAR_SW_SFC_NET,
        VAR_SW_TOA_NET,
        VAR_TS,
    ]
    missing = [v for v in needed if v not in ds.variables]
    if missing:
        raise KeyError(f"Missing variables: {missing}")

    #compute net fluxes
    net_toa = ds[VAR_SW_TOA_NET] - ds[VAR_LW_TOA_UP]
    net_sfc = ds[VAR_SW_SFC_NET] + ds[VAR_LW_SFC_NET]
    ts = ds[VAR_TS]

    #name variables
    net_toa = net_toa.rename(OUT_NET_TOA)
    net_sfc = net_sfc.rename(OUT_NET_SFC)
    ts = ts.rename(OUT_TS)

    #add metadata
    net_toa.attrs["long_name"] = "Net downward TOA radiative flux, clear-sky"
    net_toa.attrs["units"] = ds[VAR_SW_TOA_NET].attrs.get("units", "")
    net_toa.attrs["formula"] = f"{VAR_SW_TOA_NET} - {VAR_LW_TOA_UP}"

    net_sfc.attrs["long_name"] = "Net downward surface radiative flux, clear-sky"
    net_sfc.attrs["units"] = ds[VAR_SW_SFC_NET].attrs.get("units", "")
    net_sfc.attrs["formula"] = f"{VAR_SW_SFC_NET} + {VAR_LW_SFC_NET}"

    ts.attrs["long_name"] = ts.attrs.get("long_name", "Skin surface temperature")

    out = xr.Dataset({
        OUT_NET_TOA: net_toa,
        OUT_NET_SFC: net_sfc,
        OUT_TS: ts,
    })

    #keep lat/lon/time in order
    for coord in ["time", "lat", "lon"]:
        if coord in out.coords:
            out = out.sortby(coord)

    return out


def area_weighted_mean(da: xr.DataArray) -> xr.DataArray:
    """
    Area-weighted mean over lat/lon using cosine(latitude).
    """
    weights = np.cos(np.deg2rad(da["lat"]))
    weights.name = "weights"
    return da.weighted(weights).mean(dim=("lat", "lon"))


def linear_trend_per_decade(da_time_series: xr.DataArray) -> float:
    """
    Compute linear trend per decade from a 1D monthly time series.
    Uses decimal years derived from the time coordinate.
    """
    if "time" not in da_time_series.dims:
        raise ValueError("Input DataArray must have a time dimension.")

    #drop NaNs just in case
    y = da_time_series.values
    t = pd.to_datetime(da_time_series["time"].values)

    mask = np.isfinite(y)
    y = y[mask]
    t = t[mask]

    if len(y) < 2:
        return np.nan

    #decimal year
    x = np.array([
        ti.year + (ti.dayofyear - 1) / (366 if ti.is_leap_year else 365)
        for ti in t
    ])

    slope_per_year, intercept = np.polyfit(x, y, 1)
    slope_per_decade = slope_per_year * 10.0
    return slope_per_decade


#main
def main():
    files = get_files(DATA_DIR, FILE_GLOB)
    print(f"Found {len(files)} files.")

    #open all files and preprocess into only output variables wanted
    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        preprocess=preprocess_one,
        parallel=False,
        chunks=None,
        decode_times=True,
        engine="netcdf4",
    )

    #sort by time just to be safe
    ds = ds.sortby("time")

    #add global metadata
    ds.attrs["title"] = "MERRA-2 Eastern US clear-sky net TOA flux, net surface flux, and skin temperature"
    ds.attrs["source_directory"] = str(DATA_DIR)
    ds.attrs["note"] = (
        f"{OUT_NET_TOA} = {VAR_SW_TOA_NET} - {VAR_LW_TOA_UP}; "
        f"{OUT_NET_SFC} = {VAR_SW_SFC_NET} + {VAR_LW_SFC_NET}. "
        "Data already clipped to eastern US."
    )

    #save combined NetCDF
    encoding = {
        OUT_NET_TOA: {"zlib": True, "complevel": 4},
        OUT_NET_SFC: {"zlib": True, "complevel": 4},
        OUT_TS: {"zlib": True, "complevel": 4},
    }

    ds.to_netcdf(OUTPUT_NC, encoding=encoding)
    print(f"Saved combined NetCDF to:\n{OUTPUT_NC}")

    #do area-weighting regional monthly means
    toa_mean = area_weighted_mean(ds[OUT_NET_TOA])
    sfc_mean = area_weighted_mean(ds[OUT_NET_SFC])
    ts_mean = area_weighted_mean(ds[OUT_TS])

    #compute trends per decade
    toa_trend_decade = linear_trend_per_decade(toa_mean)
    sfc_trend_decade = linear_trend_per_decade(sfc_mean)
    ts_trend_decade = linear_trend_per_decade(ts_mean)

    #print results
    print("\nArea-weighted eastern US trends over full time period:")
    print(f"{OUT_NET_TOA}: {toa_trend_decade:.6f} {ds[OUT_NET_TOA].attrs.get('units', '')}/decade")
    print(f"{OUT_NET_SFC}: {sfc_trend_decade:.6f} {ds[OUT_NET_SFC].attrs.get('units', '')}/decade")
    print(f"{OUT_TS}:      {ts_trend_decade:.6f} {ds[OUT_TS].attrs.get('units', '')}/decade")


if __name__ == "__main__":
    main()
