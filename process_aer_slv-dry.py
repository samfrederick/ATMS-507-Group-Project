import numpy as np
import pandas as pd
import xarray as xr

# =========================
# USER SETTINGS
# =========================
AER_FILE = "/scratch/sjacker2/project_data/merra2_aer_eastern_us_subset_combined.nc"
SLV_FILE = "/scratch/sjacker2/project_data/merra2_slv_eastern_us_subset_combined.nc"
OUTPUT_NC = "/scratch/sjacker2/project_data/merra2_pm25_t2m_eastern_us_monthly_means_DRY.nc"

YEAR_START = 1980
YEAR_END = 2025

AER_VARS = ["DUSMASS25", "OCSMASS", "BCSMASS", "SSSMASS25", "SO4SMASS"]
SLV_VARS = ["T2M"]


def area_weighted_mean(da):
    weights = np.cos(np.deg2rad(da["lat"]))
    weights.name = "weights"
    return da.weighted(weights).mean(dim=("lat", "lon"))


def linear_trend_per_decade(da_time_series):
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


def main():
    ds_aer = xr.open_dataset(AER_FILE)[AER_VARS]
    ds_slv = xr.open_dataset(SLV_FILE)[SLV_VARS]

    # Keep overlapping time range
    start_time = max(
        pd.to_datetime(ds_aer.time.values[0]),
        pd.to_datetime(ds_slv.time.values[0])
    )
    end_time = min(
        pd.to_datetime(ds_aer.time.values[-1]),
        pd.to_datetime(ds_slv.time.values[-1])
    )

    ds_aer = ds_aer.sel(time=slice(start_time, end_time))
    ds_slv = ds_slv.sel(time=slice(start_time, end_time))

    ds_aer = ds_aer.sel(time=slice(f"{YEAR_START}-01-01", f"{YEAR_END}-12-31"))
    ds_slv = ds_slv.sel(time=slice(f"{YEAR_START}-01-01", f"{YEAR_END}-12-31"))

    # -----------------------------
    # Correct dry MERRA-2 PM2.5 estimate
    # -----------------------------
    pm25_dry = (
        ds_aer["DUSMASS25"]
        + ds_aer["SSSMASS25"]
        + ds_aer["BCSMASS"]
        + ds_aer["OCSMASS"]
        + ((132.14/96.06) * ds_aer["SO4SMASS"])
    ) * 1e9

    pm25_dry.name = "PM25_mean"
    pm25_dry.attrs["long_name"] = "Dry total PM2.5 from MERRA-2 aerosol species"
    pm25_dry.attrs["units"] = "ug m-3"

    pm25_mean = area_weighted_mean(pm25_dry)
    t2m_mean = area_weighted_mean(ds_slv["T2M"])

    pm25_mean.name = "PM25_mean"
    t2m_mean.name = "T2M_mean"

    t2m_mean.attrs["long_name"] = "2-m air temperature"
    t2m_mean.attrs["units"] = ds_slv["T2M"].attrs.get("units", "K")

    out_ds = xr.Dataset({
        "PM25_mean": pm25_mean,
        "T2M_mean": t2m_mean,
    })

    out_ds.attrs["title"] = "Monthly eastern US means of dry MERRA-2 PM2.5 and 2-m temperature"
    out_ds.attrs["pm25_formula"] = (
        "PM2.5_dry = DUSMASS25 + SSSMASS25 + BCSMASS "
        "+ OCSMASS + 1.375*SO4SMASS"
    )

    encoding = {
        "PM25_mean": {"zlib": True, "complevel": 4},
        "T2M_mean": {"zlib": True, "complevel": 4},
    }

    out_ds.to_netcdf(OUTPUT_NC, encoding=encoding)

    print(f"Saved corrected dry monthly means NetCDF to:\n{OUTPUT_NC}")

    pm25_trend = linear_trend_per_decade(out_ds["PM25_mean"])
    t2m_trend = linear_trend_per_decade(out_ds["T2M_mean"])

    print("\nEastern US monthly-mean trends over full time period:")
    print(f"PM25_mean dry: {pm25_trend:.6f} ug m-3/decade")
    print(f"T2M_mean:      {t2m_trend:.6f} K/decade")

    print("\nMonthly PM2.5 summary:")
    print(out_ds["PM25_mean"].to_series().describe())

    ds_aer.close()
    ds_slv.close()
    out_ds.close()


if __name__ == "__main__":
    main()