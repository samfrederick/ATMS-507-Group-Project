import numpy as np
import pandas as pd
import xarray as xr

# =========================
# USER SETTINGS
# =========================
AER_FILE = "/scratch/sjacker2/project_data/merra2_aer_eastern_us_subset_combined.nc"
SLV_FILE = "/scratch/sjacker2/project_data/merra2_slv_eastern_us_subset_combined.nc"
OUTPUT_NC = "/scratch/sjacker2/project_data/merra2_pm25_t2m_eastern_us_monthly_fields_DRY.nc"

YEAR_START = 1980
YEAR_END = 2025

AER_VARS = ["DUSMASS25", "OCSMASS", "BCSMASS", "SSSMASS25", "SO4SMASS"]
SLV_VARS = ["T2M"]


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
    # Keep full spatial fields for mapping
    # -----------------------------
    pm25_dry = (
        ds_aer["DUSMASS25"]
        + ds_aer["SSSMASS25"]
        + ds_aer["BCSMASS"]
        + ds_aer["OCSMASS"]
        + ((132.14 / 96.06) * ds_aer["SO4SMASS"])
    ) * 1e9

    pm25_dry.name = "PM25_dry"
    pm25_dry.attrs["long_name"] = "Dry total PM2.5 from MERRA-2 aerosol species"
    pm25_dry.attrs["units"] = "ug m-3"

    t2m = ds_slv["T2M"].copy()
    t2m.name = "T2M"
    t2m.attrs["long_name"] = "2-m air temperature"
    t2m.attrs["units"] = ds_slv["T2M"].attrs.get("units", "K")

    # Output full monthly gridded fields
    out_ds = xr.Dataset({
        "PM25_dry": pm25_dry,
        "T2M": t2m,
    })

    out_ds.attrs["title"] = "Monthly eastern US gridded dry MERRA-2 PM2.5 and 2-m temperature"
    out_ds.attrs["pm25_formula"] = (
        "PM2.5_dry = DUSMASS25 + SSSMASS25 + BCSMASS "
        "+ OCSMASS + (132.14/96.06)*SO4SMASS"
    )
    out_ds.attrs["note"] = "Full clipped eastern US monthly fields; no regional averaging applied."

    encoding = {
        "PM25_dry": {"zlib": True, "complevel": 4},
        "T2M": {"zlib": True, "complevel": 4},
    }

    out_ds.to_netcdf(OUTPUT_NC, encoding=encoding)

    print(f"Saved corrected dry monthly gridded NetCDF to:\n{OUTPUT_NC}")

    # Optional: still print eastern-US average trends for reference
    weights = np.cos(np.deg2rad(out_ds["lat"]))
    weights.name = "weights"

    pm25_mean = out_ds["PM25_dry"].weighted(weights).mean(dim=("lat", "lon"))
    t2m_mean = out_ds["T2M"].weighted(weights).mean(dim=("lat", "lon"))

    pm25_trend = linear_trend_per_decade(pm25_mean)
    t2m_trend = linear_trend_per_decade(t2m_mean)

    print("\nEastern US area-weighted monthly-mean trends over full time period:")
    print(f"PM25_dry: {pm25_trend:.6f} ug m-3/decade")
    print(f"T2M:      {t2m_trend:.6f} K/decade")

    print("\nMonthly PM2.5 summary from area-weighted mean:")
    print(pm25_mean.to_series().describe())

    ds_aer.close()
    ds_slv.close()
    out_ds.close()


if __name__ == "__main__":
    main()