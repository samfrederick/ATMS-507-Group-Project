import numpy as np
import pandas as pd
import xarray as xr


#settings
AER_FILE = "/scratch/sjacker2/project_data/merra2_aer_eastern_us_subset_combined.nc"
SLV_FILE = "/scratch/sjacker2/project_data/merra2_slv_eastern_us_subset_combined.nc"
OUTPUT_NC = "/scratch/sjacker2/project_data/merra2_pm25_t2m_eastern_us_monthly_means.nc"

YEAR_START = 1980
YEAR_END = 2025

AER_VARS = ["DUSMASS25", "OCSMASS", "BCSMASS", "SSSMASS25", "SO4SMASS"]
SLV_VARS = ["QV2M", "T2M", "PS"]


#humidity growth factors
RH_NODES = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=float)

FRH_OC = np.array([1.0, 1.0, 1.0, 1.03, 1.06, 1.09, 1.14, 1.21, 1.33, 1.69])
FRH_BC = np.array([1.0, 1.0, 1.0, 1.01, 1.02, 1.03, 1.05, 1.08, 1.14, 1.30])
FRH_SS = np.array([1.0, 1.36, 1.52, 1.65, 1.79, 1.95, 2.17, 2.50, 3.09, 4.71])
FRH_SO4 = np.array([1.0, 1.0, 1.01, 1.04, 1.09, 1.17, 1.30, 1.54, 2.03, 3.61])


def get_growth_factor(rh_fraction: np.ndarray, frh_table: np.ndarray) -> np.ndarray:
    rh_pct = np.clip(rh_fraction * 100.0, 0.0, 90.0)
    return np.interp(rh_pct, RH_NODES, frh_table)


#RH from QV2M, T2M, PS
def specific_humidity_to_rh(qv: np.ndarray, t: np.ndarray, ps: np.ndarray) -> np.ndarray:
    t_c = t - 273.15
    es = 611.2 * np.exp(17.67 * t_c / (t_c + 243.5))  # Pa
    e = qv * ps / (0.622 + 0.378 * qv)
    rh = np.clip(e / es, 0.0, 1.0)
    return rh


#area-weighted mean
def area_weighted_mean(da: xr.DataArray) -> xr.DataArray:
    weights = np.cos(np.deg2rad(da["lat"]))
    weights.name = "weights"
    return da.weighted(weights).mean(dim=("lat", "lon"))


#trend per decade
def linear_trend_per_decade(da_time_series: xr.DataArray) -> float:
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
    #open combined files
    ds_aer = xr.open_dataset(AER_FILE)[AER_VARS]
    ds_slv = xr.open_dataset(SLV_FILE)[SLV_VARS]

    #keep only overlapping time range and wanted years
    start_time = max(pd.to_datetime(ds_aer.time.values[0]), pd.to_datetime(ds_slv.time.values[0]))
    end_time = min(pd.to_datetime(ds_aer.time.values[-1]), pd.to_datetime(ds_slv.time.values[-1]))

    ds_aer = ds_aer.sel(time=slice(start_time, end_time))
    ds_slv = ds_slv.sel(time=slice(start_time, end_time))

    ds_aer = ds_aer.sel(time=slice(f"{YEAR_START}-01-01", f"{YEAR_END}-12-31"))
    ds_slv = ds_slv.sel(time=slice(f"{YEAR_START}-01-01", f"{YEAR_END}-12-31"))

    #calc RH field
    rh = specific_humidity_to_rh(
        ds_slv["QV2M"].values,
        ds_slv["T2M"].values,
        ds_slv["PS"].values,
    )

    rh_da = xr.DataArray(
        rh,
        dims=ds_slv["T2M"].dims,
        coords=ds_slv["T2M"].coords,
        name="RH"
    )

    #growth factor fields
    frh_oc = xr.DataArray(
        get_growth_factor(rh_da.values, FRH_OC),
        dims=rh_da.dims,
        coords=rh_da.coords,
    )
    frh_bc = xr.DataArray(
        get_growth_factor(rh_da.values, FRH_BC),
        dims=rh_da.dims,
        coords=rh_da.coords,
    )
    frh_ss = xr.DataArray(
        get_growth_factor(rh_da.values, FRH_SS),
        dims=rh_da.dims,
        coords=rh_da.coords,
    )
    frh_so4 = xr.DataArray(
        get_growth_factor(rh_da.values, FRH_SO4),
        dims=rh_da.dims,
        coords=rh_da.coords,
    )

    #PM2.5 in ug/m^3
    pm25 = (
        ds_aer["DUSMASS25"]
        + frh_oc * ds_aer["OCSMASS"]
        + frh_bc * ds_aer["BCSMASS"]
        + frh_ss * ds_aer["SSSMASS25"]
        + frh_so4 * ds_aer["SO4SMASS"] * 1.3756
    ) * 1e9

    pm25.name = "PM25_mean"
    pm25.attrs["long_name"] = "Total PM2.5"
    pm25.attrs["units"] = "ug m-3"

    #eastern US monthly means
    pm25_mean = area_weighted_mean(pm25)
    t2m_mean = area_weighted_mean(ds_slv["T2M"])

    pm25_mean.name = "PM25_mean"
    t2m_mean.name = "T2M_mean"
    t2m_mean.attrs["long_name"] = "2-m air temperature"
    t2m_mean.attrs["units"] = ds_slv["T2M"].attrs.get("units", "K")

    #save monthly means
    out_ds = xr.Dataset({
        "PM25_mean": pm25_mean,
        "T2M_mean": t2m_mean,
    })

    out_ds.attrs["title"] = "Monthly eastern US means of total PM2.5 and 2-m temperature"
    out_ds.attrs["pm25_formula"] = (
        "PM2.5 = DUSMASS25 + frh_oc*OCSMASS + frh_bc*BCSMASS + "
        "frh_ss*SSSMASS25 + frh_so4*SO4SMASS*1.3756"
    )

    encoding = {
        "PM25_mean": {"zlib": True, "complevel": 4},
        "T2M_mean": {"zlib": True, "complevel": 4},
    }

    out_ds.to_netcdf(OUTPUT_NC, encoding=encoding)
    print(f"Saved monthly means NetCDF to:\n{OUTPUT_NC}")

    #trends per decade
    pm25_trend = linear_trend_per_decade(out_ds["PM25_mean"])
    t2m_trend = linear_trend_per_decade(out_ds["T2M_mean"])

    print("\nEastern US monthly-mean trends over full time period:")
    print(f"PM25_mean: {pm25_trend:.6f} {out_ds['PM25_mean'].attrs.get('units', 'ug m-3')}/decade")
    print(f"T2M_mean:  {t2m_trend:.6f} {out_ds['T2M_mean'].attrs.get('units', 'K')}/decade")

    ds_aer.close()
    ds_slv.close()
    out_ds.close()


if __name__ == "__main__":
    main()