import re
import time
from pathlib import Path
from urllib.parse import unquote

import requests
import xarray as xr


#settings
LINKS_FILE = "subset_M2TMNXAER_5.12.4_20260422_215822_.txt"
DOWNLOAD_DIR = Path("merra2_aer_downloads")
OUTPUT_NC = "merra2_aer_eastern_us_subset_combined.nc"

#set to None to process all links found
MAX_FILES = None

#retry if have download problems
MAX_RETRIES = 3
RETRY_DELAY_SEC = 10
TIMEOUT_SEC = 120

AER_VARS = ["DUSMASS25", "OCSMASS", "BCSMASS", "SSSMASS25", "SO4SMASS"]

def extract_urls(txt_path: str) -> list[str]:
    """Extract all HTTPS URLs from the text file I downloaded."""
    text = Path(txt_path).read_text(encoding="utf-8", errors="ignore")

    urls = re.findall(r"https://\S+", text)
    urls = [
        u.strip()
        for u in urls
        if "opendap.earthdata.nasa.gov" in u and "M2TMNXAER.5.12.4" in u
    ]

    seen = set()
    unique_urls = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique_urls.append(u)

    return unique_urls


def filename_from_url(url: str) -> str:
    """
    Extract a local filename from a DAP4 Earthdata URL.
    Example:
    ...M2TMNXAER.5.12.4%3AMERRA2_100.tavgM_2d_aer_Nx.198001.nc4.dap.nc4?dap4.ce=...
    -> MERRA2_100.tavgM_2d_aer_Nx.198001.nc4
    """
    decoded = unquote(url)

    m = re.search(r"M2TMNXAER\.5\.12\.4:(MERRA2_\d+\.tavgM_2d_aer_Nx\.\d{6}\.nc4)", decoded)
    if m:
        return m.group(1)

    # fallback
    return f"download_{abs(hash(url))}.nc4"


def download_one_url(session: requests.Session, url: str, out_dir: Path) -> Path:
    """
    Download one DAP4 subset result to a local NetCDF file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename_from_url(url)

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  Already downloaded: {out_path.name}")
        return out_path

    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with session.get(url, stream=True, timeout=TIMEOUT_SEC) as r:
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

            if out_path.stat().st_size == 0:
                raise RuntimeError("Downloaded file is empty.")

            return out_path

        except Exception as e:
            last_err = e
            print(f"[Attempt {attempt}/{MAX_RETRIES}] Download failed:")
            print(f"  URL: {url}")
            print(f"  Error: {e}")

            if out_path.exists():
                out_path.unlink(missing_ok=True)

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)

    raise RuntimeError(f"Could not download URL after {MAX_RETRIES} attempts:\n{url}\n{last_err}")


def preprocess_local_file(fp: Path) -> xr.Dataset:
    """
    Open a downloaded local NetCDF file and keep only the desired aerosol vars.
    """
    ds = xr.open_dataset(fp)

    keep_vars = [v for v in AER_VARS if v in ds.data_vars]
    ds = ds[keep_vars]

    if "time" in ds.coords:
        ds = ds.sortby("time")
    if "lat" in ds.coords:
        ds = ds.sortby("lat")
    if "lon" in ds.coords:
        ds = ds.sortby("lon")

    keep_coords = {"time", "lat", "lon"}
    drop_coords = [c for c in ds.coords if c not in keep_coords and ds[c].ndim == 0]
    if drop_coords:
        ds = ds.drop_vars(drop_coords, errors="ignore")

    ds = ds.load()
    return ds


def main():
    print(f"Reading links from: {LINKS_FILE}")
    urls = extract_urls(LINKS_FILE)

    if not urls:
        raise ValueError("No valid M2TMNXAER OPeNDAP URLs were found in the links file.")

    if MAX_FILES is not None:
        urls = urls[:MAX_FILES]

    print(f"Found {len(urls)} OPeNDAP URLs.")

    session = requests.Session()

    #Earthdata works with .netrc automatically through requests,
    #but this makes sure that session should trust env/netrc config since it was giving me problems earlier
    session.trust_env = True

    downloaded_files = []
    failed = []

    for i, url in enumerate(urls, start=1):
        print(f"\n[{i}/{len(urls)}] Downloading:")
        print(url)

        try:
            fp = download_one_url(session, url, DOWNLOAD_DIR)
            downloaded_files.append(fp)
            print(f"  Saved: {fp}")

        except Exception as e:
            print(f"  FAILED permanently: {e}")
            failed.append(url)

    if not downloaded_files:
        raise RuntimeError("No files were successfully downloaded.")

    print("\nOpening downloaded files and concatenating along time...")
    datasets = [preprocess_local_file(fp) for fp in sorted(downloaded_files)]

    combined = xr.concat(
        datasets,
        dim="time",
        data_vars="all",
        coords="minimal",
        compat="override",
        combine_attrs="override",
    )

    if "time" in combined.coords:
        combined = combined.sortby("time")
        time_index = combined.indexes["time"]
        if hasattr(time_index, "duplicated"):
            keep = ~time_index.duplicated()
            combined = combined.isel(time=keep)

    combined.attrs["title"] = "Combined MERRA-2 M2TMNXAER subset from provided Earthdata DAP4 URLs"
    combined.attrs["source_links_file"] = LINKS_FILE
    combined.attrs["variables"] = ", ".join(AER_VARS)
    combined.attrs["note"] = (
        "Downloaded from provided Earthdata DAP4 subset URLs to local files first, "
        "then concatenated. No spatial or temporal averaging performed."
    )

    encoding = {
        var: {"zlib": True, "complevel": 4}
        for var in combined.data_vars
    }

    print(f"\nWriting output to: {OUTPUT_NC}")
    combined.to_netcdf(OUTPUT_NC, encoding=encoding)

    print("\nDone.")
    print(f"Saved combined NetCDF: {OUTPUT_NC}")
    print(f"Downloaded files: {len(downloaded_files)}")

    if failed:
        failed_txt = Path(OUTPUT_NC).with_suffix(".failed_urls.txt")
        failed_txt.write_text("\n".join(failed), encoding="utf-8")
        print(f"Failed URLs: {len(failed)}")
        print(f"List written to: {failed_txt}")


if __name__ == "__main__":
    main()
