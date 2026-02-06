# {{docs-fragment imports}}
import asyncio
import gc
import io
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import flyte
import numpy as np
import pandas as pd
import xarray as xr
from flyte.io import File
from flyteplugins.dask import Dask, Scheduler, WorkerGroup
# {{/docs-fragment imports}}


# {{docs-fragment dataclasses}}
@dataclass
class SimulationParams:
    grid_resolution_km: float = 10.0
    time_step_minutes: int = 10
    simulation_hours: int = 240
    physics_model: Literal["WRF", "MPAS", "CAM"] = "WRF"
    boundary_layer_scheme: str = "YSU"
    microphysics_scheme: str = "Thompson"
    radiation_scheme: str = "RRTMG"

    # Ensemble forecasting parameters
    ensemble_size: int = 800
    perturbation_magnitude: float = 0.5

    # Convergence criteria for adaptive refinement
    convergence_threshold: float = 0.1  # 10% of initial ensemble spread
    max_iterations: int = 3


@dataclass
class ClimateMetrics:
    timestamp: str
    iteration: int
    convergence_rate: float
    energy_conservation_error: float
    max_wind_speed_mps: float
    min_pressure_mb: float
    detected_phenomena: list[str]
    compute_time_seconds: float
    ensemble_spread: float


@dataclass
class SimulationSummary:
    total_iterations: int
    final_resolution_km: float
    avg_convergence_rate: float
    total_compute_time_seconds: float
    hurricanes_detected: int
    heatwaves_detected: int
    converged: bool
    region: str
    output_files: list[File]
    date_range: list[str, str]
# {{/docs-fragment dataclasses}}


# {{docs-fragment image}}
climate_image = (
    flyte.Image.from_debian_base(name="climate_modeling_h200")
    .with_apt_packages(
        "libnetcdf-dev",  # NetCDF for climate data
        "libhdf5-dev",  # HDF5 for large datasets
        "libeccodes-dev",  # GRIB format support (ECMWF's native format)
        "libudunits2-dev",  # Unit conversions
    )
    .with_pip_packages(
        "numpy==2.3.5",
        "pandas==2.3.3",
        "xarray==2025.11.0",
        "torch==2.9.1",
        "netCDF4==1.7.3",
        "s3fs==2025.10.0",
        "aiohttp==3.13.2",
        "ecmwf-datastores-client==0.4.1",
        "h5netcdf==1.7.3",
        "cfgrib==0.9.15.1",
        "pyarrow==22.0.0",
        "scipy==1.15.1",
        "flyteplugins-dask>=2.0.0b33",
        "nvidia-ml-py3==7.352.0",
    )
    .with_env_vars({"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"})
)
# {{/docs-fragment image}}

# {{docs-fragment task-envs}}
gpu_env = flyte.TaskEnvironment(
    name="climate_modeling_gpu",
    resources=flyte.Resources(
        cpu=5,
        memory="130Gi",
        gpu="H200:1",
    ),
    image=climate_image,
    cache="auto",
)

dask_env = flyte.TaskEnvironment(
    name="climate_modeling_dask",
    plugin_config=Dask(
        scheduler=Scheduler(resources=flyte.Resources(cpu=2, memory="6Gi")),
        workers=WorkerGroup(
            number_of_workers=2,
            resources=flyte.Resources(cpu=2, memory="12Gi"),
        ),
    ),
    image=climate_image,
    resources=flyte.Resources(cpu=2, memory="12Gi"),  # Head node
    cache="auto",
)


cpu_env = flyte.TaskEnvironment(
    name="climate_modeling_cpu",
    resources=flyte.Resources(cpu=8, memory="64Gi"),
    image=climate_image,
    cache="auto",
    secrets=[
        flyte.Secret(key="cds_api_key", as_env_var="ECMWF_DATASTORES_KEY"),
        flyte.Secret(key="cds_api_url", as_env_var="ECMWF_DATASTORES_URL"),
    ],
    depends_on=[gpu_env, dask_env],
)
# {{/docs-fragment task-envs}}


# {{docs-fragment ingest-satellite}}
@cpu_env.task
async def ingest_satellite_data(region: str, date_range: list[str, str]) -> File:
    """Ingest GOES satellite imagery from NOAA's public S3 buckets."""
    # {{/docs-fragment ingest-satellite}}
    import s3fs
    import xarray as xr

    start_date, end_date = date_range[0], date_range[1]

    # Initialize S3 filesystem (anonymous access for NOAA data)
    s3 = s3fs.S3FileSystem(anon=True)

    # Define broader region bounds for geographic filtering [S, W, N, E]
    region_bounds = {
        "atlantic": [
            -10,
            -100,
            60,
            20,
        ],  # Atlantic Ocean + Caribbean + North America coast
        "pacific": [-20, -180, 50, -100],  # Eastern + Western Pacific
        "indian": [-40, 40, 30, 120],  # Indian Ocean
    }

    bounds = region_bounds.get(
        region, [-20, -180, 60, 180]
    )  # Default: Global tropical/mid-latitude

    # Determine satellite based on region
    if region == "atlantic":
        bucket = "noaa-goes16"  # GOES-East covers Atlantic
    elif region == "pacific":
        bucket = "noaa-goes17"  # GOES-West covers Pacific
    else:
        bucket = "noaa-goes16"  # Default to GOES-16

    # Parse date range
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

    # Generate all dates in range
    all_dates = []
    current_date = start_date_obj
    while current_date <= end_date_obj:
        all_dates.append(current_date)
        current_date += timedelta(days=1)

    # Define async function to fetch data for a single day
    async def fetch_day(date_obj):
        year = date_obj.year
        day_of_year = date_obj.timetuple().tm_yday
        hour = 12  # Use noon data
        date_str = date_obj.strftime("%Y-%m-%d")

        datasets_to_fetch = [
            ("ABI-L2-MCMIPC", "Cloud and Moisture Imagery"),  # Multi-band imagery
            ("ABI-L2-TPWC", "Total Precipitable Water"),  # Atmospheric moisture
        ]

        day_data = []

        for product, description in datasets_to_fetch:
            # Construct S3 path
            s3_path = f"{bucket}/{product}/{year}/{day_of_year:03d}/{hour:02d}/"

            # List available files
            files = s3.ls(s3_path)

            if not files:
                print(f"No files found for {product} on {date_str}, skipping")
                continue

            # Take first file (most recent)
            file_path = files[0]

            # Open with xarray and load data into memory before file handle closes
            with s3.open(file_path, "rb") as f:
                ds = xr.open_dataset(f, engine="h5netcdf")
                ds.load()  # Load data into memory before file handle closes
                day_data.append(ds)

        # If we successfully fetched data for this day, merge and add to collection
        if day_data:
            # Merge datasets for this day
            daily_ds = xr.merge(day_data, compat="override")
            # Add time coordinate for this day
            daily_ds = daily_ds.expand_dims(time=[date_obj])
            return daily_ds
        else:
            print(f"No data available for {date_str}, skipping")
            return None

    # Fetch all days in parallel
    daily_datasets_raw = await asyncio.gather(*[fetch_day(date) for date in all_dates])

    # Filter out None values
    daily_datasets = [ds for ds in daily_datasets_raw if ds is not None]

    # Check if we have any data
    if not daily_datasets:
        raise RuntimeError(
            f"No satellite data available for {region} from {start_date} to {end_date}. "
            f"Check that GOES satellite data exists for these dates and region."
        )

    # Concatenate all days along time dimension
    combined_ds = xr.concat(daily_datasets, dim="time")

    # Apply geographic filtering to region bounds
    lat_coord = None
    lon_coord = None

    # GOES satellites use various coordinate naming conventions
    for lat_name in ["latitude", "lat", "y"]:
        if lat_name in combined_ds.coords:
            lat_coord = lat_name
            break

    for lon_name in ["longitude", "lon", "x"]:
        if lon_name in combined_ds.coords:
            lon_coord = lon_name
            break

    if lat_coord and lon_coord:
        # Check if lat/lon are dimension coordinates (1D, indexed) or non-dimension (2D arrays)
        lat_is_dim = lat_coord in combined_ds.dims
        lon_is_dim = lon_coord in combined_ds.dims

        # For 2D lat/lon arrays (typical for GOES satellite data), use x/y filtering
        if not (lat_is_dim and lon_is_dim):
            # Get lat/lon data (could be coords or data_vars)
            lat_data = None
            lon_data = None

            if "latitude" in combined_ds.coords:
                lat_data = combined_ds.coords["latitude"].values
            elif "latitude" in combined_ds.data_vars:
                lat_data = combined_ds["latitude"].values

            if "longitude" in combined_ds.coords:
                lon_data = combined_ds.coords["longitude"].values
            elif "longitude" in combined_ds.data_vars:
                lon_data = combined_ds["longitude"].values

            if lat_data is not None and lon_data is not None:
                # Create mask for the region
                lat_mask = (lat_data >= bounds[0]) & (lat_data <= bounds[2])
                lon_mask = (lon_data >= bounds[1]) & (lon_data <= bounds[3])
                region_mask = lat_mask & lon_mask

                # Apply mask to x and y dimensions
                if region_mask.ndim == 2:
                    # Find the bounding box of True values in the mask
                    rows = region_mask.any(axis=1)
                    cols = region_mask.any(axis=0)
                    y_indices = rows.nonzero()[0]
                    x_indices = cols.nonzero()[0]

                    if len(y_indices) > 0 and len(x_indices) > 0:
                        y_min, y_max = y_indices.min(), y_indices.max() + 1
                        x_min, x_max = x_indices.min(), x_indices.max() + 1

                        # Crop the dataset using x/y indices
                        if "y" in combined_ds.dims and "x" in combined_ds.dims:
                            combined_ds = combined_ds.isel(
                                y=slice(y_min, y_max), x=slice(x_min, x_max)
                            )
                        else:
                            print(
                                "Warning: Could not find x/y dimensions, using full dataset"
                            )
                    else:
                        print(
                            "Warning: No data found in region bounds, using full dataset"
                        )
                else:
                    print("Warning: Unable to apply 2D mask, using full dataset")
            else:
                print(
                    "Warning: Could not find lat/lon data for filtering, using full dataset"
                )
        else:
            # Direct lat/lon coordinates - use simple selection
            try:
                combined_ds = combined_ds.sel(
                    {
                        lat_coord: slice(bounds[0], bounds[2]),
                        lon_coord: slice(bounds[1], bounds[3]),
                    }
                )
            except Exception as e:
                print(
                    f"Warning: Could not filter by coordinates ({e}), using full dataset"
                )
    else:
        print("Warning: Could not find lat/lon coordinates, using full dataset")

    # Save to NetCDF
    output = File.new_remote(file_name="satellite.nc")

    # Convert to bytes and write
    buffer = io.BytesIO()
    combined_ds.to_netcdf(buffer)
    buffer.seek(0)

    async with output.open("wb") as f:
        await f.write(buffer.read())

    return output
# {{/docs-fragment ingest-satellite}}


# {{docs-fragment ingest-reanalysis}}
@cpu_env.task
async def ingest_reanalysis_data(region: str, date_range: list[str, str]) -> File:
    """Fetch ERA5 reanalysis from Copernicus Climate Data Store."""
    # {{/docs-fragment ingest-reanalysis}}
    from ecmwf.datastores import Client

    start_date, end_date = date_range[0], date_range[1]

    # Initialize new ECMWF datastores client
    client = Client()

    # Parse date range
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

    region_bounds = {
        "atlantic": [
            -10,
            -100,
            60,
            20,
        ],  # [S, W, N, E] - Atlantic Ocean + Caribbean + North America coast
        "pacific": [-20, -180, 50, -100],  # Eastern + Western Pacific
        "indian": [-40, 40, 30, 120],  # Indian Ocean
    }

    bounds = region_bounds.get(
        region, [-20, -180, 60, 180]
    )  # Default: Global tropical/mid-latitude

    # ERA5 API expects [N, W, S, E] but our bounds are [S, W, N, E]
    # Convert to correct format
    bounds_era5 = [bounds[2], bounds[1], bounds[0], bounds[3]]  # [N, W, S, E]

    pressure_levels = [
        "1000",
        "950",
        "925",
        "900",
        "850",
        "800",
        "750",
        "700",
        "650",
        "600",
        "550",
        "500",
        "450",
        "400",
        "350",
        "300",
        "250",
        "200",
        "150",
        "100",
        "50",
    ]

    # Generate all dates in range
    all_dates = []
    current_date = start_date_obj
    while current_date <= end_date_obj:
        all_dates.append(current_date)
        current_date += timedelta(days=1)

    # Define async function to fetch data for a single day
    async def fetch_era5_day(date_obj):
        """Fetch ERA5 data for a single day (runs in thread pool to avoid blocking)"""
        year = date_obj.strftime("%Y")
        month = date_obj.strftime("%m")
        day = date_obj.strftime("%d")
        date_str = date_obj.strftime("%Y-%m-%d")

        # Create temporary file for this day's download
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".grib")
        temp_path = temp_file.name
        temp_file.close()

        try:
            # Request ERA5 data for this day
            request = {
                "product_type": "reanalysis",
                "data_format": "grib",
                "variable": [
                    "temperature",  # Air temperature
                    "u_component_of_wind",  # Zonal wind
                    "v_component_of_wind",  # Meridional wind
                    "geopotential",  # For altitude calculation
                    "relative_humidity",
                ],
                "pressure_level": pressure_levels,
                "year": [year],
                "month": [month],
                "day": [day],
                "time": ["00:00"],  # Start of day
                "area": bounds_era5,
            }

            # Run blocking retrieve call in thread pool
            await asyncio.to_thread(
                client.retrieve,
                "reanalysis-era5-pressure-levels",
                request,
                target=temp_path,
            )

            # Read first few bytes to check format
            with open(temp_path, "rb") as f:
                magic_bytes = f.read(4)

            # Load the downloaded GRIB file
            ds = await asyncio.to_thread(xr.open_dataset, temp_path, engine="cfgrib")
            ds.load()  # Load into memory

            # Add time coordinate for this day
            ds = ds.expand_dims(time=[date_obj])
            return ds
        finally:
            # Cleanup temp file
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    # Fetch all days in parallel
    daily_datasets = await asyncio.gather(*[fetch_era5_day(date) for date in all_dates])

    # Check if we have any data
    if not daily_datasets:
        raise RuntimeError(
            f"No ERA5 data available for {region} from {start_date} to {end_date}."
        )

    # Concatenate all days along time dimension
    combined_ds = xr.concat(daily_datasets, dim="time")

    # Clean up any invalid values that could cause NetCDF write errors
    for var_name in combined_ds.data_vars:
        data = combined_ds[var_name]
        n_nan = (
            data.isnull().sum().compute()
            if hasattr(data, "compute")
            else data.isnull().sum()
        )
        n_inf = np.isinf(data.values).sum() if hasattr(data.values, "sum") else 0
        if n_nan > 0 or n_inf > 0:
            print(f"  {var_name}: {n_nan} NaN, {n_inf} Inf values - filling with 0")
            combined_ds[var_name] = data.fillna(0)
            if n_inf > 0:
                combined_ds[var_name] = combined_ds[var_name].where(
                    np.isfinite(combined_ds[var_name]), 0
                )

    # Save combined dataset to a temporary file
    temp_fd, final_temp_path = tempfile.mkstemp(suffix=".nc")
    os.close(temp_fd)

    # Remove the empty file created by mkstemp - let xarray create it fresh
    try:
        os.unlink(final_temp_path)
    except:
        pass

    # Strict: write as NETCDF4, fail clearly if there are issues
    combined_ds.to_netcdf(final_temp_path, format="NETCDF4", engine="netcdf4")

    # Verify the file is readable before returning
    try:
        test_ds = xr.open_dataset(final_temp_path, engine="netcdf4")
        test_ds.close()
    except Exception as e:
        raise RuntimeError(f"Failed to verify written NetCDF file: {e}")

    # Return as Flyte file
    return await File.from_local(final_temp_path)
# {{/docs-fragment ingest-reanalysis}}


# {{docs-fragment ingest-station}}
@cpu_env.task
async def ingest_station_data(
    region: str, date_range: list[str, str], max_stations: int = 100
) -> File:
    """Fetch ground observations from NOAA's Integrated Surface Database."""
    # {{/docs-fragment ingest-station}}
    import aiohttp
    import s3fs

    start_date, end_date = date_range[0], date_range[1]

    # Define broader region bounds for filtering stations
    region_bounds = {
        "atlantic": {
            "lat": (-10, 60),
            "lon": (-100, 20),
        },  # Atlantic Ocean + Caribbean + North America coast
        "pacific": {"lat": (-20, 50), "lon": (-180, -100)},  # Eastern + Western Pacific
        "indian": {"lat": (-40, 30), "lon": (40, 120)},  # Indian Ocean
    }

    bounds = region_bounds.get(region, {"lat": (-20, 60), "lon": (-180, 180)})

    # Parse date range
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

    # Determine which years we need to fetch data for
    years_needed = set()
    current_date = start_date_obj
    while current_date <= end_date_obj:
        years_needed.add(current_date.year)
        current_date += timedelta(days=1)

    s3 = s3fs.S3FileSystem(anon=True)

    # Fetch station metadata once (shared across all years)
    stations_url = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.txt"

    station_metadata = {}
    async with aiohttp.ClientSession() as session:
        async with session.get(stations_url) as response:
            if response.status == 200:
                stations_text = await response.text()
                lines = stations_text.split("\n")[22:]  # Skip header

                for line in lines:
                    if len(line) < 80:
                        continue
                    try:
                        station_id = line[0:12].strip()
                        lat = float(line[57:64].strip())
                        lon = float(line[65:73].strip())
                        name = line[13:43].strip()
                        country = line[43:46].strip()

                        station_metadata[station_id] = {
                            "lat": lat,
                            "lon": lon,
                            "name": name,
                            "country": country,
                        }
                    except (ValueError, IndexError):
                        continue

    # Collect stations from all years
    all_stations = {}

    for year in sorted(years_needed):
        # List all CSV files for this year
        s3_prefix = f"noaa-global-hourly-pds/{year}/"

        try:
            all_files = s3.ls(s3_prefix)

            # Extract station IDs from filenames and limit to max_stations
            for file_path in all_files[:max_stations]:
                # Extract filename: "noaa-global-hourly-pds/2024/72506414768.csv" -> "72506414768"
                filename = file_path.split("/")[-1]
                if filename.endswith(".csv"):
                    station_id_filename = filename.replace(".csv", "")
                    # Convert back to ISD format with space: "72506414768" -> "725064 14768"
                    if len(station_id_filename) >= 11:
                        station_id_formatted = (
                            f"{station_id_filename[:6]} {station_id_filename[6:]}"
                        )

                        # Get metadata and filter by region
                        metadata = station_metadata.get(station_id_formatted, {})

                        # Filter by geographic bounds if we have coordinates
                        if metadata:
                            lat = metadata.get("lat", 0)
                            lon = metadata.get("lon", 0)

                            if not (
                                bounds["lat"][0] <= lat <= bounds["lat"][1]
                                and bounds["lon"][0] <= lon <= bounds["lon"][1]
                            ):
                                continue

                        # Add to all_stations dict (keyed by station_id to avoid duplicates)
                        if station_id_formatted not in all_stations:
                            all_stations[station_id_formatted] = {
                                "station_id": station_id_formatted,
                                "station_id_filename": station_id_filename,
                                "name": metadata.get("name", "Unknown Station"),
                                "country": metadata.get("country", ""),
                                "lat": metadata.get("lat", 0),
                                "lon": metadata.get("lon", 0),
                                "years": set(),
                            }
                        all_stations[station_id_formatted]["years"].add(year)
        except Exception as e:
            print(f"Error listing stations for {year}: {e}")
            continue

    stations = list(all_stations.values())

    # Fallback: if no stations in region, use available stations regardless of location
    if not stations:
        print(
            f"No stations found in region bounds, using up to {max_stations} available stations"
        )

        all_stations_no_filter = {}

        for year in sorted(years_needed):
            s3_prefix = f"noaa-global-hourly-pds/{year}/"

            try:
                all_files = s3.ls(s3_prefix)

                for file_path in all_files[:max_stations]:
                    filename = file_path.split("/")[-1]
                    if filename.endswith(".csv"):
                        station_id_filename = filename.replace(".csv", "")
                        if len(station_id_filename) >= 11:
                            station_id_formatted = (
                                f"{station_id_filename[:6]} {station_id_filename[6:]}"
                            )
                            metadata = station_metadata.get(station_id_formatted, {})

                            if station_id_formatted not in all_stations_no_filter:
                                all_stations_no_filter[station_id_formatted] = {
                                    "station_id": station_id_formatted,
                                    "station_id_filename": station_id_filename,
                                    "name": metadata.get("name", "Unknown Station"),
                                    "country": metadata.get("country", ""),
                                    "lat": metadata.get("lat", 0),
                                    "lon": metadata.get("lon", 0),
                                    "years": set(),
                                }
                            all_stations_no_filter[station_id_formatted]["years"].add(
                                year
                            )
            except Exception as e:
                print(f"Error listing stations for {year}: {e}")
                continue

        stations = list(all_stations_no_filter.values())

    if not stations:
        raise RuntimeError(
            f"No station data available for any region from {start_date} to {end_date}. "
            f"Verify station data is available on AWS S3 for this date range."
        )

    # Fetch observations from all stations, filtering by date range
    observations = []

    for station in stations:
        station_id = station["station_id"]
        station_id_filename = station["station_id_filename"]
        station_years = station["years"]

        # Fetch data from all years this station has data for
        for year in station_years:
            try:
                # ISD data path: s3://noaa-global-hourly-pds/YEAR/STATIONID.csv
                s3_path = f"noaa-global-hourly-pds/{year}/{station_id_filename}.csv"

                with s3.open(s3_path, "r") as f:
                    station_data = pd.read_csv(f)

                    station_data["DATE"] = pd.to_datetime(
                        station_data["DATE"], errors="coerce"
                    )

                    # Filter observations within our date range
                    mask = (station_data["DATE"] >= start_date_obj) & (
                        station_data["DATE"] <= end_date_obj
                    )
                    filtered_data = station_data[mask]

                    if len(filtered_data) == 0:
                        continue

                    # Extract relevant observations
                    for _, row in filtered_data.iterrows():
                        obs = {
                            "station_id": station_id,
                            "station_name": station["name"],
                            "lat": station["lat"],
                            "lon": station["lon"],
                            "datetime": row.get("DATE"),
                            "temperature_c": row.get("TMP", np.nan),
                            "dewpoint_c": row.get("DEW", np.nan),
                            "pressure_mb": row.get("SLP", np.nan),
                            "wind_speed_mps": row.get("WND", np.nan),
                            "wind_direction": row.get("WND_DIR", np.nan),
                            "visibility_m": row.get("VIS", np.nan),
                        }
                        observations.append(obs)
            except Exception as e:
                continue

    if not observations:
        raise RuntimeError(
            f"No station observations collected for {region} from {start_date} to {end_date}. "
            f"Verify station data is available on AWS S3 for this region and date range."
        )

    stations_df = pd.DataFrame(observations)

    # Clean and convert data
    if "temperature_c" in stations_df.columns:
        # ISD stores temp in tenths of degrees C
        stations_df["temperature_c"] = (
            pd.to_numeric(stations_df["temperature_c"], errors="coerce") / 10.0
        )

    if "pressure_mb" in stations_df.columns:
        stations_df["pressure_mb"] = (
            pd.to_numeric(stations_df["pressure_mb"], errors="coerce") / 10.0
        )

    if "wind_speed_mps" in stations_df.columns:
        stations_df["wind_speed_mps"] = (
            pd.to_numeric(stations_df["wind_speed_mps"], errors="coerce") / 10.0
        )

    # Save to Parquet
    output = File.new_remote(file_name="station_observations.parquet")

    buffer = io.BytesIO()
    stations_df.to_parquet(buffer, index=False)
    buffer.seek(0)

    async with output.open("wb") as f:
        await f.write(buffer.read())

    return output
# {{/docs-fragment ingest-station}}


# {{docs-fragment preprocess}}
@dask_env.task
async def preprocess_atmospheric_data(
    satellite_data: File,
    reanalysis_data: File,
    station_data: File,
    target_resolution_km: float,
) -> File:
    """Preprocess atmospheric data using Dask for lazy/chunked operations on larger datasets"""
    from distributed import Client, wait

    # Connect to Dask client
    client = Client()

    def open_netcdf_from_bytes(data: bytes, description: str) -> xr.Dataset:
        # 1) Try h5netcdf directly from BytesIO
        try:
            ds = xr.open_dataset(io.BytesIO(data), engine="h5netcdf")
            return ds
        except Exception as e:
            print(f"{description}: BytesIO + h5netcdf failed: {e!r}")

        # 2) Fall back to a temp file + default/netcdf4 engine
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            ds = xr.open_dataset(
                tmp_path
            )  # let xarray choose engine or specify "netcdf4"
            return ds
        except Exception as e:
            raise

    async with satellite_data.open("rb") as f:
        sat_bytes = await f.read()

    # Open satellite with h5netcdf from BytesIO
    sat_ds = xr.open_dataset(
        io.BytesIO(sat_bytes),
        engine="h5netcdf",
    )

    # Rechunk explicitly
    sat_chunk_dict = {}
    if "time" in sat_ds.dims:
        sat_chunk_dict["time"] = 1
    for dim in ["latitude", "lat", "y"]:
        if dim in sat_ds.dims:
            sat_chunk_dict[dim] = 50
            break
    for dim in ["longitude", "lon", "x"]:
        if dim in sat_ds.dims:
            sat_chunk_dict[dim] = 50
            break

    if sat_chunk_dict:
        sat_ds = sat_ds.chunk(sat_chunk_dict)

    async with reanalysis_data.open("rb") as f:
        reanalysis_bytes = await f.read()

    reanalysis_ds = open_netcdf_from_bytes(reanalysis_bytes, "reanalysis")

    # Rechunk for Dask
    chunk_spec = {}
    if "time" in reanalysis_ds.dims:
        chunk_spec["time"] = 1
    for dim in ["latitude", "lat", "y"]:
        if dim in reanalysis_ds.dims:
            chunk_spec[dim] = 50
            break
    for dim in ["longitude", "lon", "x"]:
        if dim in reanalysis_ds.dims:
            chunk_spec[dim] = 50
            break

    if chunk_spec:
        reanalysis_ds = reanalysis_ds.chunk(chunk_spec)

    async with station_data.open("rb") as f:
        station_bytes = await f.read()
    stations_df = pd.read_parquet(io.BytesIO(station_bytes))

    lat_coords = None
    lon_coords = None

    for coord_name in ["latitude", "lat", "y"]:
        if coord_name in reanalysis_ds.coords:
            lat_coords = reanalysis_ds.coords[coord_name]
            break

    for coord_name in ["longitude", "lon", "x"]:
        if coord_name in reanalysis_ds.coords:
            lon_coords = reanalysis_ds.coords[coord_name]
            break

    if lat_coords is not None and lon_coords is not None:
        merged_ds = reanalysis_ds.copy()

        try:
            existing_vars_lower = {str(v).lower() for v in merged_ds.data_vars}
            satellite_vars_to_add = [
                var
                for var in sat_ds.data_vars
                if str(var).lower() not in existing_vars_lower
            ]

            if satellite_vars_to_add:
                for var in satellite_vars_to_add:
                    merged_ds[var] = sat_ds[var]
            else:
                print("No unique satellite variables to merge")
        except Exception as e:
            print(f"Could not merge satellite vars: {e}")
    else:
        print("Could not find lat/lon coordinates; using reanalysis only")
        merged_ds = reanalysis_ds.copy()

    # Quality control
    for var in merged_ds.data_vars:
        data_array = merged_ds[var]
        n_missing = int(data_array.isnull().sum().compute())

        if n_missing > 0:
            filled = data_array.ffill(dim="time").bfill(dim="time")
            merged_ds[var] = filled.fillna(data_array.mean())

    # Station metadata
    if len(stations_df) > 0:
        merged_ds.attrs["station_observations"] = len(stations_df)
        merged_ds.attrs["station_coverage"] = (
            f"{stations_df['lat'].min():.1f}-{stations_df['lat'].max():.1f}°N"
        )

    merged_ds.attrs["preprocessed"] = "true"
    merged_ds.attrs["target_resolution_km"] = target_resolution_km
    merged_ds.attrs["preprocessing_timestamp"] = datetime.now().isoformat()

    merged_ds = merged_ds.unify_chunks()

    merged_ds = merged_ds.persist()
    wait(merged_ds)

    output = File.new_remote(file_name="climate_preprocessed.nc")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp:
        temp_nc_path = tmp.name

    merged_ds.to_netcdf(
        temp_nc_path, engine="h5netcdf", invalid_netcdf=True, compute=True
    )

    with open(temp_nc_path, "rb") as f:
        nc_bytes = f.read()

    async with output.open("wb") as f:
        await f.write(nc_bytes)

    os.unlink(temp_nc_path)
    return output
# {{/docs-fragment preprocess}}


# {{docs-fragment gpu-simulation-signature}}
@gpu_env.task
async def run_atmospheric_simulation(
    input_data: File,
    params: SimulationParams,
    partition_id: int = 0,
    ensemble_start: int | None = None,
    ensemble_end: int | None = None,
) -> tuple[File, ClimateMetrics]:
    """Run GPU-accelerated atmospheric simulation with ensemble forecasting."""
    # {{/docs-fragment gpu-simulation-signature}}
    import torch

    # Determine ensemble subset for this GPU
    if ensemble_start is not None and ensemble_end is not None:
        actual_ensemble_size = ensemble_end - ensemble_start
    else:
        actual_ensemble_size = params.ensemble_size
        ensemble_start = 0
        ensemble_end = params.ensemble_size

    device = torch.device("cuda:0")

    # Load atmospheric state from NetCDF
    async with input_data.open("rb") as f:
        data_bytes = await f.read()

    # Parse NetCDF data
    buffer = io.BytesIO(data_bytes)
    ds = xr.open_dataset(buffer)

    # Show geographic bounds if available
    for coord_name in ["latitude", "lat", "y"]:
        if coord_name in ds.coords:
            break
    for coord_name in ["longitude", "lon", "x"]:
        if coord_name in ds.coords:
            break

    # Extract atmospheric variables
    # Map from various possible variable names to our standard names
    var_mapping = {
        "temperature": ["temperature", "t", "temp", "T"],
        "pressure": ["pressure", "p", "press", "P"],
        "humidity": ["humidity", "rh", "relative_humidity", "q"],
        "wind_u": ["wind_u", "u", "u_wind", "U"],
        "wind_v": ["wind_v", "v", "v_wind", "V"],
    }

    # Find and load variables (two-pass approach for consistent shapes)
    atmospheric_data = {}

    # First pass: load all variables that exist in the dataset
    for standard_name, possible_names in var_mapping.items():
        for var_name in possible_names:
            if var_name in ds.data_vars:
                atmospheric_data[standard_name] = ds[var_name].values
                break

    # Determine reference shape from found variables or use default
    if atmospheric_data:
        reference_shape = list(atmospheric_data.values())[0].shape
    else:
        reference_shape = (1000, 1000, 50)  # Default: 1000km × 1000km × 50 levels
        print(f"No variables found, using default shape: {reference_shape}")

    # Second pass: initialize missing variables with reference shape
    for standard_name in var_mapping.keys():
        if standard_name not in atmospheric_data:
            print(f"Variable '{standard_name}' not found, initializing with zeros")
            atmospheric_data[standard_name] = np.zeros(reference_shape)

    # Check for NaN values in loaded data and handle them
    for standard_name, data in atmospheric_data.items():
        n_nans = np.isnan(data).sum()
        if n_nans > 0:
            pct_nan = (n_nans / data.size) * 100
            # Fill NaNs with mean of non-NaN values
            if pct_nan < 100:
                mean_val = np.nanmean(data)
                atmospheric_data[standard_name] = np.nan_to_num(data, nan=mean_val)
            else:
                # All NaN - use reasonable defaults
                if standard_name == "temperature":
                    atmospheric_data[standard_name] = np.full_like(data, 288.0)  # 15°C
                elif standard_name == "pressure":
                    atmospheric_data[standard_name] = np.full_like(
                        data, 1013.0
                    )  # Sea level
                elif standard_name == "humidity":
                    atmospheric_data[standard_name] = np.full_like(data, 50.0)  # 50% RH
                else:
                    atmospheric_data[standard_name] = np.zeros_like(data)
        else:
            print(f"{standard_name}: No NaN values")

    # Get dimensions
    first_var = list(atmospheric_data.values())[0]
    if len(first_var.shape) == 4:
        # 4D data: (time, levels, lat, lon) - select first time step
        # Update all variables to use first time step
        for key in atmospheric_data.keys():
            atmospheric_data[key] = atmospheric_data[key][0]  # Select t=0
        first_var = atmospheric_data[list(atmospheric_data.keys())[0]]
        n_levels, grid_x, grid_y = first_var.shape
    elif len(first_var.shape) == 3:
        grid_x, grid_y, n_levels = first_var.shape
    elif len(first_var.shape) == 2:
        grid_x, grid_y = first_var.shape
        n_levels = 1
    else:
        raise ValueError(f"Unexpected data shape: {first_var.shape}")

    print(f"Grid: {grid_x} × {grid_y} × {n_levels} levels")

    # Calculate expected data size
    total_elements = grid_x * grid_y * n_levels * 5  # 5 variables

    # Convert to PyTorch tensors on GPU
    temperature = torch.tensor(
        atmospheric_data["temperature"], device=device, dtype=torch.float32
    )
    pressure = torch.tensor(
        atmospheric_data["pressure"], device=device, dtype=torch.float32
    )
    humidity = torch.tensor(
        atmospheric_data.get(
            "humidity", np.zeros_like(atmospheric_data["temperature"])
        ),
        device=device,
        dtype=torch.float32,
    )
    wind_u = torch.tensor(
        atmospheric_data.get("wind_u", np.zeros_like(atmospheric_data["temperature"])),
        device=device,
        dtype=torch.float32,
    )
    wind_v = torch.tensor(
        atmospheric_data.get("wind_v", np.zeros_like(atmospheric_data["temperature"])),
        device=device,
        dtype=torch.float32,
    )

    # Stack into single state tensor: [variables, x, y, z]
    base_state = torch.stack([temperature, pressure, humidity, wind_u, wind_v], dim=0)

    # Generate ensemble members with perturbed initial conditions
    # Each member is an independent forecast
    # Use ensemble_start to ensure reproducible but different perturbations across GPUs

    ensemble_states = []

    # Generate only the ensemble members assigned to this GPU
    for local_idx, member_id in enumerate(range(ensemble_start, ensemble_end)):
        if member_id == 0:
            # Member 0: Control run (unperturbed)
            perturbed_state = base_state.clone()
        else:
            # Add random perturbations to initial conditions
            # This represents uncertainty in observations/analysis
            torch.manual_seed(42 + member_id)  # Reproducible perturbations across GPUs

            perturbed_state = base_state.clone()

            # Perturb temperature (±0.5K typical)
            temp_perturbation = (
                torch.randn_like(perturbed_state[0]) * params.perturbation_magnitude
            )
            perturbed_state[0] = perturbed_state[0] + temp_perturbation

            # Perturb pressure (scaled appropriately)
            pressure_perturbation = torch.randn_like(perturbed_state[1]) * (
                params.perturbation_magnitude * 0.1
            )
            perturbed_state[1] = perturbed_state[1] + pressure_perturbation

            # Perturb humidity (scaled)
            humidity_perturbation = torch.randn_like(perturbed_state[2]) * (
                params.perturbation_magnitude * 0.5
            )
            perturbed_state[2] = torch.clamp(
                perturbed_state[2] + humidity_perturbation, min=0
            )

            # Perturb winds (±0.2 m/s typical)
            wind_perturbation = torch.randn_like(perturbed_state[3]) * (
                params.perturbation_magnitude * 0.4
            )
            perturbed_state[3] = perturbed_state[3] + wind_perturbation
            perturbed_state[4] = perturbed_state[4] + wind_perturbation

            if local_idx < 5 or member_id == ensemble_end - 1:
                print(
                    f"Member {member_id:2d}: Perturbed (δT={temp_perturbation.std().item():.3f}K)"
                )

        ensemble_states.append(perturbed_state)

    if actual_ensemble_size > 10:
        print(f"  ... ({actual_ensemble_size - 6} more members) ...")

    # Stack ensemble members: [ensemble_members, variables, x, y, z]
    # Each member evolves INDEPENDENTLY through time
    state = torch.stack(ensemble_states, dim=0)

    # Explicitly verify tensors are on GPU
    print(f"State tensor device: {state.device}")
    print(f"State tensor dtype: {state.dtype}")

    torch.cuda.synchronize()  # Ensure all CUDA operations complete

    # Simulation loop - run full simulation
    n_timesteps = (params.simulation_hours * 60) // params.time_step_minutes
    detected_phenomena = []
    start_time = datetime.now()

    # Physics constants
    dt = params.time_step_minutes * 60  # Convert to seconds
    dx = params.grid_resolution_km * 1000  # Convert to meters

    # {{docs-fragment physics-step}}
    @torch.compile(mode="reduce-overhead")
    def physics_step(state_tensor, dt_val, dx_val):
        """Compiled atmospheric physics - 3-4x faster with torch.compile."""
        # Advection: transport by wind
        temp_grad_x = torch.roll(state_tensor[:, 0], -1, dims=2) - torch.roll(
            state_tensor[:, 0], 1, dims=2
        )
        temp_grad_y = torch.roll(state_tensor[:, 0], -1, dims=3) - torch.roll(
            state_tensor[:, 0], 1, dims=3
        )
        advection = -(
            state_tensor[:, 3] * temp_grad_x + state_tensor[:, 4] * temp_grad_y
        ) / (2 * dx_val)
        state_tensor[:, 0] = state_tensor[:, 0] + advection * dt_val

        # Pressure gradient with Coriolis
        pressure_grad_x = (
            torch.roll(state_tensor[:, 1], -1, dims=2)
            - torch.roll(state_tensor[:, 1], 1, dims=2)
        ) / (2 * dx_val)
        pressure_grad_y = (
            torch.roll(state_tensor[:, 1], -1, dims=3)
            - torch.roll(state_tensor[:, 1], 1, dims=3)
        ) / (2 * dx_val)

        coriolis_param = 1e-4  # ~45°N latitude
        coriolis_u = coriolis_param * state_tensor[:, 4]
        coriolis_v = -coriolis_param * state_tensor[:, 3]

        state_tensor[:, 3] = (
            state_tensor[:, 3] - pressure_grad_x * dt_val * 0.01 + coriolis_u * dt_val
        )
        state_tensor[:, 4] = (
            state_tensor[:, 4] - pressure_grad_y * dt_val * 0.01 + coriolis_v * dt_val
        )

        # Turbulent diffusion
        diffusion_coeff = 10.0
        laplacian_temp = (
            torch.roll(state_tensor[:, 0], 1, dims=2)
            + torch.roll(state_tensor[:, 0], -1, dims=2)
            + torch.roll(state_tensor[:, 0], 1, dims=3)
            + torch.roll(state_tensor[:, 0], -1, dims=3)
            - 4 * state_tensor[:, 0]
        ) / (dx_val * dx_val)
        state_tensor[:, 0] = (
            state_tensor[:, 0] + diffusion_coeff * laplacian_temp * dt_val
        )

        # Moisture condensation
        sat_vapor_pressure = 611.2 * torch.exp(
            17.67 * state_tensor[:, 0] / (state_tensor[:, 0] + 243.5)
        )
        condensation = torch.clamp(
            state_tensor[:, 2] - sat_vapor_pressure * 0.001, min=0
        )
        state_tensor[:, 2] = state_tensor[:, 2] - condensation * 0.1
        state_tensor[:, 0] = state_tensor[:, 0] + condensation * 2.5e6 / 1005 * dt_val

        return state_tensor
    # {{/docs-fragment physics-step}}

    torch.cuda.reset_peak_memory_stats()

    for timestep in range(n_timesteps):
        # Apply physics with mixed precision (FP16) and compiled kernels
        with torch.cuda.amp.autocast():
            state = physics_step(state, dt, dx)

        # Check for extreme phenomena every 10 timesteps (across all ensemble members)
        if timestep % 10 == 0:
            # Check for NaN values (numerical instability)
            if torch.isnan(state).any():
                nan_count = torch.isnan(state).sum().item()
                # Replace NaNs with zeros to continue (emergency fallback)
                state = torch.nan_to_num(state, nan=0.0)

            # Calculate ensemble spread (measure of forecast uncertainty)
            ensemble_spread = state[:, 0].std(dim=0).mean().item()

            # Detect hurricane conditions
            # Count how many ensemble members predict hurricane
            hurricane_members = (
                (state[:, 2:4].abs().max(dim=1)[0].max(dim=1)[0] > 33).sum().item()
            )
            if hurricane_members > 0:
                hurricane_probability = (hurricane_members / actual_ensemble_size) * 100
                detected_phenomena.append("hurricane_detected")

            # Detect heatwave
            max_temp = state[:, 0].max().item()
            if max_temp > 2.0:  # Anomaly threshold
                detected_phenomena.append("heatwave_detected")

    torch.cuda.synchronize()
    compute_time = (datetime.now() - start_time).total_seconds()

    # Check for NaN/Inf in final state before computing statistics
    if torch.isnan(state).any() or torch.isinf(state).any():
        state = torch.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)

    ensemble_mean = state.mean(dim=0)  # [variables, x, y, z]
    ensemble_std = state.std(dim=0)  # [variables, x, y, z]

    # Ensure ensemble_std doesn't have NaN
    if torch.isnan(ensemble_std).any():
        ensemble_std = torch.nan_to_num(ensemble_std, nan=0.0)

    ensemble_spread = ensemble_std.mean().item()  # scalar measure of uncertainty

    # Final safety check
    if np.isnan(ensemble_spread):
        ensemble_spread = 0.0

    energy_error = abs(torch.sum(state[0] ** 2).item() - 1000000) / 1000000
    final_mean = ensemble_mean.cpu().numpy()
    final_std = ensemble_std.cpu().numpy()
    control_member = state[0].cpu().numpy()

    # Create xarray Dataset with proper structure
    if len(final_mean.shape) == 4:
        _, nx, ny, nz = final_mean.shape
        result_ds = xr.Dataset(
            {
                # Ensemble mean (most likely forecast)
                "temperature_mean": (["x", "y", "z"], final_mean[0]),
                "pressure_mean": (["x", "y", "z"], final_mean[1]),
                "humidity_mean": (["x", "y", "z"], final_mean[2]),
                "wind_u_mean": (["x", "y", "z"], final_mean[3]),
                "wind_v_mean": (["x", "y", "z"], final_mean[4]),
                # Ensemble spread (forecast uncertainty)
                "temperature_std": (["x", "y", "z"], final_std[0]),
                "pressure_std": (["x", "y", "z"], final_std[1]),
                "humidity_std": (["x", "y", "z"], final_std[2]),
                "wind_u_std": (["x", "y", "z"], final_std[3]),
                "wind_v_std": (["x", "y", "z"], final_std[4]),
                # Control member (unperturbed forecast)
                "temperature_control": (["x", "y", "z"], control_member[0]),
                "pressure_control": (["x", "y", "z"], control_member[1]),
                "humidity_control": (["x", "y", "z"], control_member[2]),
                "wind_u_control": (["x", "y", "z"], control_member[3]),
                "wind_v_control": (["x", "y", "z"], control_member[4]),
            },
            coords={
                "x": np.arange(nx),
                "y": np.arange(ny),
                "z": np.arange(nz),
            },
        )
    else:
        # Fallback for different shapes
        result_ds = xr.Dataset(
            {"atmospheric_state_mean": (list(range(len(final_mean.shape))), final_mean)}
        )

    # Add ensemble forecasting metadata
    result_ds.attrs["partition_id"] = partition_id
    result_ds.attrs["simulation_hours"] = params.simulation_hours
    result_ds.attrs["timesteps_completed"] = n_timesteps
    result_ds.attrs["grid_resolution_km"] = params.grid_resolution_km
    result_ds.attrs["physics_model"] = params.physics_model
    result_ds.attrs["ensemble_size"] = params.ensemble_size
    result_ds.attrs["ensemble_perturbation_magnitude"] = params.perturbation_magnitude
    result_ds.attrs["forecast_type"] = "probabilistic_ensemble"
    result_ds.attrs["completion_time"] = datetime.now().isoformat()
    result_ds.attrs["description"] = (
        f"Ensemble forecast with {params.ensemble_size} members. "
        f"Includes mean (most likely), std (uncertainty), and control run."
    )

    # Save as NetCDF
    output = File.new_remote(file_name=f"sim_results_{partition_id}.nc")

    buffer = io.BytesIO()
    result_ds.to_netcdf(buffer)
    buffer.seek(0)

    async with output.open("wb") as f:
        await f.write(buffer.read())

    # Helper to sanitize numeric values (replace nan/inf with valid values)
    def sanitize_float(value: float, default: float = 0.0) -> float:
        """Replace nan/inf values with default for safe serialization"""
        if np.isnan(value) or np.isinf(value):
            return default
        return float(value)

    # Compile metrics (use first timestep for scalar metrics)
    metrics = ClimateMetrics(
        timestamp=datetime.now().isoformat(),
        iteration=partition_id,
        convergence_rate=1.0,
        energy_conservation_error=sanitize_float(energy_error, default=0.0),
        max_wind_speed_mps=sanitize_float(
            state[:, 2:4].abs().max().item(), default=0.0
        ),
        min_pressure_mb=sanitize_float(
            state[:, 3].min().item() * 10 + 1000, default=1013.0
        ),
        detected_phenomena=list(set(detected_phenomena)),
        compute_time_seconds=sanitize_float(compute_time, default=0.0),
        ensemble_spread=sanitize_float(ensemble_spread, default=0.0),
    )

    del state, ensemble_mean, ensemble_std
    del control_member, final_mean, final_std
    del temperature, pressure, humidity, wind_u, wind_v, base_state
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return output, metrics
# {{/docs-fragment gpu-simulation}}


# {{docs-fragment distributed-ensemble}}
@cpu_env.task
async def run_distributed_simulation_ensemble(
    preprocessed_data: File, params: SimulationParams, n_gpus: int
) -> tuple[list[File], list[ClimateMetrics]]:
    total_members = params.ensemble_size
    members_per_gpu = total_members // n_gpus

    # Distribute ensemble members across GPUs
    tasks = []
    for gpu_id in range(n_gpus):
        # Calculate ensemble range for this GPU
        ensemble_start = gpu_id * members_per_gpu
        # Last GPU gets any remainder members
        if gpu_id == n_gpus - 1:
            ensemble_end = total_members
        else:
            ensemble_end = ensemble_start + members_per_gpu

        # Launch GPU task with ensemble subset
        gpu_task = run_atmospheric_simulation(
            preprocessed_data,
            params,
            gpu_id,
            ensemble_start=ensemble_start,
            ensemble_end=ensemble_end,
        )
        tasks.append(gpu_task)

    # Execute all GPUs in parallel
    results = await asyncio.gather(*tasks)

    output_files = [r[0] for r in results]
    metrics = [r[1] for r in results]

    return output_files, metrics
# {{/docs-fragment distributed-ensemble}}


# {{docs-fragment analytics}}
@flyte.trace
async def analyze_simulation_convergence(
    metrics: ClimateMetrics,
    threshold: float = 0.001,
) -> dict:
    """
    Real-time convergence analysis running alongside simulation
    """
    analysis = {
        "converged": metrics.convergence_rate < threshold,
        "convergence_rate": metrics.convergence_rate,
        "energy_conserved": metrics.energy_conservation_error < 0.01,
        "recommendation": (
            "continue" if metrics.convergence_rate >= threshold else "stop"
        ),
    }

    if not analysis["converged"]:
        print(f"Not converged yet: {metrics.convergence_rate:.6f} > {threshold}")
    else:
        print(f"Converged: {metrics.convergence_rate:.6f} < {threshold}")

    return analysis


@flyte.trace
async def detect_extreme_events(metrics: ClimateMetrics) -> dict:
    """Real-time detection of hurricanes, heatwaves, and other extreme events"""
    events = {
        "hurricanes": [],
        "heatwaves": [],
        "severe_weather": [],
    }

    # Detect hurricanes (wind speed > 33 m/s, low pressure)
    if metrics.max_wind_speed_mps > 33 and metrics.min_pressure_mb < 980:
        hurricane = {
            "type": "hurricane",
            "intensity": "category_"
            + str(min(5, int((metrics.max_wind_speed_mps - 33) / 10) + 1)),
            "max_wind_mps": metrics.max_wind_speed_mps,
            "min_pressure_mb": metrics.min_pressure_mb,
            "timestamp": metrics.timestamp,
        }
        events["hurricanes"].append(hurricane)
        print(f"Hurricane detected: {hurricane['intensity'].replace('_', ' ').title()}")
        print(
            f"Wind: {metrics.max_wind_speed_mps:.1f} m/s, Pressure: {metrics.min_pressure_mb:.1f} mb"
        )

    # Detect heatwaves
    if "heatwave_detected" in metrics.detected_phenomena:
        events["heatwaves"].append(
            {
                "type": "heatwave",
                "timestamp": metrics.timestamp,
            }
        )
        print(f"Heatwave detected")

    # Overall assessment
    total_events = sum(len(v) for v in events.values())
    print(f"Total extreme events detected: {total_events}")

    return events


def get_metrics_json(
    all_metrics: list[ClimateMetrics],
    all_events: list[dict],
    simulation_params: SimulationParams,
    iteration: int,
) -> str:
    """
    Serialize current simulation state to JSON for dynamic updates
    """
    # Calculate summary statistics
    if all_metrics:
        latest_metrics = all_metrics[-1]
        avg_convergence = sum(m.convergence_rate for m in all_metrics) / len(
            all_metrics
        )
        total_compute_time = sum(m.compute_time_seconds for m in all_metrics)

        # Count extreme events
        total_hurricanes = sum(len(e.get("hurricanes", [])) for e in all_events)
        total_heatwaves = sum(len(e.get("heatwaves", [])) for e in all_events)
        total_severe = sum(len(e.get("severe_weather", [])) for e in all_events)
    else:
        latest_metrics = None
        avg_convergence = 0
        total_compute_time = 0
        total_hurricanes = total_heatwaves = total_severe = 0

    # Build convergence history
    convergence_history = [
        {"iteration": i + 1, "rate": m.convergence_rate}
        for i, m in enumerate(all_metrics)
    ]

    # Build event timeline
    event_timeline = []
    for i, events in enumerate(all_events):
        n_events = len(events.get("hurricanes", [])) + len(events.get("heatwaves", []))
        if n_events > 0:
            event_timeline.append({"iteration": i + 1, "count": n_events})

    # Collect all events
    all_hurricanes = []
    all_heatwaves = []
    for events in all_events:
        all_hurricanes.extend(events.get("hurricanes", []))
        all_heatwaves.extend(events.get("heatwaves", []))

    # Performance metrics (last 4 partitions)
    performance_metrics = []
    for m in all_metrics[-4:]:
        performance_metrics.append(
            {
                "partition_id": m.iteration,
                "compute_time": m.compute_time_seconds,
                "energy_error": m.energy_conservation_error,
                "max_wind": m.max_wind_speed_mps,
                "min_pressure": m.min_pressure_mb,
            }
        )

    data = {
        "iteration": iteration + 1,
        "max_iterations": simulation_params.max_iterations,
        "converged": (
            latest_metrics.convergence_rate < simulation_params.convergence_threshold
            if latest_metrics
            else False
        ),
        "convergence_rate": latest_metrics.convergence_rate if latest_metrics else 0,
        "convergence_threshold": simulation_params.convergence_threshold,
        "avg_convergence": avg_convergence,
        "total_compute_time": total_compute_time,
        "convergence_history": convergence_history,
        "event_timeline": event_timeline,
        "total_hurricanes": total_hurricanes,
        "total_heatwaves": total_heatwaves,
        "total_severe": total_severe,
        "hurricanes": all_hurricanes,
        "heatwaves": all_heatwaves,
        "performance_metrics": performance_metrics,
        "params": {
            "resolution_km": simulation_params.grid_resolution_km,
            "timestep_min": simulation_params.time_step_minutes,
            "duration_hours": simulation_params.simulation_hours,
            "physics_model": simulation_params.physics_model,
            "boundary_layer": simulation_params.boundary_layer_scheme,
            "microphysics": simulation_params.microphysics_scheme,
        },
    }

    return json.dumps(data, indent=2)


def build_static_report_template(
    simulation_params: SimulationParams, region: str
) -> str:
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Climate Simulation Report - {region}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: #fff;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
            }}
            h1 {{
                text-align: center;
                font-size: 2.5em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            }}
            .subtitle {{
                text-align: center;
                font-size: 1.2em;
                margin-bottom: 30px;
                opacity: 0.9;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .stat-card {{
                background: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                padding: 20px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }}
            .stat-card h3 {{
                margin: 0 0 10px 0;
                font-size: 0.9em;
                text-transform: uppercase;
                opacity: 0.8;
            }}
            .stat-value {{
                font-size: 2.5em;
                font-weight: bold;
                margin: 10px 0;
            }}
            .stat-label {{
                font-size: 0.9em;
                opacity: 0.7;
            }}
            .chart-container {{
                background: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                backdrop-filter: blur(10px);
            }}
            .chart-container h2 {{
                margin-top: 0;
                color: #64c8ff;
            }}
            .bar-chart {{
                display: flex;
                align-items: flex-end;
                height: 200px;
                gap: 4px;
                padding: 10px 0;
            }}
            .bar {{
                flex: 1;
                background: linear-gradient(to top, #4ecdc4, #45b7d1);
                border-radius: 4px 4px 0 0;
                min-height: 2px;
                transition: all 0.3s ease;
            }}
            .bar:hover {{
                filter: brightness(1.3);
                transform: scaleY(1.05);
            }}
            .timeline {{
                position: relative;
                height: 60px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 10px;
                margin: 20px 0;
            }}
            .event-marker {{
                position: absolute;
                width: 4px;
                height: 100%;
                background: #ff6b6b;
                top: 0;
            }}
            .events-list {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }}
            .event-card {{
                background: rgba(255, 107, 107, 0.2);
                border-left: 4px solid #ff6b6b;
                border-radius: 8px;
                padding: 15px;
                backdrop-filter: blur(5px);
            }}
            .event-card.heatwave {{
                background: rgba(255, 165, 0, 0.2);
                border-left-color: #ffa500;
            }}
            .event-card h4 {{
                margin: 0 0 10px 0;
                font-size: 1.2em;
            }}
            .event-card p {{
                margin: 5px 0;
                font-size: 0.9em;
            }}
            .status-badge {{
                display: inline-block;
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 0.85em;
                font-weight: bold;
                margin-top: 10px;
            }}
            .status-running {{
                background: rgba(33, 150, 243, 0.3);
                border: 1px solid #2196f3;
            }}
            .status-converged {{
                background: rgba(76, 175, 80, 0.3);
                border: 1px solid #4caf50;
            }}
            .parameters {{
                background: rgba(255, 255, 255, 0.05);
                border-radius: 8px;
                padding: 15px;
                margin: 20px 0;
            }}
            .parameters h3 {{
                margin-top: 0;
                color: #ffd93d;
            }}
            .param-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 10px;
            }}
            .param-item {{
                padding: 10px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 5px;
            }}
            .param-item strong {{
                color: #64c8ff;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding: 20px;
                opacity: 0.7;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌍 Climate Simulation Report</h1>
            <p class="subtitle">H200 GPU-Accelerated Atmospheric Modeling - {region.replace('_', ' ').title()}</p>

            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Iteration</h3>
                    <div class="stat-value" id="iteration">-</div>
                    <div class="stat-label" id="iteration-label">of {simulation_params.max_iterations} max iterations</div>
                    <span class="status-badge" id="status-badge">🔄 Running</span>
                </div>

                <div class="stat-card">
                    <h3>Convergence Rate</h3>
                    <div class="stat-value" id="convergence-rate">-</div>
                    <div class="stat-label">Target: &lt; {simulation_params.convergence_threshold}</div>
                </div>

                <div class="stat-card">
                    <h3>Compute Time</h3>
                    <div class="stat-value" id="compute-time">-</div>
                    <div class="stat-label">Total GPU time</div>
                </div>
            </div>

            <div class="parameters">
                <h3>Simulation Parameters</h3>
                <div class="param-grid" id="params-grid">
                    <!-- Updated dynamically -->
                </div>
            </div>

            <div class="chart-container">
                <h2>🌀 Extreme Events Detection</h2>
                <div class="stats-grid" style="margin-bottom: 20px;">
                    <div class="stat-card">
                        <h3>🌀 Hurricanes</h3>
                        <div class="stat-value" id="total-hurricanes">0</div>
                    </div>
                    <div class="stat-card">
                        <h3>🔥 Heatwaves</h3>
                        <div class="stat-value" id="total-heatwaves">0</div>
                    </div>
                    <div class="stat-card">
                        <h3>⛈️ Severe Weather</h3>
                        <div class="stat-value" id="total-severe">0</div>
                    </div>
                </div>

                <h3>Detected Events</h3>
                <div class="events-list" id="events-list">
                    <p style="text-align: center; opacity: 0.5; grid-column: 1/-1;">
                        No extreme events detected yet. Monitoring continues...
                    </p>
                </div>
            </div>

            <div class="chart-container">
                <h2>🖥️ Performance Metrics</h2>
                <div class="stats-grid" id="performance-metrics">
                    <!-- Updated dynamically -->
                </div>
            </div>

            <div class="footer">
                <p>Generated by Flyte Climate Modeling Workflow</p>
                <p>Powered by NVIDIA H200 GPUs | Real-time atmospheric simulation</p>
            </div>
        </div>

        <script>
            function updateMetrics(data) {{
                try {{
                    if (!data || Object.keys(data).length === 0) {{
                        return; // No data yet
                    }}

                    // Update iteration stats
                    document.getElementById('iteration').textContent = data.iteration || '-';
                    document.getElementById('convergence-rate').textContent =
                        (data.convergence_rate || 0).toFixed(6);
                    document.getElementById('compute-time').textContent =
                        (data.total_compute_time || 0).toFixed(1) + 's';

                    // Update status badge
                    const statusBadge = document.getElementById('status-badge');
                    if (data.converged) {{
                        statusBadge.className = 'status-badge status-converged';
                        statusBadge.textContent = '✓ Converged';
                    }} else {{
                        statusBadge.className = 'status-badge status-running';
                        statusBadge.textContent = '🔄 Running';
                    }}

                    // Update parameters
                    if (data.params) {{
                        const paramsGrid = document.getElementById('params-grid');
                        paramsGrid.innerHTML = `
                            <div class="param-item"><strong>Resolution:</strong> ${{data.params.resolution_km}} km</div>
                            <div class="param-item"><strong>Timestep:</strong> ${{data.params.timestep_min}} min</div>
                            <div class="param-item"><strong>Duration:</strong> ${{data.params.duration_hours}} hours</div>
                            <div class="param-item"><strong>Physics Model:</strong> ${{data.params.physics_model}}</div>
                            <div class="param-item"><strong>Boundary Layer:</strong> ${{data.params.boundary_layer}}</div>
                            <div class="param-item"><strong>Microphysics:</strong> ${{data.params.microphysics}}</div>
                        `;
                    }}

                    // Update event counts
                    document.getElementById('total-hurricanes').textContent = data.total_hurricanes || 0;
                    document.getElementById('total-heatwaves').textContent = data.total_heatwaves || 0;
                    document.getElementById('total-severe').textContent = data.total_severe || 0;

                    // Update events list
                    const eventsList = document.getElementById('events-list');
                    let eventsHTML = '';

                    if (data.hurricanes && data.hurricanes.length > 0) {{
                        eventsHTML += data.hurricanes.map(h => `
                            <div class="event-card">
                                <h4>🌀 Hurricane - ${{(h.intensity || 'Unknown').replace(/_/g, ' ').replace(/\\b\\w/g, c => c.toUpperCase())}}</h4>
                                <p><strong>Wind Speed:</strong> ${{(h.max_wind_mps || 0).toFixed(1)}} m/s</p>
                                <p><strong>Pressure:</strong> ${{(h.min_pressure_mb || 0).toFixed(1)}} mb</p>
                                <p><strong>Time:</strong> ${{h.timestamp || 'N/A'}}</p>
                            </div>
                        `).join('');
                    }}

                    if (data.heatwaves && data.heatwaves.length > 0) {{
                        eventsHTML += data.heatwaves.map(hw => `
                            <div class="event-card heatwave">
                                <h4>🔥 Heatwave Detected</h4>
                                <p><strong>Time:</strong> ${{hw.timestamp || 'N/A'}}</p>
                                <p>Anomalously high temperatures detected</p>
                            </div>
                        `).join('');
                    }}

                    if (eventsHTML) {{
                        eventsList.innerHTML = eventsHTML;
                    }} else {{
                        eventsList.innerHTML = '<p style="text-align: center; opacity: 0.5; grid-column: 1/-1;">No extreme events detected yet. Monitoring continues...</p>';
                    }}

                    // Update performance metrics
                    if (data.performance_metrics && data.performance_metrics.length > 0) {{
                        const perfGrid = document.getElementById('performance-metrics');
                        perfGrid.innerHTML = data.performance_metrics.map(m => `
                            <div class="stat-card">
                                <h3>Partition ${{m.partition_id}}</h3>
                                <p><strong>Compute Time:</strong> ${{m.compute_time.toFixed(2)}}s</p>
                                <p><strong>Energy Error:</strong> ${{m.energy_error.toFixed(6)}}</p>
                                <p><strong>Max Wind:</strong> ${{m.max_wind.toFixed(1)}} m/s</p>
                                <p><strong>Min Pressure:</strong> ${{m.min_pressure.toFixed(1)}} mb</p>
                            </div>
                        `).join('');
                    }}

                }} catch (e) {{
                    console.error('Error updating report:', e);
                }}
            }}
        </script>
    </body>
    </html>
    """
    return html_content


@cpu_env.task
async def refine_mesh_for_extreme_events(
    preprocessed_data: File, current_params: SimulationParams
) -> tuple[File, SimulationParams]:
    """
    Adaptive mesh refinement: refines spatial resolution when extreme events detected.
    Creates 2x finer grid through interpolation for better accuracy around phenomena.
    """
    import xarray as xr

    print(f"Current resolution: {current_params.grid_resolution_km} km")

    # Load preprocessed data
    async with preprocessed_data.open("rb") as f:
        data_bytes = await f.read()

    # Use h5netcdf here to match how the file was written upstream
    try:
        ds = xr.open_dataset(io.BytesIO(data_bytes), engine="h5netcdf")
    except Exception as e:
        # Fallback: write to a temp file and let xarray auto-detect engine
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp:
            tmp.write(data_bytes)
            tmp_path = tmp.name

        try:
            ds = xr.open_dataset(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    lat_coord = None
    lon_coord = None

    for coord_name in ["latitude", "lat", "y"]:
        if coord_name in ds.coords:
            lat_coord = coord_name
            break

    for coord_name in ["longitude", "lon", "x"]:
        if coord_name in ds.coords:
            lon_coord = coord_name
            break

    if not lat_coord or not lon_coord:
        print("Could not find lat/lon coordinates, skipping refinement")
        return preprocessed_data, current_params

    # Get original coordinate arrays
    lat_vals = ds[lat_coord].values
    lon_vals = ds[lon_coord].values

    # Need at least 2 points to define a resolution
    if lat_vals.size < 2 or lon_vals.size < 2:
        print(
            f"Not enough points to refine grid "
            f"(lat: {lat_vals.size}, lon: {lon_vals.size}), skipping refinement"
        )
        return preprocessed_data, current_params

    # Build a 2x finer grid using linspace: this avoids sign issues and empty arrays.
    # If original has N points, there are (N-1) intervals.
    # Doubling resolution gives 2*(N-1) intervals -> 2*(N-1)+1 points.
    n_lat_new = (lat_vals.size - 1) * 2 + 1
    n_lon_new = (lon_vals.size - 1) * 2 + 1

    new_lat = np.linspace(float(lat_vals[0]), float(lat_vals[-1]), n_lat_new)
    new_lon = np.linspace(float(lon_vals[0]), float(lon_vals[-1]), n_lon_new)

    print(f"Original grid: {lat_vals.size} × {lon_vals.size} points")
    print(f"Refined grid: {len(new_lat)} × {len(new_lon)} points (2x finer)")

    # Interpolate to finer grid
    refined_ds = ds.interp(
        {lat_coord: new_lat, lon_coord: new_lon},
        method="linear",
    )

    # Save refined dataset
    output = File.new_remote(file_name="climate_refined.nc")

    buffer = io.BytesIO()
    # Write as h5netcdf as well to keep everything consistent
    refined_ds.to_netcdf(buffer, engine="h5netcdf")
    buffer.seek(0)

    async with output.open("wb") as f:
        payload = buffer.read()
        await f.write(payload)

    # Update parameters
    refined_ensemble_size = max(100, int(current_params.ensemble_size * 0.75))

    refined_params = SimulationParams(
        grid_resolution_km=current_params.grid_resolution_km * 0.5,  # 2x finer
        time_step_minutes=max(
            1, current_params.time_step_minutes // 2
        ),  # Smaller timestep for stability
        simulation_hours=current_params.simulation_hours,
        physics_model=current_params.physics_model,
        boundary_layer_scheme=current_params.boundary_layer_scheme,
        microphysics_scheme=current_params.microphysics_scheme,
        radiation_scheme=current_params.radiation_scheme,
        ensemble_size=refined_ensemble_size,
        perturbation_magnitude=current_params.perturbation_magnitude,
    )

    return output, refined_params


@flyte.trace
async def recommend_parameter_adjustments(
    metrics: ClimateMetrics,
    current_params: SimulationParams,
) -> SimulationParams:
    """Rule-based parameter adjustment based on simulation metrics"""
    # Copy ALL parameters from current_params to preserve refinement changes
    new_params = SimulationParams(
        grid_resolution_km=current_params.grid_resolution_km,
        time_step_minutes=current_params.time_step_minutes,
        simulation_hours=current_params.simulation_hours,
        physics_model=current_params.physics_model,
        boundary_layer_scheme=current_params.boundary_layer_scheme,
        microphysics_scheme=current_params.microphysics_scheme,
        radiation_scheme=current_params.radiation_scheme,
        ensemble_size=current_params.ensemble_size,
        perturbation_magnitude=current_params.perturbation_magnitude,
        convergence_threshold=current_params.convergence_threshold,
        max_iterations=current_params.max_iterations,
    )

    # Adjust timestep based on convergence
    if metrics.convergence_rate > 0.01:
        # Poor convergence: reduce timestep
        new_params.time_step_minutes = max(1, current_params.time_step_minutes - 1)
        print(
            f"Reducing timestep: {current_params.time_step_minutes} → {new_params.time_step_minutes} min"
        )
    elif metrics.convergence_rate < 0.0001:
        # Excellent convergence: can increase timestep for speed
        new_params.time_step_minutes = min(10, current_params.time_step_minutes + 1)
        print(
            f"Increasing timestep: {current_params.time_step_minutes} → {new_params.time_step_minutes} min"
        )

    # Refine resolution near extreme events
    if "hurricane_detected" in metrics.detected_phenomena:
        new_params.grid_resolution_km = max(
            0.5, current_params.grid_resolution_km * 0.8
        )
        print(
            f"Refining resolution near hurricane: {current_params.grid_resolution_km} → {new_params.grid_resolution_km} km"
        )

    return new_params


@cpu_env.task
async def analyze_gpu_results(
    metrics: ClimateMetrics, current_params: SimulationParams
) -> tuple[dict, dict, SimulationParams]:
    """Wrapper task to group all analytics for one GPU (reduces UI clutter)"""
    convergence = await analyze_simulation_convergence(
        metrics, current_params.convergence_threshold
    )
    events = await detect_extreme_events(metrics)
    params = await recommend_parameter_adjustments(metrics, current_params)

    return convergence, events, params
# {{/docs-fragment analytics}}


# {{docs-fragment main-workflow}}
@cpu_env.task(report=True)
async def adaptive_climate_modeling_workflow(
    region: str = "atlantic",
    date_range: list[str, str] = ["2024-09-01", "2024-09-10"],
    current_params: SimulationParams = SimulationParams(),
    enable_multi_gpu: bool = True,
    n_gpus: int = 5,
) -> SimulationSummary:
    """Orchestrates multi-source ingestion, GPU simulation, and adaptive refinement."""
    # {{/docs-fragment main-workflow}}

    # {{docs-fragment workflow-ingestion}}
    # Parallel data ingestion from three sources
    with flyte.group("data-ingestion"):
        satellite_task = ingest_satellite_data(region, date_range)
        reanalysis_task = ingest_reanalysis_data(region, date_range)
        station_task = ingest_station_data(region, date_range)

        satellite_data, reanalysis_data, station_data = await asyncio.gather(
            satellite_task,
            reanalysis_task,
            station_task,
        )
    # {{/docs-fragment workflow-ingestion}}

    preprocessed_data = await preprocess_atmospheric_data(
        satellite_data,
        reanalysis_data,
        station_data,
        target_resolution_km=current_params.grid_resolution_km,
    )

    all_results = []
    all_metrics = []
    all_events = []

    iteration = 0
    max_iterations = current_params.max_iterations
    refinement_count = 0  # Track number of refinements (limit to 1 for production)

    # Log static HTML template once at the start
    static_template = build_static_report_template(current_params, region)
    await flyte.report.log.aio(static_template, do_flush=True)

    # Track previous iteration's mean state for convergence calculation
    previous_mean_state: np.ndarray | None = None

    while iteration < max_iterations:
        with flyte.group(f"iteration-{iteration + 1}"):
            # GPU Simulation
            if enable_multi_gpu and n_gpus > 1:
                # Ensemble-parallel multi-GPU simulation
                sim_results, sim_metrics = await run_distributed_simulation_ensemble(
                    preprocessed_data, current_params, n_gpus
                )
            else:
                # Single GPU simulation (uses full ensemble on one GPU)
                result, metrics = await run_atmospheric_simulation(
                    preprocessed_data, current_params
                )
                sim_results = [result]
                sim_metrics = [metrics]

            # Compute convergence by comparing forecast between iterations
            # Load the first simulation result to get the mean ensemble forecast
            try:
                async with sim_results[0].open("rb") as f:
                    result_bytes = await f.read()
                current_ds = xr.open_dataset(io.BytesIO(result_bytes))

                # Extract mean temperature field as a representative variable
                if "temperature_mean" in current_ds.data_vars:
                    current_mean_state = current_ds["temperature_mean"].values
                else:
                    # Fallback to first available variable
                    first_var = list(current_ds.data_vars)[0]
                    current_mean_state = current_ds[first_var].values

                current_ds.close()
            except Exception as e:
                print(
                    f"WARNING: Could not load simulation results for convergence: {e}"
                )
                current_mean_state = None

            # Calculate convergence as relative change in forecast
            if current_mean_state is not None:
                if previous_mean_state is None:
                    # First iteration: no previous state to compare
                    convergence_rate = 1.0
                else:
                    # Handle different shapes due to adaptive refinement
                    if current_mean_state.shape != previous_mean_state.shape:
                        from scipy.ndimage import zoom

                        print(
                            f"Shape mismatch: current={current_mean_state.shape}, "
                            f"previous={previous_mean_state.shape}"
                        )
                        print("Interpolating to common coarse grid for comparison...")

                        # Use the coarser (smaller) grid as reference
                        if current_mean_state.size < previous_mean_state.size:
                            reference_shape = current_mean_state.shape
                            resized_current = current_mean_state

                            # Downsample previous to match current
                            zoom_factors = [
                                reference_shape[i] / previous_mean_state.shape[i]
                                for i in range(len(reference_shape))
                            ]
                            resized_previous = zoom(
                                previous_mean_state, zoom_factors, order=1
                            )
                        else:
                            reference_shape = previous_mean_state.shape
                            resized_previous = previous_mean_state

                            zoom_factors = [
                                reference_shape[i] / current_mean_state.shape[i]
                                for i in range(len(reference_shape))
                            ]
                            resized_current = zoom(
                                current_mean_state, zoom_factors, order=1
                            )

                        print(f"Resized to common shape: {reference_shape}")
                    else:
                        resized_current = current_mean_state
                        resized_previous = previous_mean_state

                    # Calculate normalized difference between iterations
                    # convergence_rate = ||current - previous|| / ||previous||
                    state_diff = np.abs(resized_current - resized_previous)
                    mean_diff = np.nanmean(state_diff)
                    mean_magnitude = np.nanmean(np.abs(resized_previous))

                    if mean_magnitude > 0:
                        convergence_rate = float(mean_diff / mean_magnitude)
                    else:
                        convergence_rate = 1.0

                # Update previous state for next iteration
                previous_mean_state = current_mean_state.copy()
            else:
                # Fallback to ensemble spread if we can't load results
                convergence_rate = 1.0
                print(f"[ITER {iteration+1}] Could not compute convergence, using 1.0")

            # Write this convergence_rate back into all GPU metrics for this iteration
            for m in sim_metrics:
                m.convergence_rate = convergence_rate

            all_results.extend(sim_results)
            all_metrics.extend(sim_metrics)

            # Real-time Analytics
            analytics_tasks = [
                analyze_gpu_results(metrics, current_params) for metrics in sim_metrics
            ]

            analytics_results = await asyncio.gather(*analytics_tasks)

            # Unpack results from wrapper tasks
            convergence_analyses = [result[0] for result in analytics_results]
            event_detections = [result[1] for result in analytics_results]
            param_recommendations = [result[2] for result in analytics_results]

            all_events.extend(event_detections)

            # Stream update to report
            metrics_json = get_metrics_json(
                all_metrics, all_events, current_params, iteration
            )

            update_script = f"""<script>updateMetrics({metrics_json});</script>"""
            await flyte.report.log.aio(update_script, do_flush=True)

            # Check if converged
            all_converged = all(
                analysis["converged"] for analysis in convergence_analyses
            )
            for i, analysis in enumerate(convergence_analyses):
                print(
                    f"GPU {i}: {'Converged' if analysis['converged'] else 'Not converged'} "
                    f"(rate: {analysis['convergence_rate']:.6f}, threshold: {current_params.convergence_threshold})"
                )

            if all_converged:
                print("ALL PARTITIONS CONVERGED - Stopping early")
                print(f"Completed {iteration + 1} iterations (max: {max_iterations})")
                break
            else:
                print(f"\nContinuing to iteration {iteration + 2}...")

            # Apply parameter adjustments (use first recommendation)
            if param_recommendations:
                current_params = param_recommendations[0]

            # Adaptive Refinement (if needed)
            has_extreme_events = any(
                len(events["hurricanes"]) > 0 or len(events["heatwaves"]) > 0
                for events in event_detections
            )

            if (
                has_extreme_events
                and current_params.grid_resolution_km > 0.5
                and refinement_count < 1
            ):
                print(f"Extreme events detected: will refine mesh for next iteration")

                # Refine mesh to 2x higher resolution
                refined_data, refined_params = await refine_mesh_for_extreme_events(
                    preprocessed_data, current_params
                )

                # Update state for next iteration
                preprocessed_data = refined_data
                current_params = refined_params
                refinement_count += 1

                print(
                    f"Mesh refined to {refined_params.grid_resolution_km}km resolution"
                )
            elif has_extreme_events and refinement_count >= 1:
                print(
                    "\n>>>Refinement limit reached (1/1) - continuing with current resolution"
                )
                print(
                    f"Current: {current_params.grid_resolution_km}km, {current_params.ensemble_size} members"
                )

            gc.collect()
            iteration += 1

    # Send final update to report
    completed_iterations = iteration

    metrics_json = get_metrics_json(
        all_metrics,
        all_events,
        current_params,
        completed_iterations - 1,  # Adjust for 0-indexing
    )
    final_update_script = f"""<script>updateMetrics({metrics_json});</script>"""
    await flyte.report.log.aio(final_update_script, do_flush=True)

    # Create summary statistics
    total_hurricanes = sum(len(e["hurricanes"]) for e in all_events)
    total_heatwaves = sum(len(e["heatwaves"]) for e in all_events)
    avg_convergence = sum(m.convergence_rate for m in all_metrics) / len(all_metrics)
    total_compute_time = sum(m.compute_time_seconds for m in all_metrics)

    # Return summary statistics with output files
    return SimulationSummary(
        total_iterations=completed_iterations,
        final_resolution_km=current_params.grid_resolution_km,
        avg_convergence_rate=avg_convergence,
        total_compute_time_seconds=total_compute_time,
        hurricanes_detected=total_hurricanes,
        heatwaves_detected=total_heatwaves,
        converged=avg_convergence < 0.001,
        region=region,
        output_files=all_results,  # NetCDF simulation outputs from all iterations
        date_range=date_range,
    )
# {{/docs-fragment main-workflow}}


# {{docs-fragment main}}
if __name__ == "__main__":
    flyte.init_from_config()
    run_multi_gpu = flyte.run(adaptive_climate_modeling_workflow)

    print(f"Run URL: {run_multi_gpu.url}")
# {{/docs-fragment main}}