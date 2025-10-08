import argparse
import time
import cdsapi
import xarray as xr
import numpy as np
from pyproj import Geod
import pandas as pd
from datetime import datetime, timedelta
import geopandas as gpd
import rioxarray
from shapely.geometry import Point
import os
import requests
import zipfile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import calendar
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# FUNCTIONS 
def download_month_plus_buffer(c, year, month, area, pressures, times, dt_hours):
    
    # get number of days 
    num_days = calendar.monthrange(year, month)[1]

    # if hourly processing 
    if dt_hours == 1:
        ranges = [(1, 10), (11, 20), (21, num_days)]
        datasets = []

        for start_day, end_day in ranges:
            days = [f'{d:02d}' for d in range(start_day, end_day + 1)]
            fname_part = f'era5_{year}_{month:02d}_{start_day:02d}-{end_day:02d}.nc'

            c.retrieve(
                'reanalysis-era5-pressure-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': ['u_component_of_wind', 'v_component_of_wind', 'vertical_velocity'],
                    'pressure_level': pressures,
                    'year': str(year),
                    'month': f'{month:02d}',
                    'day': days,
                    'time': times,
                    'area': area,
                    'format': 'netcdf',
                },
                fname_part
            )
            datasets.append(xr.open_dataset(fname_part))

        # also retrieve spillover days
        if month == 12:
            next_month = 1
            next_year = year + 1
        else:
            next_month = month + 1
            next_year = year

        fname_next = f'era5_{next_year}_{next_month:02d}_spill.nc'
        fname_combined = f'era5_{year}_{month:02d}_{dt_hours:02d}timestep_full.nc'
        spillover_days = ['01', '02']

        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'variable': ['u_component_of_wind', 'v_component_of_wind', 'vertical_velocity'],
                'pressure_level': pressures,
                'year': str(next_year),
                'month': f'{next_month:02d}',
                'day': spillover_days,
                'time': times,
                'area': area,
                'format': 'netcdf',
            },
            fname_next
        )

        datasets.append(xr.open_dataset(fname_next))

        # combine and save
        combined = xr.concat(datasets, dim='time')
        combined.to_netcdf(fname_combined, mode="w")
        print(f"Saved merged file to {fname_combined}")
        
        # remove temporary files
        for start_day, end_day in ranges:
            fname_part = f'era5_{year}_{month:02d}_{start_day:02d}-{end_day:02d}.nc'
            os.remove(fname_part)
        os.remove(fname_next)
        print(f"Deleted intermediate files.")

    else:
        base_days = [f'{d:02d}' for d in range(1, num_days + 1)]

        if month == 12:
            next_month = 1
            next_year = year + 1
        else:
            next_month = month + 1
            next_year = year

        spillover_days = ['01', '02']

        fname_main = f'era5_{year}_{month:02d}.nc'
        fname_next = f'era5_{next_year}_{next_month:02d}_spill.nc'
        fname_combined = f'era5_{year}_{month:02d}_{dt_hours:02d}timestep_full.nc'

        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'variable': ['u_component_of_wind', 'v_component_of_wind', 'vertical_velocity'],
                'pressure_level': pressures,
                'year': str(year),
                'month': f'{month:02d}',
                'day': base_days,
                'time': times,
                'area': area,
                'format': 'netcdf',
            },
            fname_main
        )

        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'variable': ['u_component_of_wind', 'v_component_of_wind', 'vertical_velocity'],
                'pressure_level': pressures,
                'year': str(next_year),
                'month': f'{next_month:02d}',
                'day': spillover_days,
                'time': times,
                'area': area,
                'format': 'netcdf',
            },
            fname_next
        )

        # combine and save
        ds1 = xr.open_dataset(fname_main)
        ds2 = xr.open_dataset(fname_next)
        combined = xr.concat([ds1, ds2], dim='time')
        combined.to_netcdf(fname_combined, mode="w")
        print(f"Saved merged file to {fname_combined}")
        
        # remove temporary files
        os.remove(fname_next)
        os.remove(fname_main)
        print(f"Deleted intermediate files.")

def simulate_trajectory_daily(args, valid_times, ds, available_levels, geod, n_steps, dt_hours):
    pid, lat, lon, start_time = args

    try:
        start_idx = np.where(valid_times == start_time)[0][0]
        time_window = valid_times[start_idx:start_idx + n_steps]
    except IndexError:
        return None  # Not enough data left

    pressure = 1000
    positions = [(start_time, lat, lon, pressure)]

    for t in time_window:
        next_time = t + timedelta(hours=dt_hours)

        # interpolate wind components
        u = ds['u'].sel(valid_time=t, pressure_level=pressure).interp(latitude=lat, longitude=lon).values
        v = ds['v'].sel(valid_time=t, pressure_level=pressure).interp(latitude=lat, longitude=lon).values
        w = ds['w'].sel(valid_time=t, pressure_level=pressure).interp(latitude=lat, longitude=lon).values

        # horizontal motion
        speed = np.sqrt(u**2 + v**2)
        direction = np.degrees(np.arctan2(u, v)) % 360
        distance = speed * 3600 * dt_hours
        lon, lat, _ = geod.fwd(lon, lat, direction, distance)

        # vertical motion
        delta_p = w * 3600 * dt_hours
        new_pressure = pressure + delta_p
        pressure = available_levels[np.argmin(np.abs(available_levels - new_pressure))]

        positions.append((next_time, lat, lon, pressure))

    return {
        "particle_id": pid,
        "start_time": start_time,
        "trajectory": positions
    }

def simulate_trajectory_monthly(args, u_avg, v_avg, w_avg, available_levels, geod, n_steps, dt_hours):
    pid, lat, lon = args
    pressure = 1000  # hPa
    positions = [(0, lat, lon, pressure)]  # step 0

    for step in range(1, n_steps + 1):
        # interpolate wind components
        u = u_avg.sel(pressure_level=pressure).interp(latitude=lat, longitude=lon).item()
        v = v_avg.sel(pressure_level=pressure).interp(latitude=lat, longitude=lon).item()
        w = w_avg.sel(pressure_level=pressure).interp(latitude=lat, longitude=lon).item()

        # horizontal motion
        speed = np.sqrt(u**2 + v**2)
        direction = np.degrees(np.arctan2(u, v)) % 360
        distance = speed * 3600 * dt_hours
        lon, lat, _ = geod.fwd(lon, lat, direction, distance)

        # vertical motion
        delta_p = w * 3600 * dt_hours
        new_pressure = pressure + delta_p
        pressure = available_levels[np.argmin(np.abs(available_levels - new_pressure))]

        positions.append((step * dt_hours, lat, lon, pressure))

    return {
        "particle_id": pid,
        "trajectory": positions,
        "start_lat": positions[0][1],
        "start_lon": positions[0][2]
    }

def main(year_to_run, month_to_run, dt_hours_input):

    # make variables global 
    global ds, valid_times, dt_hours, available_levels, geod, n_steps, u_avg, v_avg, w_avg
    
    total_start = time.time()
    print(f"Running particle transport for {year_to_run}-{month_to_run:02d}")
    
    # API client 
    c = cdsapi.Client()
    
    # parameters for downloads
    year_to_run = int(year_to_run) 
    month_to_run = int(month_to_run)
    dt_hours = int(dt_hours_input)
    
    # determine times based on dt_hours
    if dt_hours == 1:
        times = [f"{h:02d}:00" for h in range(24)]
    elif dt_hours == 3:
        times = ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']
    elif dt_hours == 6:
        times = ['00:00', '06:00', '12:00', '18:00']
    else:
        raise ValueError("Invalid dt_hours. Must be one of: 1, 3, 6.")

    # set length and n_steps
    length = 24
    n_steps =  int(length / dt_hours)
    
    # pressure levels
    pressures = ['1000', '975', '950', '925', '900', '850', '800', '700', '600', '500', '300', '100']
    
    # area 
    area = [30, -110, 55, -90]
    
    # intersecting points
    overlap_points = pd.read_csv("selected_era5_points.csv")

    # DOWNLOAD 
    download_start = time.time()
    download_month_plus_buffer(c, year=year_to_run, month=month_to_run, area=area, pressures=pressures, times=times, dt_hours=dt_hours)
    download_end = time.time()
    
    # file name based on year/month/hours
    ds = xr.open_dataset(f'era5_{year_to_run}_{month_to_run:02d}_{dt_hours:02d}timestep_full.nc')
    
    # fix time array issue 
    expver_array = ds['expver'].values 
    good_time_idx = np.where(expver_array[:, 0] == '0001')[0][0]
    
    # now select only that slice along time
    ds = ds.isel(time=good_time_idx)
    
    # constants and levels 
    geod = Geod(ellps="WGS84")
    available_levels = ds.pressure_level.values
    valid_times = pd.to_datetime(ds.valid_time.values)

    # DAILY 
    daily_start = time.time()
    
    # get start times 
    start_times = [t for t in valid_times if t.hour == 12]
    if len(start_times) >= 2:
        start_times = start_times[:-2]
    
    # prepare task list: all (particle_id, lat, lon, start_time) combinations
    task_list = []
    for start_time in start_times:
        for pid, row in enumerate(overlap_points.itertuples()):
            task_list.append((pid, row.latitude, row.longitude, start_time))

    # run in parallel
    print("Starting daily...")
    simulate_daily_partial = partial(
        simulate_trajectory_daily,
        valid_times=valid_times,
        ds=ds,
        available_levels=available_levels,
        geod=geod,
        n_steps=n_steps,
        dt_hours=dt_hours
    )
    with ProcessPoolExecutor(max_workers=32) as executor:  
        results = list(executor.map(simulate_daily_partial, task_list))
    
    # filter out any failed simulations (None values)
    all_trajectories = [res for res in results if res is not None]
    
    # save dataframe of results 
    records = []
    for traj in all_trajectories:
        for t, lat, lon, p in traj["trajectory"]:
            records.append({
                "particle_id": traj["particle_id"],
                "start_time": traj["start_time"],
                "step_time": t,
                "latitude": lat,
                "longitude": lon,
                "pressure": p
            })
    
    df = pd.DataFrame(records)
    df.to_csv(f"output/daily/trajectories_daily_{year_to_run}_{month_to_run}_{dt_hours}timestep.csv")
    daily_end = time.time()

    # MONTHLY 
    monthly_start = time.time()
    
    # keep only current month 
    first_time = ds['valid_time'].values[0]
    current_month = np.datetime64(first_time, 'M')  
    next_month = current_month + np.timedelta64(1, 'M')  
    ds = ds.sel(valid_time=slice(current_month, next_month - np.timedelta64(1, 'ns')))
    
    # average wind over all time steps
    print("Averaging wind components over all time steps...")
    u_avg = ds['u'].mean(dim='valid_time')
    v_avg = ds['v'].mean(dim='valid_time')
    w_avg = ds['w'].mean(dim='valid_time')
    
    # prepare arguments
    args_list = [(pid, row.latitude, row.longitude) for pid, row in enumerate(overlap_points.itertuples())]
    
    # run in parallel
    print("Starting monthly...")
    simulate_monthly_partial = partial(
        simulate_trajectory_monthly,
        u_avg=u_avg,
        v_avg=v_avg,
        w_avg=w_avg,
        available_levels=available_levels,
        geod=geod,
        n_steps=n_steps,
        dt_hours=dt_hours
    )
    with ProcessPoolExecutor(max_workers=32) as executor:
        all_trajectories = list(executor.map(simulate_monthly_partial, args_list))
    
    records = []
    for traj in all_trajectories:
        pid = traj["particle_id"]
        start_lat = traj["start_lat"]
        start_lon = traj["start_lon"]
    
        for elapsed_hours, lat, lon, pressure in traj["trajectory"]:
            records.append({
                "particle_id": pid,
                "elapsed_hours": elapsed_hours,
                "latitude": lat,
                "longitude": lon,
                "pressure": pressure
            })
    
    df = pd.DataFrame(records)
    df.to_csv(f"output/monthly/trajectories_monthly_{year_to_run}_{month_to_run}_{dt_hours}timestep.csv")

    monthly_end = time.time()
        
    # total end 
    total_end = time.time()
    
    # times 
    print(f"Download took {(download_end - download_start)/60:.2f} minutes.")    
    print(f"Daily projection took {(daily_end - daily_start)/60:.2f} minutes.")
    print(f"Monthly projection took {(monthly_end - monthly_start)/60:.2f} minutes.")
    print(f"Total script execution took {(total_end - total_start)/60:.2f} minutes.")
    
    # delete file 
    fname_delete = f'era5_{year_to_run}_{month_to_run:02d}_{dt_hours:02d}timestep_full.nc'
    os.remove(fname_delete)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run particle transport model for given year, month, and resolution-hours.")
    parser.add_argument("year", type=int, help="Year to run")
    parser.add_argument("month", type=int, help="Month to run")
    parser.add_argument("dt_hours", type=int, choices=[1, 3, 6], help="Timestep in hours (valid: 1, 3, 6)")
    args = parser.parse_args()
    main(args.year, args.month, args.dt_hours)
