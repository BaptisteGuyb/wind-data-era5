import os
import subprocess

start_year = 1976
end_year = 1990
dt_hours = 1  # ou 3 ou 6

non_summer_months <- [1,2,3,4,5,9,10,11,12]

for year in range(start_year, end_year + 1):
    for month in non_summer_months:
        print(f"==> Running for {year}-{month:02d}")
        cmd = [
            "python", 
            "particle_transport_scale_cli_parallel_32.py", 
            str(year), 
            str(month), 
            str(dt_hours)
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Fail for {year}-{month:02d}")
