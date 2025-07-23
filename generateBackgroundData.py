"""
This script samples background (non-earthquake) waveform data from an infrasound sensor 
for use in machine learning or statistical analysis.

Main workflow:
--------------
1. Load a CSV of earthquake event times.
2. Define a continuous range of hourly timestamps between the earliest and latest event.
3. Exclude all hours that fall within ±1 hour of any earthquake (to avoid contamination).
4. Randomly sample 1000 clean "background" hours from the remaining set.
5. For each sampled hour, query infrasound waveform data from an InfluxDB for a short 
   window (15 seconds before to 45 seconds after the timestamp).
6. Store the waveform data and associated timestamp in a dictionary.
7. Export all background data to a `.mat` file for MATLAB or Python use.

This process ensures that the background data is temporally decorrelated from seismic activity, 
making it suitable for use as negative class examples in event classification tasks.

for more information about paros_data_grabber, see https://pypi.org/project/paros-data-grabber/

Ethan Gelfand, 07/23/2025
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.io import savemat
from pathlib import Path
from paros_data_grabber import query_influx_data

def generate_background_hours(start_time, end_time, earthquake_datetimes, buffer_hours=1):
    """Generate hourly timestamps excluding ±buffer_hours around earthquake times."""
    all_hours = pd.date_range(start=start_time, end=end_time, freq='H')
    excluded = pd.DatetimeIndex([])

    for dt in earthquake_datetimes:
        buffer_range = pd.date_range(dt - pd.Timedelta(hours=buffer_hours),
                                     dt + pd.Timedelta(hours=buffer_hours),
                                     freq='H')
        excluded = excluded.union(buffer_range)

    return all_hours.difference(excluded).unique()

def sample_background_hours(available_hours, num_samples, seed=42):
    """Randomly sample background hours."""
    num_samples = min(num_samples, len(available_hours))
    rng = np.random.default_rng(seed)
    return pd.DatetimeIndex(rng.choice(available_hours, size=num_samples, replace=False))

# --- Load earthquake data ---
earthquake_data = pd.read_csv(r"C:\Users\YourUserName\Documents\EarthquakeData\EarthQuakeData.csv") # Adjust path as needed
earthquake_datetimes = pd.to_datetime(earthquake_data['time'])


# --- Define range of interest ---
start_time = earthquake_datetimes.min().floor('h')
end_time = earthquake_datetimes.max().ceil('h')

# --- Get valid background hours ---
background_hours = generate_background_hours(start_time, end_time, earthquake_datetimes, buffer_hours=1)

# --- Sample N background hours ---
selected_hours = sample_background_hours(background_hours, num_samples=1000)

print(f"Selected {len(selected_hours)} background hours.")

# --- InfluxDB fetch config ---
box_id = "parost2"
sensor_id = "141929"
time_before = timedelta(seconds=15)
time_after = timedelta(seconds=45)

# --- Fetch and store ---
all_data = {}
event_counter = 1

for idx, timestamp in enumerate(selected_hours.sort_values()):
    try:
        start_t = (timestamp - time_before).strftime("%Y-%m-%dT%H:%M:%S")
        end_t = (timestamp + time_after).strftime("%Y-%m-%dT%H:%M:%S")

        if idx == 0:
            print(f"First iteration start_time: {start_t}")
            print(f"First iteration end_time:   {end_t}")

        data = query_influx_data(
            start_time=start_t,
            end_time=end_t,
            box_id=box_id,
            sensor_id=sensor_id,
            password="*****",  # Replace with actual password
        )

        if not data:
            print(f"No data returned for hour {idx+1} at {timestamp}")
            continue

        data_arrays = {key: df_.values for key, df_ in data.items()}

        all_data[f"background_{event_counter:04d}"] = {
            'waveform': data_arrays,
            'timestamp': timestamp.strftime("%Y-%m-%dT%H:%M:%S")
        }

        print(f"Saved background_{event_counter:04d}")
        event_counter += 1

    except Exception as e:
        print(f"Failed on hour {idx+1}: {e}")
        continue

# --- Save as .mat file ---
output_file = Path(r"C:\Users\YourUserName\Documents\EarthquakeData\background_data.mat") # Adjust path as needed
savemat(output_file, all_data)
print(f"\nAll background data saved to: {output_file.resolve()}")
