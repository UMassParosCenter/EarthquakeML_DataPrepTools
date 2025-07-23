"""
This script automates the extraction and export of infrasound waveform data around earthquake events.

Key components:
- InfrasoundUtils: Provides a method to calculate a fixed surface wave arrival delay based on geodesic distance 
  and typical Rayleigh wave velocity (3.4 km/s).
  
- EarthquakeCatalog: Loads and formates an earthquake event catalog from a CSV file, ensuring proper data types 
  and filtering invalid entries.

- EarthquakeDataExporter: For each earthquake event, it calculates the expected surface wave arrival time at a 
  seismic station using the delay. It then queries waveform data from an InfluxDB (via `query_influx_data`, **don't forget to insert the password!**) 
  for a time window around this arrival time (default: 15 sec before, 45 sec after). The waveform data and 
  event metadata are stored in a dictionary.

- The collected data for all events are saved into a MATLAB .mat file for further analysis and preprocessing.

The script's main section configures file paths, station coordinates, and authentication details, 
processes all catalog events, and exports the data.

for more information about paros_data_grabber, see https://pypi.org/project/paros-data-grabber/

Ethan Gelfand, 07/23/2025
"""

from datetime import timedelta
from pathlib import Path
from geopy.distance import geodesic
import pandas as pd
from scipy.io import savemat
from paros_data_grabber import query_influx_data


class InfrasoundUtils:
    @staticmethod
    def surface_wave_delay(event_lat, event_lon, station_lat, station_lon):
        dist_km = geodesic((event_lat, event_lon), (station_lat, station_lon)).km
        vsurface = 3.4  # km/s typical Rayleigh wave group velocity 
        return dist_km / vsurface


class EarthquakeCatalog:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self._clean()

    def _clean(self):
        self.df.columns = self.df.columns.str.strip().str.lower()
        self.df['time'] = pd.to_datetime(self.df['time'], errors='coerce')
        self.df['latitude'] = pd.to_numeric(self.df['latitude'], errors='coerce')
        self.df['longitude'] = pd.to_numeric(self.df['longitude'], errors='coerce')
        self.df['depth'] = pd.to_numeric(self.df['depth'], errors='coerce')
        self.df['mag'] = pd.to_numeric(self.df['mag'], errors='coerce')
        self.df['magtype'] = self.df['magtype'].str.strip().str.lower()
        self.df.dropna(subset=['time'], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    def get_events(self):
        return self.df.iterrows()


class EarthquakeDataExporter:
    def __init__(self, station_lat, station_lon, box_id, sensor_id, password, output_path,
                 time_before=timedelta(seconds=15), time_after=timedelta(seconds=45)):
        self.station_lat = station_lat
        self.station_lon = station_lon
        self.box_id = box_id
        self.sensor_id = sensor_id
        self.password = password
        self.time_before = time_before
        self.time_after = time_after
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_path / "EarthQuakeEvents_SurfaceWaveDelay.mat"
        self.data_dict = {}
        self.counter = 1

    def process_event(self, idx, row):
        try:
            event_time = row['time']
            event_lat = row['latitude']
            event_lon = row['longitude']

            # Calculate fixed surface wave delay
            delay = InfrasoundUtils.surface_wave_delay(event_lat, event_lon, self.station_lat, self.station_lon)
            arrival_time = event_time + timedelta(seconds=delay)

            start_time = (arrival_time - self.time_before).strftime("%Y-%m-%dT%H:%M:%S")
            end_time = (arrival_time + self.time_after).strftime("%Y-%m-%dT%H:%M:%S")

            if idx == 0:
                print(f"[First Event] Waveform window: {start_time} to {end_time}")

            data = query_influx_data(
                start_time=start_time,
                end_time=end_time,
                box_id=self.box_id,
                sensor_id=self.sensor_id,
                password=self.password
            )

            if not data:
                print(f"[Warning] No data for event {idx+1} at {event_time}")
                return

            data_arrays = {key: df_.values for key, df_ in data.items()}
            metadata = {
                'time': event_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                'latitude': event_lat,
                'longitude': event_lon,
                'depth': row['depth'],
                'magnitude': row['mag'],
                'magtype': row['magtype'],
                'arrival_time': arrival_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            }

            key = f"event_{self.counter:03d}"
            self.data_dict[key] = {
                'waveform': data_arrays,
                'metadata': metadata
            }

            print(f"[Saved] {key} - arrival at {arrival_time}")
            self.counter += 1

        except Exception as e:
            print(f"[Error] Event {idx+1} failed: {e}")

    def export(self):
        savemat(self.output_file, self.data_dict)
        print(f"[Done] Data saved to {self.output_file.resolve()}")


if __name__ == "__main__":
    catalog_path = r"C:\Users\YourUserName\Documents\EarthquakeData\EarthQuakeData.csv" # Adjust path as needed
    output_dir = r"C:\Users\YourUserName\Documents\EarthquakeExports" # Adjust path as needed

    station_lat, station_lon = 24.07396028832464, 121.1286975322632
    box_id = "parost2"
    sensor_id = "141929"
    password = "*******" # Insert the password here

    catalog = EarthquakeCatalog(catalog_path)
    exporter = EarthquakeDataExporter(
        station_lat=station_lat,
        station_lon=station_lon,
        box_id=box_id,
        sensor_id=sensor_id,
        password=password,
        output_path=output_dir
    )

    for idx, row in catalog.df.iterrows():
        exporter.process_event(idx, row)

    exporter.export()
