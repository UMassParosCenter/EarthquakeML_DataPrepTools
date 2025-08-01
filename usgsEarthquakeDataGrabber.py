"""
This script automates the extraction and export of infrasound waveform data around earthquake events.

Key components:
- InfrasoundUtils: Provides a method to calculate a fixed surface wave arrival delay based on geodesic distance 
  and typical Rayleigh wave velocity (3.4 km/s).
  
- EarthquakeCatalog: Loads and formats an earthquake event catalog from a CSV file, ensuring proper data types 
  and filtering invalid entries.

- EarthquakeDataExporter: Iterates through each event, calculates estimated infrasound arrival time at a given 
  station, queries the waveform data from InfluxDB, and stores the waveform along with event metadata.

The final dataset is saved as a `.pkl` file for downstream use, such as in training or evaluation of 
machine learning models for earthquake signal classification.

Note: make sure to enter password for query_influx_data

Ethan Gelfand, 07/31/2025
"""

from datetime import timedelta
from pathlib import Path
from geopy.distance import geodesic
import pandas as pd
import pickle
from paros_data_grabber import query_influx_data
from tqdm import tqdm


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
        self.output_file = self.output_path / "EarthQuakeEvents.pkl"
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

            data = query_influx_data(
                start_time=start_time,
                end_time=end_time,
                box_id=self.box_id,
                sensor_id=self.sensor_id,
                password=self.password
            )

            if not data:
                tqdm.write(f"[Warning] No data for event {idx+1} at {event_time}")
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
            self.counter += 1

        except Exception as e:
            tqdm.write(f"[Error] Event {idx+1} failed: {e}")

    def export(self):
        with open(self.output_file, 'wb') as f:
            pickle.dump(self.data_dict, f)
        print(f"[Done] Data saved to {self.output_file.resolve()}")


if __name__ == "__main__":
    catalog_path = "EarthQuakeData.csv"
    output_dir = "Exported_Paros_Data"

    station_lat, station_lon = 24.07396028832464, 121.1286975322632
    box_id = "parost2"
    sensor_id = "141929"
    password = "******" # Enter password

    catalog = EarthquakeCatalog(catalog_path)
    exporter = EarthquakeDataExporter(
        station_lat=station_lat,
        station_lon=station_lon,
        box_id=box_id,
        sensor_id=sensor_id,
        password=password,
        output_path=output_dir
    )

    # Clean event loop with tqdm
    for idx in tqdm(range(len(catalog.df)), desc="Processing Events", colour="green"):
        row = catalog.df.iloc[idx]
        exporter.process_event(idx, row)

    exporter.export()