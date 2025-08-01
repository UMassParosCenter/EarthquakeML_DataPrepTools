"""
This script processes raw infrasound waveform data from earthquake events and extracts power spectral density (PSD) 
features along with event metadata.

Main steps:
1. Loads previously exported earthquake waveform data from a .pkl file.
2. Applies resampling, DC blocking, and high-pass filtering to each waveform.
3. Segments each waveform into overlapping windows (default: 10s duration with 50% overlap).
4. Computes the Welch PSD for each window, keeping frequency content up to 10 Hz.
5. Attaches event metadata (e.g., magnitude, depth, coordinates, arrival time) to each processed event.
6. Skips events with insufficient samples or fewer than 11 windows.
7. Saves the resulting PSD windows and metadata to a new .pkl file for further analysis or model training.

This output is used in machine learning pipelines to classify or study earthquake-related infrasound signals.

Ethan Gelfand, 07/31/2025
"""

import pickle
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

## --- Functions for processing earthquake data --- ##
def dc_block(x, a=0.999):
    b = [1, -1]
    a_coeffs = [1, -a]
    return signal.filtfilt(b, a_coeffs, x)
def preprocess(x, fs):
    x = dc_block(x)

    low_cutoff = 0.1
    Wn = low_cutoff / (fs / 2)
    b, a = signal.butter(4, Wn, btype='high')
    return signal.filtfilt(b, a, x)
def welch_psd(x, fs):
    window_duration = 5
    nperseg = int(fs * window_duration)
    noverlap = int(nperseg * 0.75)
    nfft = int(2 ** np.ceil(np.log2(nperseg)))

    window = signal.windows.hann(nperseg)
    f, pxx = signal.welch(x, fs, window=window, noverlap=noverlap, nfft=nfft)

    keep = f <= 10
    return pxx[keep], f[keep]
def safe_resample(x, fs_in, fs_out):
    x = dc_block(x)

    fc = 0.9 * min(fs_in, fs_out) / 2
    b_lp, a_lp = signal.butter(4, fc / (fs_in / 2), btype='low')
    x = signal.filtfilt(b_lp, a_lp, x)

    y = signal.resample_poly(x, fs_out, fs_in)
    return y

## --- load pickle file --- ##
file_path = "Exported_Paros_Data/EarthQuakeEvents.pkl"

with open(file_path, 'rb') as f:
    data = pickle.load(f)

## --- start processing earthquake data --- ##

# parameters
fs_in = 20          # original sampling frequency (Hz)
fs_out = 100        # target sampling frequency (Hz)
delta_t = 10        # window length in seconds
overlap = 0.5       # 50% overlap
fs = fs_out
psdResults = {}
goodEventCounter = 1

eventNames = list(data.keys())

for eventName in tqdm(eventNames, colour="green"):
    try:
        eventStuct = data[eventName]
        waveform = eventStuct['waveform']['parost2_141929'][:, -1]
        waveform = np.array(waveform, dtype=float)
        metadata = eventStuct['metadata']

        waveform = safe_resample(waveform, fs_in, fs_out)

        window_size = int(fs * delta_t) 
        stride = int(window_size * (1 - overlap))

        if len(waveform) < window_size:
            tqdm.write(f'Skipping {eventName}: not enough samples ({len(waveform)})')
            continue

        waveform = preprocess(waveform, fs)

        num_windows = (len(waveform) - window_size) // stride + 1

        if num_windows < 11:
            tqdm.write(f'Skipping {eventName}: only {num_windows} windows (need at least 11)')
            continue

        eventPSD = {'metadata': metadata}
        for w in range(num_windows):
            idx_start = w * stride
            idx_end = idx_start + window_size
            segment = waveform[idx_start:idx_end]

            pxx, f = welch_psd(segment, fs)
            winName = f'window_{w+1:03d}'
            eventPSD[winName] = {'power': pxx, 'frequency': f}

        keyName = f'event_{goodEventCounter:03d}'
        psdResults[keyName] = eventPSD
        goodEventCounter += 1

    except Exception as e:
        tqdm.write(f'Error processing {eventName}: {e}')

# Save as pickle
output_path = os.path.join(os.getcwd(), "Exported_Paros_Data/PSD_Windows_Earthquake_100Hz.pkl")
with open(output_path, 'wb') as f:
    pickle.dump(psdResults, f)

print(f'Saved PSDs with metadata to {output_path}')