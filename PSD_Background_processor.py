"""
This script processes raw background infrasound waveform data and extracts power spectral density (PSD) windows 
for use in machine learning and signal analysis.

Main steps:
1. Loads waveform data from a .pkl file containing background events.
2. Applies resampling, DC blocking, and high-pass filtering to each waveform.
3. Segments waveforms into overlapping windows (default: 10s duration with 50% overlap).
4. Computes the Welch PSD for each window, keeping frequencies up to 10 Hz.
5. Skips events with insufficient data or too few valid windows (less than 11).
6. Saves all processed PSD windows into a new .pkl file.

This output is used as model input for classification tasks such as distinguishing earthquakes from background noise.

Ethan Gelfand, 07/31/2025
"""

import pickle
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

## --- Functions for processing waveform data --- ##
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

## --- Load pickle file --- ##
file_path = "Exported_Paros_Data/background_data.pkl"

with open(file_path, 'rb') as f:
    data = pickle.load(f)

## --- Start processing background data --- ##
fs_in = 20          # Original sampling frequency (Hz)
fs_out = 100        # Target sampling frequency (Hz)
delta_t = 10        # Window length in seconds
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

        # Resample the waveform to the target frequency
        waveform = safe_resample(waveform, fs_in, fs_out)

        # Split the waveform into windows
        window_size = int(fs * delta_t) 
        stride = int(window_size * (1 - overlap))

        if len(waveform) < window_size:
            tqdm.write(f'Skipping {eventName}: not enough samples ({len(waveform)})')
            continue

        # Preprocess
        waveform = preprocess(waveform, fs)

        num_windows = (len(waveform) - window_size) // stride + 1

        if num_windows < 11:
            tqdm.write(f'Skipping {eventName}: only {num_windows} windows (need at least 11)')
            continue

        eventPSD = {}
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
output_path = os.path.join(os.getcwd(), "Exported_Paros_Data/PSD_Windows_Background_100Hz.pkl")
with open(output_path, 'wb') as f:
    pickle.dump(psdResults, f)

print(f'Saved PSDs to {output_path}')
