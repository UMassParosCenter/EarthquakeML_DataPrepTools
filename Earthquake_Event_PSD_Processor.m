%{
Earthquake Event PSD Processor with Resampling and Preprocessing

This script processes a collection of earthquake waveform signals by:
  1. Loading raw waveform data from a MAT file.
  2. Upsampling each signal from 20 Hz to 100 Hz for improved temporal
     resolution and PSD stability.
  3. Segmenting each waveform into overlapping windows (default: 5s 
     duration with 75% overlap).
  4. Preprocessing each segment via:
      - DC removal
      - Wavelet denoising
      - High-pass filtering (>0.1 Hz)
  5. Computing the Power Spectral Density (PSD) for each segment using
     Welch's method, and trimming results to the 0–20 Hz band.
  6. Storing the PSDs and corresponding metadata into a structured output.

The resulting PSDs are saved in a MAT file for downstream use in 
classification models

Author: Ethan Gelfand
Date: 07/27/2025
%}
clear; clc;

% --- Load MAT file ---
path = "EarthQuakeEvents_SurfaceWaveDelay.mat";
data = load(path);
eventNames = fieldnames(data);

% --- Original and target sampling frequencies ---
fs_in = 20;    % original sampling frequency (Hz)
fs_out = 100;   % upsampled frequency (Hz)

% --- Global Parameters ---
delta_t = 10;  % window length in seconds
overlap = 0.5; % 50% overlap

% --- Output Struct ---
psdResults = struct();

% --- Process Each Event ---
goodEventCounter = 1; % Counter for accepted events only

for i = 1:length(eventNames)
    eventName = eventNames{i};
    try
        eventStruct = data.(eventName);
        waveform = eventStruct.waveform.parost2_141929;
        metadata = eventStruct.metadata;

        % Extract signal (handle cell or numeric)
        if iscell(waveform)
            signal = cell2mat(waveform(:, end));
        else
            signal = waveform(:, end);
        end

        % Upsample signal from original fs to target fs
        signal = safe_resample(signal, fs_in, fs_out);

        % Update sampling frequency and window params for upsampled signal
        fs = fs_out;
        window_size = delta_t * fs;          % samples per window
        stride = floor(window_size * (1 - overlap)); % stride for overlapping windows

        % Skip if signal length is too short for one window
        if length(signal) < window_size
            fprintf('Skipping %s: not enough samples (%d)\n', eventName, length(signal));
            continue; % Do not save or increment counter
        end

        % Preprocess (DC block, denoise, highpass filter)
        signal = preprocess(signal, fs);

        % Calculate number of windows based on stride and window size
        num_windows = floor((length(signal) - window_size) / stride) + 1;
        fprintf('Processing %s with %d windows\n', eventName, num_windows);

        % Skip events with fewer than 11 windows
        if num_windows < 11
            fprintf('Skipping %s: only %d windows (need at least 11)\n', eventName, num_windows);
            continue; % Do not save or increment counter
        end

        % Prepare event PSD struct with metadata
        eventPSD = struct();
        eventPSD.metadata = metadata;

        % Compute PSD for each window segment
        for w = 1:num_windows
            idx_start = (w - 1) * stride + 1;
            idx_end = idx_start + window_size - 1;
            segment = signal(idx_start:idx_end);
            [pxx, f] = WelchPSD(segment, fs);
            winName = sprintf("window_%03d", w);
            eventPSD.(winName).power = pxx;
            eventPSD.(winName).frequency = f;
        end

        % Store the event with sequential numbering for accepted events only
        keyName = sprintf('event_%03d', goodEventCounter);
        psdResults.(keyName) = eventPSD;
        goodEventCounter = goodEventCounter + 1;

    catch ME
        fprintf('Error processing %s: %s\n', eventName, ME.message);
    end
end

% --- Save Result ---
outputPath = fullfile(pwd, 'PSD_Windows_Earthquake_100Hz.mat');
save(outputPath, 'psdResults');
fprintf('Saved PSDs with metadata to %s\n', outputPath);


%% FUNCTIONS
function y = preprocess(x, fs)
    % --- DC Block ---
    x = DCBlock(x);

    % --- Wavelet Denoising ---
    x = safe_wdenoise(x, fs);

    % --- Highpass Filter (e.g., 0.1 Hz and above) ---
    low_cutoff = 0.1;  % Hz
    Wn = low_cutoff / (fs / 2);  % Normalize
    [b, a] = butter(4, Wn, 'high');
    y = filtfilt(b, a, x);
end
function [pxx, f] = WelchPSD(signal, fs)
    window_duration = 5;  % in seconds
    nperseg = round(fs * window_duration);  % adapt segment length to sampling rate
    noverlap = round(nperseg * 0.75); % 75% window overlap
    nfft = 2^nextpow2(nperseg); % 128 for 20Hz
    window = hann(nperseg);
    [pxx, f] = pwelch(signal, window, noverlap, nfft, fs);
    
    % Keep only 0–20 Hz part
    keep = f <= 20;
    pxx = pxx(keep);
    f = f(keep);
end
function y = safe_resample(x, fs_in, fs_out)
% SAFE_RESAMPLE Resamples a signal from fs_in to fs_out with anti-imaging protection.
%
%   y = safe_resample(x, fs_in, fs_out)
%
%   Inputs:
%       x       - Input signal (column or row vector)
%       fs_in   - Original sampling frequency (Hz)
%       fs_out  - Desired sampling frequency (Hz)
%
%   Output:
%       y       - Resampled signal at fs_out

    % --- Step 1: DC block to remove drift ---
    x = DCBlock(x);
    
    % --- Step 2: Pre-filter to remove high frequency content before upsampling ---
    nyquist_in = fs_in / 2;
    fc = 0.9 * min(fs_in, fs_out) / 2;  % Slightly below Nyquist
    [b_lp, a_lp] = butter(4, fc / nyquist_in, 'low');
    x = filtfilt(b_lp, a_lp, x);
    
    % --- Step 3: Resample with built-in anti-aliasing FIR filter ---
    y = resample(x, fs_out, fs_in);  % handles upsampling and filtering internally
end
function y = safe_wdenoise(x, fs)
% SAFE_WDENOISE Applies safe wavelet denoising without introducing harmonics
%
%   y = safe_wdenoise(x, fs)
%
%   Inputs:
%       x  - input signal
%       fs - sampling frequency (Hz)
%
%   Output:
%       y  - denoised signal

    % --- Use a suitable wavelet ---
    waveletType = 'sym8';  % smoother than 'db' family, less likely to introduce artifacts
    level = wmaxlev(length(x), waveletType);  % maximum safe level
    level = min(level, 6); % cap level (can tune down to reduce distortion)

    % --- Apply denoising ---
    y = wdenoise(x, level, ...
        'Wavelet', waveletType, ...
        'DenoisingMethod', 'Bayes', ...
        'ThresholdRule', 'Soft', ...
        'NoiseEstimate', 'LevelDependent');

    % --- Lowpass to clean residual ringing ---
    fc = 0.9 * (fs/2);  % just below Nyquist
    [b, a] = butter(4, fc/(fs/2), 'low');
    y = filtfilt(b, a, y);
end
function y = DCBlock(x)
    a = 0.999;
    b = [1 -1];
    a_coeffs = [1 -a];
    y = filtfilt(b, a_coeffs, x);
end
