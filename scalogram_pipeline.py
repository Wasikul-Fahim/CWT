# -*- coding: utf-8 -*-
"""
scalogram_pipeline.py

Automatically generate CWT scalogram images (X, Y, Z) for all files in SAxx folders.

- Searches for folders named SAxx in BASE_DIR
- For each .txt file inside SAxx, reads the first 3 columns (handles commas/spaces/tabs)
- Generates CWT scalograms using PyWavelets (Morlet)
- Saves images to: BASE_DIR / "Scalogram images" / SAxx / <file_basename>_X.png, etc.

Notes:
- Adjust FS (sampling rate) to match your dataset.
- Adjust scales and wavelet if you want different time-frequency resolution.
"""

import os
import re
import sys
import math
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import pywt
from PIL import Image

# ---------------- USER SETTINGS ----------------
# Base directory where SA01, SA02, ... folders reside.
# Default: current working directory. Change to your project folder if needed.
BASE_DIR = Path.cwd()  # or Path(r"D:\3.2\Thesis\SisFall Model")

# Name of output root folder (inside BASE_DIR)
OUTPUT_ROOT = BASE_DIR / "Scalogram images"

# Pattern to find subject folders: SA followed by digits (case-insensitive)
SUBJECT_FOLDER_PATTERN = re.compile(r'^SA\d+', re.IGNORECASE)

# File matching pattern (optional check). We'll process all .txt files by default.
TXT_SUFFIX = '.txt'

# Sampling frequency (Hz) — change to your actual sensor sampling rate
FS = 100.0

# Wavelet and scales for CWT
WAVELET = 'morl'            # Morlet wavelet (robust default)
SCALES = np.arange(1, 128)  # adjust upper bound for frequency resolution

# Output image options
DPI = 150
RESIZE_TO = None  # e.g., (224,224) to resize for ML; set to None to keep original

# How to treat noisy/missing columns
MIN_COLS_REQUIRED = 3

# ------------------------------------------------

def find_subject_folders(base_dir: Path):
    """Return list of subject folder Paths in base_dir matching SUBJECT_FOLDER_PATTERN."""
    folders = []
    for p in base_dir.iterdir():
        if p.is_dir() and SUBJECT_FOLDER_PATTERN.match(p.name):
            folders.append(p)
    return sorted(folders)

def robust_read_txt(file_path: Path):
    """
    Read a .txt/CSV-like file using pandas with flexible separator.
    Returns a DataFrame or raises Exception.
    """
    # Try several read attempts for robustness
    # 1) let pandas infer with regex sep (handles spaces, tabs, commas)
    try:
        df = pd.read_csv(file_path, header=None, sep=r'[,\s]+', engine='python', comment='#')
    except Exception:
        # 2) try reading with delimiter=',' then fallback to whitespace
        try:
            df = pd.read_csv(file_path, header=None, delimiter=',', engine='python', comment='#')
        except Exception:
            try:
                df = pd.read_csv(file_path, header=None, delim_whitespace=True, engine='python', comment='#')
            except Exception as ex:
                raise RuntimeError(f"Failed to read {file_path}: {ex}")

    # drop fully-empty columns (e.g., trailing comma produced an empty column)
    df = df.dropna(axis=1, how='all')

    return df

def preprocess_signal(arr):
    """
    - Convert to numpy float array
    - Drop NaN
    - Remove DC offset
    - Normalize to max abs 1 (if nonzero)
    """
    arr = np.asarray(arr, dtype=float)
    # coerce inf or extremely large values
    mask = np.isfinite(arr)
    arr = arr[mask]
    if arr.size == 0:
        return arr
    arr = arr - np.mean(arr)
    max_abs = np.max(np.abs(arr))
    if max_abs == 0 or np.isnan(max_abs) or not np.isfinite(max_abs):
        return arr
    return arr / max_abs

def generate_cwt_scalogram(signal, fs=FS, scales=SCALES, wavelet=WAVELET):
    """
    Returns magnitude (2D np.array) from CWT.
    shape = (len(scales), len(signal))
    """
    coeffs, freqs = pywt.cwt(signal, scales, wavelet, sampling_period=1.0/fs)
    magnitude = np.abs(coeffs)
    return magnitude

def save_scalogram_image(magnitude, time_len, scales, out_path: Path, dpi=DPI, resize_to=RESIZE_TO):
    """
    Save magnitude as an image (imshow). magnitude shape: (n_scales, n_times)
    out_path: full path (including filename) where image will be saved
    """
    plt.figure(figsize=(8, 4.5))
    # extent: [t0, t1, scale_max, scale_min] so y-axis (scale) is top->bottom when inverted
    extent = [0, time_len, max(scales), min(scales)]
    plt.imshow(magnitude, aspect='auto', extent=extent)
    plt.gca().invert_yaxis()
    plt.xlabel("Time (samples)")   # time in samples; user can interpret using FS
    plt.ylabel("Scale")
    plt.colorbar(label='Magnitude')
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()

    # Optionally resize to fixed shape (for ML)
    if resize_to is not None:
        try:
            img = Image.open(out_path)
            img = img.resize(resize_to, resample=Image.BILINEAR)
            img.save(out_path)
        except Exception as e:
            print(f"Warning: failed to resize {out_path}: {e}")

def process_file(file_path: Path, output_folder: Path, fs=FS):
    """
    Process a single .txt file: create scalograms for first 3 columns and save images.
    """
    base_name = file_path.stem  # e.g., F01_SA01_R01
    try:
        df = robust_read_txt(file_path)
    except Exception as e:
        print(f"ERROR reading {file_path}: {e}")
        return

    # Drop columns that are entirely NaN
    df = df.dropna(axis=1, how='all')

    if df.shape[1] < MIN_COLS_REQUIRED:
        print(f"SKIP {file_path.name}: less than {MIN_COLS_REQUIRED} columns after cleaning (found {df.shape[1]})")
        return

    # Ensure output subfolder exists
    output_folder.mkdir(parents=True, exist_ok=True)

    axis_labels = ['X', 'Y', 'Z']
    for i, axis in enumerate(axis_labels):
        if i >= df.shape[1]:
            print(f"SKIP {base_name}_{axis}: column {i} missing")
            continue

        # Convert to numeric, coerce errors to NaN, then drop NaN
        col = pd.to_numeric(df.iloc[:, i], errors='coerce').values
        sig = preprocess_signal(col)

        if sig.size < 4:
            print(f"SKIP {base_name}_{axis}: insufficient data after preprocessing (len={sig.size})")
            continue

        try:
            magnitude = generate_cwt_scalogram(sig, fs=fs, scales=SCALES, wavelet=WAVELET)
        except Exception as e:
            print(f"ERROR CWT {base_name}_{axis}: {e}")
            traceback.print_exc()
            continue

        # Build filename and save
        out_name = f"{base_name}_{axis}.png"
        out_path = output_folder / out_name
        try:
            save_scalogram_image(magnitude, time_len=sig.size, scales=SCALES, out_path=out_path)
            print(f"Saved: {out_path}")
        except Exception as e:
            print(f"ERROR saving image {out_path}: {e}")
            traceback.print_exc()

def process_all_subjects(base_dir: Path = BASE_DIR):
    # Ensure output root exists
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    subjects = find_subject_folders(base_dir)
    if not subjects:
        print(f"No subject folders (SAxx) found in {base_dir}.")
        return

    print(f"Found {len(subjects)} subject folders:")
    for s in subjects:
        print(" -", s.name)

    for subj in subjects:
        print(f"\nProcessing subject: {subj.name}")
        # output subfolder for this subject
        out_sub = OUTPUT_ROOT / subj.name
        out_sub.mkdir(parents=True, exist_ok=True)

        # list .txt files
        txt_files = sorted([f for f in subj.iterdir() if f.is_file() and f.suffix.lower() == TXT_SUFFIX])
        if not txt_files:
            print(f"  No .txt files in {subj}")
            continue

        print(f"  Found {len(txt_files)} .txt files in {subj.name}")
        for txt in txt_files:
            try:
                process_file(txt, out_sub, fs=FS)
            except Exception as e:
                print(f"  Unexpected error processing {txt}: {e}")
                traceback.print_exc()

    print("\nAll done. Check output at:", OUTPUT_ROOT)

# ----------------- Entry point -----------------
if __name__ == '__main__':
    # Optionally allow passing base dir via CLI argument
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        candidate = Path(arg).expanduser().resolve()
        if candidate.is_dir():
            BASE_DIR = candidate
            OUTPUT_ROOT = BASE_DIR / "Scalogram images"
            print("Using BASE_DIR from argument:", BASE_DIR)
        else:
            print("Argument is not a directory:", arg)
            print("Using default BASE_DIR:", BASE_DIR)

    print("BASE_DIR =", BASE_DIR)
    process_all_subjects(BASE_DIR)
