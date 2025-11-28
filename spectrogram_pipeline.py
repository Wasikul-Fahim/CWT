# -*- coding: utf-8 -*-
"""
spectrogram_pipeline.py

Automatically generate spectrogram images (X, Y, Z) for all files in SAxx folders.

- Searches for folders named SAxx in BASE_DIR
- For each .txt file inside SAxx, reads the first 3 columns (handles commas/spaces/tabs)
- Generates spectrograms using Matplotlib’s signal processing
- Saves images to: BASE_DIR / "Spectrogram images" / SAxx / <file_basename>_X.png, etc.

Notes:
- Adjust FS (sampling rate) to match your dataset.
- Adjust NFFT, noverlap, and cmap for better visualization.
"""

import os
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
from PIL import Image

# ---------------- USER SETTINGS ----------------
BASE_DIR = Path.cwd()  # or Path(r"D:\3.2\Thesis\SisFall Model")
OUTPUT_ROOT = BASE_DIR / "Spectrogram images"
SUBJECT_FOLDER_PATTERN = re.compile(r'^SA\d+', re.IGNORECASE)
TXT_SUFFIX = '.txt'

FS = 100.0  # Sampling frequency (Hz)
NFFT = 256  # Number of data points per FFT window
NOVERLAP = 128  # Overlap between windows
CMAP = 'magma'  # Colormap for spectrogram ('viridis', 'plasma', 'inferno', etc.)

DPI = 150
RESIZE_TO = None  # e.g., (224, 224)
MIN_COLS_REQUIRED = 3
# ------------------------------------------------


def find_subject_folders(base_dir: Path):
    """Return list of subject folder Paths matching SAxx pattern."""
    return sorted([p for p in base_dir.iterdir() if p.is_dir() and SUBJECT_FOLDER_PATTERN.match(p.name)])


def robust_read_txt(file_path: Path):
    """Read a .txt/CSV-like file with flexible separators."""
    try:
        df = pd.read_csv(file_path, header=None, sep=r'[,\s]+', engine='python', comment='#')
    except Exception:
        try:
            df = pd.read_csv(file_path, header=None, delimiter=',', engine='python', comment='#')
        except Exception:
            try:
                df = pd.read_csv(file_path, header=None, delim_whitespace=True, engine='python', comment='#')
            except Exception as ex:
                raise RuntimeError(f"Failed to read {file_path}: {ex}")
    df = df.dropna(axis=1, how='all')
    return df


def preprocess_signal(arr):
    """Clean, normalize, and center the signal."""
    arr = np.asarray(arr, dtype=float)
    mask = np.isfinite(arr)
    arr = arr[mask]
    if arr.size == 0:
        return arr
    arr = arr - np.mean(arr)
    max_abs = np.max(np.abs(arr))
    if max_abs == 0 or np.isnan(max_abs):
        return arr
    return arr / max_abs


def save_spectrogram_image(sig, fs, out_path, dpi=DPI, resize_to=RESIZE_TO):
    """Generate and save spectrogram image."""
    plt.figure(figsize=(8, 4.5))
    plt.specgram(sig, NFFT=NFFT, Fs=fs, noverlap=NOVERLAP, cmap=CMAP)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label='Intensity (dB)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()

    if resize_to is not None:
        try:
            img = Image.open(out_path)
            img = img.resize(resize_to, resample=Image.BILINEAR)
            img.save(out_path)
        except Exception as e:
            print(f"Warning: failed to resize {out_path}: {e}")


def process_file(file_path: Path, output_folder: Path, fs=FS):
    """Generate spectrograms for first 3 columns of a .txt file."""
    base_name = file_path.stem
    try:
        df = robust_read_txt(file_path)
    except Exception as e:
        print(f"ERROR reading {file_path}: {e}")
        return

    df = df.dropna(axis=1, how='all')
    if df.shape[1] < MIN_COLS_REQUIRED:
        print(f"SKIP {file_path.name}: less than {MIN_COLS_REQUIRED} columns.")
        return

    output_folder.mkdir(parents=True, exist_ok=True)
    axis_labels = ['X', 'Y', 'Z']

    for i, axis in enumerate(axis_labels):
        if i >= df.shape[1]:
            print(f"SKIP {base_name}_{axis}: missing column {i}")
            continue

        col = pd.to_numeric(df.iloc[:, i], errors='coerce').values
        sig = preprocess_signal(col)

        if sig.size < NFFT:
            print(f"SKIP {base_name}_{axis}: insufficient data (len={sig.size})")
            continue

        out_path = output_folder / f"{base_name}_{axis}.png"
        try:
            save_spectrogram_image(sig, fs, out_path)
            print(f"Saved: {out_path}")
        except Exception as e:
            print(f"ERROR saving {out_path}: {e}")
            traceback.print_exc()


def process_all_subjects(base_dir: Path = BASE_DIR):
    """Process all SAxx folders."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    subjects = find_subject_folders(base_dir)
    if not subjects:
        print(f"No SAxx folders found in {base_dir}")
        return

    print(f"Found {len(subjects)} subject folders:")
    for s in subjects:
        print(" -", s.name)

    for subj in subjects:
        print(f"\nProcessing subject: {subj.name}")
        out_sub = OUTPUT_ROOT / subj.name
        out_sub.mkdir(parents=True, exist_ok=True)
        txt_files = sorted([f for f in subj.iterdir() if f.is_file() and f.suffix.lower() == TXT_SUFFIX])
        if not txt_files:
            print(f"  No .txt files in {subj}")
            continue

        print(f"  Found {len(txt_files)} .txt files in {subj.name}")
        for txt in txt_files:
            process_file(txt, out_sub, fs=FS)

    print("\nAll done. Check output at:", OUTPUT_ROOT)


# ----------------- Entry Point -----------------
if __name__ == '__main__':
    if len(sys.argv) > 1:
        arg = Path(sys.argv[1]).expanduser().resolve()
        if arg.is_dir():
            BASE_DIR = arg
            OUTPUT_ROOT = BASE_DIR / "Spectrogram images"
            print("Using BASE_DIR from argument:", BASE_DIR)
        else:
            print("Invalid argument, using default BASE_DIR:", BASE_DIR)

    print("BASE_DIR =", BASE_DIR)
    process_all_subjects(BASE_DIR)
