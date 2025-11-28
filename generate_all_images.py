# generate_all_images.py
"""
Generate Scalogram, Spectrogram and Kurtogram images for SisFall dataset.

Requirements:
  - Python 3.8+
  - numpy, scipy, pandas, matplotlib, pywt, pillow

Install once:
  pip install numpy scipy pandas matplotlib pywt pillow

Usage:
  - Put this script into your project root: D:\3.2\Thesis\SisFall Model\
  - Ensure raw data in: D:\3.2\Thesis\SisFall Model\SisFall_dataset\SA01, SA02, ..., SE01 ...
  - Run:
      python generate_all_images.py
  - Or supply base directory:
      python generate_all_images.py "D:\3.2\Thesis\SisFall Model"

Behavior:
  - Reads all subject folders that start with SA or SE.
  - Processes every .txt file inside each subject folder.
  - Detects activity type by file name prefix: D.. => Daily Living, F.. => Fall.
  - Extracts columns 0,1,2 (X,Y,Z) from each file (robust to separators and trailing commas).
  - Generates three image types per axis: Scalogram (CWT), Spectrogram (STFT), Kurtogram (kurtosis-based map).
  - Saves images to:
      Generated Images/<Type>/<Subject>/<Daily Living|Fall>/FILE_X.png

Notes about kurtogram implementation:
  - A full Kurtogram (Antoni) implementation is non-trivial; here we provide a practical kurtosis-based
    time-frequency map derived from the CWT: for each scale and sliding time window we compute
    kurtosis of the absolute CWT coefficients inside the window. This produces a scale-vs-time kurtosis map
    that is informative and actionable for fault/fall detection tasks.

The script is written to be robust and well-logged. Adjust FS, CWT scales, STFT params, and kurtosis
window length near the top of the script.
"""

import sys
from pathlib import Path
import os
import re
import traceback
import warnings
import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pywt
from scipy import signal, stats
from PIL import Image

# ---------------- USER CONFIGURATION ----------------
# Default base directory (change if you want)
DEFAULT_BASE_DIR = Path(r"/Users/Fahim/Codes/ML/Model/SisFall-Model-main").resolve()
RAW_DATA_DIR_NAME = 'dataset'            # folder containing SA01, SE01 ...
OUTPUT_ROOT_NAME = 'Generated Images'            # root for all generated image types
IMAGE_TYPES = ['Scalogram', 'Spectrogram', 'Kurtogram']

# Sampling frequency in Hz (set to your dataset fs if known)
FS = 100.0

# CWT (Scalogram) params
CWT_WAVELET = 'morl'
CWT_SCALES = np.arange(1, 128)

# STFT (Spectrogram) params
STFT_NFFT = 256
STFT_NPERSEG = 256
STFT_NOOVERLAP = 128

# Kurtogram params (kurtosis over sliding windows on CWT magnitude)
KURTOGRAM_WINDOW_SAMPLES = 128  # length of sliding window in samples (time axis of signal)
KURTOGRAM_STEP = 32             # step between windows

# Image / save params
DPI = 150
RESIZE_TO = None  # e.g., (224,224) to normalize for ML, or None to keep generated size

# Subject folder regex
SUBJECT_PATTERN = re.compile(r'^(SA|SE)\d{2}$', re.IGNORECASE)
TXT_SUFFIX = '.txt'
MIN_COLS_REQUIRED = 3

# ----------------------------------------------------


def find_subject_folders(base_dir: Path):
    dataset_dir = base_dir / RAW_DATA_DIR_NAME
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Raw data folder not found: {dataset_dir}")

    folders = [p for p in sorted(dataset_dir.iterdir()) if p.is_dir() and SUBJECT_PATTERN.match(p.name)]
    return folders


def robust_read_file(file_path: Path):
    """Read a text file with unknown separator (comma, space, tab) and return DataFrame."""
    # Try regex sep first
    try:
        df = pd.read_csv(file_path, header=None, sep=r'[,\s]+', engine='python', comment='#')
    except Exception:
        try:
            df = pd.read_csv(file_path, header=None, delimiter=',', engine='python', comment='#')
        except Exception:
            try:
                df = pd.read_csv(file_path, header=None, delim_whitespace=True, engine='python', comment='#')
            except Exception as e:
                raise RuntimeError(f"Failed reading {file_path}: {e}")

    # drop fully empty columns that sometimes appear due trailing commas
    df = df.dropna(axis=1, how='all')
    return df


def preprocess_signal(col):
    arr = pd.to_numeric(col, errors='coerce').values
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return arr
    arr = arr - np.mean(arr)
    max_abs = np.max(np.abs(arr))
    if max_abs == 0 or not np.isfinite(max_abs):
        return arr
    return arr / max_abs


def make_output_path(base_output: Path, subject: str, cls: str, img_type: str):
    """Return output folder path for given subject and class (Daily Living / Fall) and image type."""
    folder = base_output / img_type / subject / cls
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# ---- Image generation functions ----

def save_figure_to_path(fig, out_path: Path, dpi=DPI, resize_to=RESIZE_TO):
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    if resize_to is not None:
        try:
            img = Image.open(out_path)
            img = img.resize(resize_to, resample=Image.BILINEAR)
            img.save(out_path)
        except Exception as e:
            warnings.warn(f"Could not resize {out_path}: {e}")


def generate_scalogram_image(signal1d, fs, scales, wavelet_name, out_path: Path, title='Scalogram'):
    coeffs, freqs = pywt.cwt(signal1d, scales, wavelet_name, sampling_period=1.0/fs)
    mag = np.abs(coeffs)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    extent = [0, len(signal1d)/fs if fs>0 else len(signal1d), max(scales), min(scales)]
    im = ax.imshow(mag, aspect='auto', extent=extent)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Scale')
    fig.colorbar(im, ax=ax, label='Magnitude')

    save_figure_to_path(fig, out_path)


def generate_spectrogram_image(signal1d, fs, nperseg, noverlap, nfft, out_path: Path, title='Spectrogram'):
    f, t_seg, Sxx = signal.spectrogram(signal1d, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=nfft, scaling='spectrum')
    # convert to dB
    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.pcolormesh(t_seg, f, Sxx_db, shading='gouraud')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='dB')

    save_figure_to_path(fig, out_path)


def generate_kurtogram_image(signal1d, fs, scales, window_samples, step, out_path: Path, title='Kurtogram'):
    # Compute CWT magnitude
    coeffs, freqs = pywt.cwt(signal1d, scales, CWT_WAVELET, sampling_period=1.0/fs)
    mag = np.abs(coeffs)  # shape: (n_scales, n_times)

    n_scales, n_times = mag.shape
    if n_times < 1:
        raise ValueError('Empty CWT result')

    # Sliding window kurtosis along time for each scale
    ws = int(window_samples)
    st = int(step)
    if ws < 3:
        ws = 3
    # number of output time positions
    positions = list(range(0, max(1, n_times - ws + 1), st))
    K = np.zeros((n_scales, len(positions)))

    for si in range(n_scales):
        for pi, p in enumerate(positions):
            window = mag[si, p:p+ws]
            if window.size < 3:
                K[si, pi] = 0.0
            else:
                # kurtosis: fisher=False to get Pearson definition (kurtosis of normal is 3)
                k = stats.kurtosis(window, fisher=False, bias=False)
                # shift to zero-mean (optional): subtract 3 to center normal at 0
                K[si, pi] = k - 3.0

    # Build time axis in seconds
    time_positions = np.array(positions) / fs

    fig, ax = plt.subplots(figsize=(8, 4.5))
    # extent: time start, time end, scale max, scale min
    extent = [time_positions[0] if time_positions.size>0 else 0,
              time_positions[-1] if time_positions.size>0 else (n_times/fs if fs>0 else n_times),
              max(scales), min(scales)]
    im = ax.imshow(K, aspect='auto', extent=extent)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Scale')
    fig.colorbar(im, ax=ax, label='Kurtosis-3')

    save_figure_to_path(fig, out_path)


def generate_xyz_with_original_image(x_norm, y_norm, z_norm, original_signal, fs, out_path: Path, title='XYZ + Original'):
    """
    Create a single figure that shows four subplots stacked vertically:
      1) X normalized signal
      2) Y normalized signal
      3) Z normalized signal
      4) Original (resultant) signal magnitude
    """
    n = min(len(x_norm), len(y_norm), len(z_norm), len(original_signal))
    if n < 8:
        raise ValueError("Insufficient data to plot unified image")

    x = x_norm[:n]
    y = y_norm[:n]
    z = z_norm[:n]
    orig = original_signal[:n]

    t = np.arange(n) / fs if fs and fs > 0 else np.arange(n)

    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)

    # 1) X
    axes[0].plot(t, x, color='tab:blue', linewidth=1.0)
    axes[0].set_ylabel('X')
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)

    # 2) Y
    axes[1].plot(t, y, color='tab:orange', linewidth=1.0)
    axes[1].set_ylabel('Y')
    axes[1].grid(True, alpha=0.3)

    # 3) Z
    axes[2].plot(t, z, color='tab:green', linewidth=1.0)
    axes[2].set_ylabel('Z')
    axes[2].grid(True, alpha=0.3)

    # 4) Original resultant magnitude
    axes[3].plot(t, orig, color='black', linewidth=1.0)
    axes[3].set_xlabel('Time (s)' if fs and fs > 0 else 'Samples')
    axes[3].set_ylabel('Original')
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure_to_path(fig, out_path)


def render_original_signal_image(original_signal, fs, width_px, height_px, title):
    """Render the original resultant signal (1D) into a PIL Image of requested pixel size."""
    n = len(original_signal)
    t = np.arange(n) / fs if fs and fs > 0 else np.arange(n)

    fig, ax = plt.subplots(figsize=(width_px/100.0, height_px/100.0), dpi=100)
    ax.plot(t, original_signal, color='black', linewidth=0.9)
    ax.set_title(title)
    ax.set_xlabel('Time (s)' if fs and fs > 0 else 'Samples')
    ax.set_ylabel('Original')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    # Ensure exact size
    img = img.resize((width_px, height_px), resample=Image.BILINEAR)
    return img


def compose_single_type_image(original_signal, fs, base_output: Path, subject: str, cls: str, base: str, img_type: str):
    """
    Create a 2-row composite image for a single modality (img_type):
      Row 1: Original resultant signal (full width)
      Row 2: X, Y, Z tiles from the given modality
    Saved into the same modality folder as <base>_All.png
    Returns the output path if created, else None.
    """
    # Expected per-axis images
    folder = base_output / img_type / subject / cls
    paths = {
        'X': folder / f"{base}_X.png",
        'Y': folder / f"{base}_Y.png",
        'Z': folder / f"{base}_Z.png",
    }
    if not all(p.exists() for p in paths.values()):
        missing = [ax for ax,p in paths.items() if not p.exists()]
        print(f"[SKIP] {img_type} composite for {base}: missing axis images: {', '.join(missing)}")
        return None

    # Layout
    tile_w, tile_h = 600, 360
    original_h = 260
    pad = 10
    cols = 3

    canvas_w = cols * tile_w + (cols + 1) * pad
    canvas_h = original_h + 3*pad + tile_h

    canvas = Image.new('RGB', (canvas_w, canvas_h), color=(255, 255, 255))

    # Original row
    orig_img = render_original_signal_image(original_signal, fs, canvas_w - 2*pad, original_h, title=f"{base} Original Resultant")
    canvas.paste(orig_img, (pad, pad))

    # Row of X/Y/Z
    y_row = pad + original_h + pad
    x = pad
    for ax in ['X','Y','Z']:
        im = Image.open(paths[ax]).convert('RGB')
        im = im.resize((tile_w, tile_h), resample=Image.BILINEAR)
        canvas.paste(im, (x, y_row))
        x += tile_w + pad

    out_folder = folder
    out_path = out_folder / f"{base}_{img_type}.png"
    canvas.save(out_path)
    print(f"[OK] {img_type} composite image: {out_path}")
    return out_path


# ---- Processing single file ----

def process_single_file(file_path: Path, base_output: Path, fs=FS):
    fname = file_path.name
    base = file_path.stem  # without extension

    # Determine class by filename prefix
    if fname.upper().startswith('D'):
        cls = 'Daily Living'
    elif fname.upper().startswith('F'):
        cls = 'Fall'
    else:
        # fallback: look at pattern <Dxx_ or Fxx_>
        if re.match(r'^[dD]\d{2}_', fname) is not None:
            cls = 'Daily Living'
        elif re.match(r'^[fF]\d{2}_', fname) is not None:
            cls = 'Fall'
        else:
            cls = 'Unknown'

    # Read data robustly
    try:
        df = robust_read_file(file_path)
    except Exception as e:
        print(f"[ERROR] Could not read {file_path}: {e}")
        return

    if df.shape[1] < MIN_COLS_REQUIRED:
        print(f"[SKIP] {file_path} has less than {MIN_COLS_REQUIRED} columns ({df.shape[1]})")
        return

    # Extract X,Y,Z columns (first three)
    norm_signals = {}
    raw_signals = {}
    for i, axis in enumerate(['X', 'Y', 'Z']):
        if i >= df.shape[1]:
            print(f"[SKIP] {base}_{axis}: missing column")
            continue

        raw_col = df.iloc[:, i]
        # Keep a raw numeric copy (drop non-finite), do not normalize
        raw_arr = pd.to_numeric(raw_col, errors='coerce').values
        raw_arr = raw_arr[np.isfinite(raw_arr)]
        sig = preprocess_signal(raw_col)
        if sig.size < 8 or raw_arr.size < 8:
            print(f"[SKIP] {base}_{axis}: insufficient data after preprocessing (len={sig.size})")
            continue
        norm_signals[axis] = sig
        raw_signals[axis] = raw_arr

        # Build output folder and filename
        subject = file_path.parent.name
        out_folder_scal = make_output_path(base_output, subject, cls, 'Scalogram')
        out_folder_spec = make_output_path(base_output, subject, cls, 'Spectrogram')
        out_folder_kurt = make_output_path(base_output, subject, cls, 'Kurtogram')

        out_name = f"{base}_{axis}.png"

        # Generate Scalogram
        try:
            out_path = out_folder_scal / out_name
            generate_scalogram_image(sig, fs, CWT_SCALES, CWT_WAVELET, out_path, title=f"{base}_{axis} Scalogram")
        except Exception as e:
            print(f"[ERROR] Scalogram {base}_{axis}: {e}")
            traceback.print_exc()

        # Generate Spectrogram
        try:
            out_path = out_folder_spec / out_name
            generate_spectrogram_image(sig, fs, STFT_NPERSEG, STFT_NOOVERLAP, STFT_NFFT, out_path, title=f"{base}_{axis} Spectrogram")
        except Exception as e:
            print(f"[ERROR] Spectrogram {base}_{axis}: {e}")
            traceback.print_exc()

        # Generate Kurtogram
        try:
            out_path = out_folder_kurt / out_name
            generate_kurtogram_image(sig, fs, CWT_SCALES, KURTOGRAM_WINDOW_SAMPLES, KURTOGRAM_STEP, out_path, title=f"{base}_{axis} Kurtogram")
        except Exception as e:
            print(f"[ERROR] Kurtogram {base}_{axis}: {e}")
            traceback.print_exc()

    # After per-axis generation, try to create unified image if all axes are present
    if all(ax in norm_signals for ax in ['X', 'Y', 'Z']):
        try:
            # Align to minimum available length among raw signals for original resultant
            n_min_raw = min(len(raw_signals['X']), len(raw_signals['Y']), len(raw_signals['Z']))
            if n_min_raw >= 8:
                rx = raw_signals['X'][:n_min_raw]
                ry = raw_signals['Y'][:n_min_raw]
                rz = raw_signals['Z'][:n_min_raw]
                original_resultant = np.sqrt(rx*rx + ry*ry + rz*rz)
            else:
                original_resultant = norm_signals['X']  # fallback, should not happen due to earlier checks

            subject = file_path.parent.name
            out_folder_unified = make_output_path(base_output, subject, cls, 'Plotting')
            out_name_unified = f"{base}_XYZ.png"
            out_path_unified = out_folder_unified / out_name_unified

            # Align normalized signals too
            n_min_norm = min(len(norm_signals['X']), len(norm_signals['Y']), len(norm_signals['Z']))
            generate_xyz_with_original_image(
                norm_signals['X'][:n_min_norm],
                norm_signals['Y'][:n_min_norm],
                norm_signals['Z'][:n_min_norm],
                original_resultant[:n_min_norm],
                fs,
                out_path_unified,
                title=f"{base} XYZ + Original"
            )
        except Exception as e:
            print(f"[ERROR] Unified XYZ image {base}: {e}")
            traceback.print_exc()

        # Build per-type composite images and remove per-axis tiles to keep only one image per sample per type
        try:
            subject = file_path.parent.name
            n_use = n_min_norm if 'n_min_norm' in locals() else min(len(norm_signals['X']), len(norm_signals['Y']), len(norm_signals['Z']))
            orig_for_comp = original_resultant[:n_use]
            for img_type in ['Scalogram', 'Spectrogram', 'Kurtogram']:
                out = compose_single_type_image(orig_for_comp, fs, base_output, subject, cls, base, img_type)
                if out is not None:
                    # delete per-axis images for this modality
                    folder = make_output_path(base_output, subject, cls, img_type)
                    for ax in ['X','Y','Z']:
                        try:
                            (folder / f"{base}_{ax}.png").unlink()
                        except FileNotFoundError:
                            pass
        except Exception as e:
            print(f"[ERROR] Per-type composite/cleanup {base}: {e}")
            traceback.print_exc()

    print(f"[DONE] {file_path}")


# ---- Main pipeline ----

def run_pipeline(base_dir: Path):
    base_dir = base_dir.resolve()
    raw_dir = base_dir / RAW_DATA_DIR_NAME
    output_root = base_dir / OUTPUT_ROOT_NAME
    output_root.mkdir(parents=True, exist_ok=True)

    subjects = find_subject_folders(base_dir)
    if not subjects:
        print(f"No subject folders found in {raw_dir}. Make sure folder names are SAxx or SExx and data placed inside.")
        return

    print(f"Found {len(subjects)} subject folders. Processing...")

    for subj in subjects:
        print(f"Processing subject: {subj.name}")
        txt_files = sorted([p for p in subj.iterdir() if p.is_file() and p.suffix.lower() == TXT_SUFFIX])
        if not txt_files:
            print(f"  No .txt files found in {subj}")
            continue

        for f in txt_files:
            try:
                process_single_file(f, output_root, fs=FS)
            except Exception as e:
                print(f"Unexpected error processing {f}: {e}")
                traceback.print_exc()

    print("\nProcessing complete. Generated images saved under:", output_root)


# ---------------- Entry point ----------------
if __name__ == '__main__':
    # parse optional CLI arg for base dir
    base = DEFAULT_BASE_DIR
    if len(sys.argv) > 1:
        arg = Path(sys.argv[1])
        if arg.is_dir():
            base = arg.resolve()
        else:
            print(f"Argument is not a directory: {arg}. Using default: {base}")

    print("Base directory:", base)
    try:
        run_pipeline(base)
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
