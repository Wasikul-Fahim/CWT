SisFall Plotting Toolkit

This repository helps you generate plots for the SisFall dataset. It produces three common time–frequency visualizations for each accelerometer axis (X, Y, Z):
- Scalogram (Continuous Wavelet Transform)
- Spectrogram (Short-Time Fourier Transform)
- Kurtogram-like map (kurtosis over the CWT magnitude)

What you’ll get
- For each data file and each plot type (Scalogram, Spectrogram, Kurtogram), a SINGLE composite image that contains:
  - Row 1: Original resultant signal (computed from X, Y, Z)
  - Row 2: Three tiles for X, Y, and Z of that plot type
- Saved under: Generated Images/<Type>/<Subject>/<Daily Living|Fall>/<FILENAME>_<Type>.png
- The intermediate per-axis images (<FILENAME>_X.png, _Y.png, _Z.png) are generated temporarily and then deleted after the composite is saved.

Requirements
- Python 3.8+
- Packages: numpy, scipy, pandas, matplotlib, pywt, pillow

Install
pip install numpy scipy pandas matplotlib pywt pillow

Dataset layout
Place the SisFall subject folders inside dataset/ at the project root:
- dataset/SA01, dataset/SA02, … or dataset/SE01, …
- Each subject folder contains .txt files like F01_SA02_R02.txt

Quick start
1) Clone/open this repo
2) Put the SisFall dataset into the dataset/ folder (so you have dataset/SA01, etc.)
3) Run the generator:
   python generate_all_images.py

Optional: specify a custom base directory that contains the dataset/ folder:
   python generate_all_images.py "/path/to/your/project/root"

Configuration (generate_all_images.py)
Adjust these near the top of the script if needed:
- FS: Sampling frequency in Hz (default 100.0)
- Scalogram (CWT): CWT_WAVELET='morl', CWT_SCALES=np.arange(1, 128)
- Spectrogram (STFT): STFT_NFFT=256, STFT_NPERSEG=256, STFT_NOOVERLAP=128
- Kurtogram-like map: KURTOGRAM_WINDOW_SAMPLES=128, KURTOGRAM_STEP=32
- Saving: DPI=150, RESIZE_TO=None (e.g., set to (224,224) to normalize image sizes)
- Output root folder name: OUTPUT_ROOT_NAME = 'Generated Images'

How plotting works (high-level)
- Scalogram (CWT): Shows how signal content varies with scale (related to frequency) over time using wavelets. Good for transients and non-stationary patterns.
- Spectrogram (STFT): Splits the signal into short overlapping windows, computes FFT in each window, and shows power vs time and frequency. Good for stationary or quasi-stationary components.
- Kurtosis map ("Kurtogram") here: Not the full Antoni kurtogram, but a practical variant. We compute the CWT, take magnitudes, then slide a time window along each scale and compute kurtosis. The map highlights impulsiveness across scales and time (kurtosis > 0 means heavier tails vs. Gaussian).

Interpreting the plots
- Time axis: seconds (uses FS to convert sample index to time).
- Frequency/Scale axis: For spectrogram, it’s frequency in Hz; for scalogram and kurtosis map, it’s scale (inverse-related to frequency depending on the mother wavelet).
- Color: Magnitude (Scalogram), dB power (Spectrogram), and Kurtosis-3 (Kurtogram-like).

Customizing for your dataset
- If your sampling rate differs, set FS to your dataset’s sampling frequency to keep axes meaningful.
- If your records are short/long, tune STFT_NPERSEG/NOOVERLAP/NFFT or CWT_SCALES.
- If images are too large or small, adjust DPI or RESIZE_TO.

Example: plotting a single file (inside Python)
You can also call functions directly to plot a single array:

from pathlib import Path
import numpy as np
from generate_all_images import (
    preprocess_signal, generate_scalogram_image,
    generate_spectrogram_image, generate_kurtogram_image,
    CWT_SCALES, CWT_WAVELET, STFT_NPERSEG, STFT_NOOVERLAP, STFT_NFFT, FS
)

# Suppose you loaded a 1D numpy array `sig` sampled at FS Hz
sig = np.random.randn(5000)
# Normalize
sig = (sig - sig.mean()) / (np.abs(sig).max() + 1e-12)

out = Path('demo.png')
# Scalogram
generate_scalogram_image(sig, FS, CWT_SCALES, CWT_WAVELET, out, title='Demo Scalogram')

# Spectrogram
generate_spectrogram_image(sig, FS, STFT_NPERSEG, STFT_NOOVERLAP, STFT_NFFT, out.with_name('demo_spec.png'), title='Demo Spectrogram')

# Kurtosis map
generate_kurtogram_image(sig, FS, CWT_SCALES, 128, 32, out.with_name('demo_kurt.png'), title='Demo Kurtogram')

Troubleshooting
- No subject folders found: Ensure dataset/ has subfolders like SA01, SA02, … or SE01, and names match exactly.
- File read errors: The script tries multiple delimiters. Corrupt lines may still fail; check the .txt file.
- Empty/short signals: Some files may have fewer than 3 numeric columns or very short length; these are skipped.
- Missing images: Check console logs for [ERROR] or [SKIP] messages.

Where outputs are saved
Generated Images/<Type>/<Subject>/<Daily Living|Fall>/FILENAME_AXIS.png

Credits
- Uses numpy, scipy, pandas, matplotlib, pywt, and pillow.

Current date/time
- 2025-11-27 17:18