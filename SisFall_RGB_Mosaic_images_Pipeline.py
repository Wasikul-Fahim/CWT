# SisFall_RGB_Mosaic_Pipeline_FINAL_v2.py
"""
SisFall Dataset  ->  RGB Scalogram / Spectrogram / Kurtogram (in-memory only)
                  ->  TWO mosaic variants, generated in a single pass:

      Mosaic_ADL_2x2        -> rows=[Acc1,Gyro] cols=[scalogram,spectrogram]
                               cell 188x188 -> canvas 384x384 (activity ID)
      Mosaic_PatchCore_2x2  -> rows=[Acc1,Gyro] cols=[scalogram,kurtogram]
                               cell 224x224 -> canvas 456x456 (anomaly / fall)

  Acc2 (second waist accelerometer) is intentionally dropped -- redundant
  with Acc1 at the same mount location; Gyro is the genuinely complementary
  modality (angular velocity vs. linear acceleration). This also makes the
  ADL canvas square (2 rows x 2 cols), so resizing to a square network input
  never distorts the aspect ratio.

Both variants are built from the SAME underlying per-window CWT/STFT/kurtosis
computation -- the transforms are computed exactly ONCE per device per window
regardless of how many mosaic variants you configure, then subsets of the
three representation arrays are recombined into each variant's mosaic. No
recomputation, no second pass over the raw data.

===============================================================================
  CHANGES IN THIS VERSION
===============================================================================
  FIX A -- Kurtogram no longer includes the top (highest-frequency) CWT
           scales, which are noise-dominated and were rendering as a flat
           white/bright band that ate into the usable luminance range.
           KURT_MIN_SCALE_IDX = 8 skips scales[0:8]; the kurtogram is now
           computed and rendered only over scales[8:], with the y-axis
           extent adjusted to match. This returns that chunk of luminance
           range to the mid/low-scale content, which is where fall-relevant
           impulsiveness actually shows up.

  FIX B -- All debug-image code removed (SAVE_DEBUG_IMAGES, DEBUG_DEVICES,
           DEBUG_MAX_PER_CLASS, _debug_figure_common, save_debug_scalogram/
           spectrogram/kurtogram, and the CMAP_KURTOGRAM_DEBUG colour path).
           Nothing but the two mosaic variants is written to disk. This also
           removes the now-unused gridspec/ticker imports and the double
           kurtogram render (model + debug) -- kurtogram is rendered ONCE
           per axis now, using CMAP_KURTOGRAM_MODEL (cividis) only.

  FIX C -- Dual mosaic variants (MOSAIC_VARIANTS config dict). Each device's
           three representation arrays are computed once per window; both
           mosaics are assembled from that same in-memory result and saved
           to their own output folders in the same pass.

===============================================================================
  CARRIED FORWARD FROM THE PREVIOUS FINAL VERSION
===============================================================================
  - Global colour bounds computed from ADL (Daily Living) files ONLY, never
    Fall files (compute_global_bounds).
  - ACTIVITY_LABELS corrected to match the official SisFall Readme.
  - Only mosaic images are saved -- no per-representation / per-device PNGs
    (SAVE_INDIVIDUAL_REPRESENTATIONS stays available, default False).
  - Bug-1/2/4/5/6/7/10 fixes from earlier versions (normalize-once,
    context-pad, STFT edge-frame insertion, kurtogram orientation, full-
    length-only windows) are all preserved unchanged.

===============================================================================
  OUTPUT FOLDER STRUCTURE (final)
===============================================================================
  Generated_Images_Mosaic/
    Mosaic_ADL_2x2/            (188x188 cells -> 384x384 canvas)
      Daily Living/{activity}/{subject}/{stem}.png
      Fall/{subject}/{stem}.png
    Mosaic_PatchCore_2x2/      (224x224 cells -> 456x456 canvas)
      Daily Living/{activity}/{subject}/{stem}.png
      Fall/{subject}/{stem}.png

  FIX D -- RESIZE_TO is now per-variant ({variant_name: (w,h)}), since
           Mosaic_ADL_2x2 and Mosaic_PatchCore_2x2 need different final
           pixel sizes. Every representation is still rendered/cached at a
           single shared BASE_CELL_SIZE (224x224) exactly once per window;
           build_mosaic() resizes each cached cell down to the requesting
           variant's target size right before pasting. The CWT/STFT/
           kurtosis computation and the matplotlib render are NEVER
           repeated per variant -- only this final, cheap PIL resize is.

===============================================================================
  VISUAL QA CHECKLIST (use this to eyeball a batch of generated images)
===============================================================================
  Scalogram    MUST show: fan-shaped ridges per stride/impact cycle; clear
               scale-band separation between periodic ADL motion and a
               fall's broadband vertical spike; intensity varies with
               activity (never uniformly washed out or uniformly dark).
               MUST NOT show: flat/uniform colour regardless of activity.

  Spectrogram  MUST show: distinct horizontal frequency bands tracking
               cadence; a fall shows a short broadband vertical streak
               across most frequency bins at impact.
               MUST NOT show: banding artifacts at window edges (context-pad
               already prevents this); a single flat colour block.

  Kurtogram    MUST show: mostly neutral/near-gray background (steady-state)
               with a sharp, spatially-tight, high-contrast region exactly at
               fall impact; the quiet-then-impulsive transition should be
               visually obvious in the mid/low scale rows (now that FIX A
               removes the noise-dominated top-scale band).
               MUST NOT show: a uniformly white/bright top strip (that was
               the noise artifact FIX A removes); uniform gray with no
               visible impact region even for confirmed fall files.

Usage:
    python SisFall_RGB_Mosaic_Pipeline_FINAL_v2.py
    python SisFall_RGB_Mosaic_Pipeline_FINAL_v2.py  path/to/project
"""

import sys
import io
import re
import traceback
import warnings
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pywt
from scipy import signal as sps, stats
from PIL import Image

# ===============================================================================
#  CONFIG  -- edit these to match your setup
# ===============================================================================
DEFAULT_BASE_DIR = Path("/Users/wasikulfahim/Codes/ML/Img_Gen").resolve()
RAW_DATA_DIR      = 'SisFall_dataset'
OUTPUT_DIR        = 'Generated_Images_Mosaic'
SUBJECT_PATTERN   = re.compile(r'^(SA|SE)\d{2}$', re.IGNORECASE)

# ── Sensor / signal settings ──────────────────────────────────────────────────
FS             = 200.0    # SisFall sensor rate (Hz)
TIME_START_S   = None
TIME_END_S     = None
SPLIT_WINDOW_S = 5.0      # window length (seconds)
SPLIT_OVERLAP  = 0.5      # 50% overlap -> hop = 2.5 s

# ── CWT settings (Scalogram + Kurtogram both use CWT magnitude) ──────────────
CWT_WAVELET  = 'morl'             # Morlet wavelet
CWT_SCALES   = np.arange(1, 129)  # 1..128 (128 scales)

# ── STFT settings (Spectrogram) ───────────────────────────────────────────────
STFT_NPERSEG  = 256
STFT_NOVERLAP = 128
STFT_NFFT     = 256

# ── Kurtogram settings ────────────────────────────────────────────────────────
KURT_WINDOW = 128     # sliding window length (samples along CWT time axis)
KURT_STEP   = 64      # hop between windows

# FIX A: skip the top (highest-frequency) CWT scales for the kurtogram only --
# these are noise-dominated and were rendering as a flat white/bright band.
# scales[0:KURT_MIN_SCALE_IDX] are excluded; scalogram/spectrogram are
# unaffected (they still use the full scale range).
KURT_MIN_SCALE_IDX = 8

# Context extension -- kills CWT/STFT edge bleedback at window boundaries
CONTEXT_S = 3.0

# ── Colourmaps ─────────────────────────────────────────────────────────────────
CMAP_SCALOGRAM   = 'viridis'   # dark-purple/blue -> green -> yellow (monotonic luma)
CMAP_SPECTROGRAM = 'inferno'   # black -> dark-red -> orange -> yellow (monotonic luma)
# Sequential, monotonic-luminance colormap -- preserves the SIGN of kurtosis
# (impulsive vs. very-regular) through grayscale conversion (colour Option 2).
CMAP_KURTOGRAM_MODEL = 'cividis'

# ── Output image settings ─────────────────────────────────────────────────────
DPI       = 150
FIG_W     = 8.0
FIG_H     = 4.5
BASE_CELL_SIZE = (224, 224)   # shared cache resolution -- every representation is
                              # rendered/resized to THIS size exactly once per window,
                              # then build_mosaic() downsizes per-variant below. This is
                              # what keeps the "compute once, reuse across variants" design
                              # intact even though the two variants need different final
                              # pixel sizes.
RESIZE_TO = {
    'Mosaic_ADL_2x2':       (188, 188),   # -> 384x384 canvas, matches your classifier input
    'Mosaic_PatchCore_2x2': (224, 224),   # unchanged, keep PatchCore's expected resolution
}

# ── Mosaic assembly settings ──────────────────────────────────────────────────
# FIX C: two mosaic variants, generated in the SAME pass from the same
# per-window transforms. Each variant only needs its own column list --
# build_mosaic() is fully generic over rows x cols, so adding, removing, or
# reshaping a variant never touches the CWT/STFT/kurtosis computation code.
MOSAIC_ROW_DEVICES = ['Acc1', 'Gyro']     # rows -- shared by all variants
MOSAIC_VARIANTS: Dict[str, List[str]] = {
    'Mosaic_ADL_2x2':       ['scalogram', 'spectrogram'],  # activity-ID input
    'Mosaic_PatchCore_2x2': ['scalogram', 'kurtogram'],    # anomaly-detection input
}
MOSAIC_GUTTER_PX    = 8            # neutral gutter width between cells (px)
MOSAIC_GUTTER_COLOR = (0, 0, 0)    # black gutters

# To add a future variant, e.g. a 1x3 kurtogram-only mosaic:
#   MOSAIC_VARIANTS['Mosaic_Kurtogram_1x3'] = ['kurtogram']
#   (and set its own MOSAIC_ROW_DEVICES subset if you also want fewer rows)

# ── Only the mosaics are saved -- no per-representation / per-device PNGs ────
SAVE_INDIVIDUAL_REPRESENTATIONS = False   # flip True only if you specifically
                                            # need standalone Scalogram/
                                            # Spectrogram/Kurtogram PNGs again.
REPRESENTATIONS_DIRNAME = 'Representations'

# ── Global colour bounds ──────────────────────────────────────────────────────
#  python SisFall_RGB_Mosaic_Pipeline_FINAL_v5.py "D:\4.2\CSE 400-B\SisFall-Model" --bounds-only
# Bounds are computed from ADL (Daily Living) files ONLY -- never Fall files.
# Falls are rendered against this ADL-only scale, so a genuine fall naturally

# clips/saturates toward the colour extreme instead of being pre-accounted for.
USE_HARDCODED_BOUNDS = True   # set True to skip the ADL-only scan and use HARDCODED_BOUNDS below
HARDCODED_BOUNDS = {
    # Placeholder values -- INVALID until you re-run the ADL-only scan below
    # (with FIX A active) and copy its printed numbers in here.
    #'scalogram':   (0.0,   3.0),
    #'spectrogram': (-80.0, 0.0),
    #'kurtogram':   (-15.0, 15.0),

    'scalogram':   (3.787852656958628e-05, 3.6432822492929433),
    'spectrogram': (-110.90984930891089, -16.119357656447107),
    'kurtogram':   (-7.1396121267864565, 7.1396121267864565),
}
GLOBAL_BOUNDS = {
    'scalogram':   (None, None),
    'spectrogram': (None, None),
    'kurtogram':   (None, None),
}

# ── Devices: SisFall CSV columns: Acc1=0,1,2 | Gyro=3,4,5 | Acc2=6,7,8 ────────
DEVICES = {
    'Acc1': [0, 1, 2],
    'Gyro': [3, 4, 5],
    'Acc2': [6, 7, 8],
}

# ── Daily-Living activity label map (Step 3 of the plan) ─────────────────────
# Corrected to match the OFFICIAL SisFall Readme.pdf exactly.
        # ACTIVITY_LABELS = {
        #     'D01': 'Walking_Slow',            # Walking slowly
        #     'D02': 'Walking_Fast',            # Walking quickly
        #     'D03': 'Jogging_Slow',            # Jogging slowly
        #     'D04': 'Jogging_Fast',            # Jogging quickly
        #     'D05': 'Stairs_Slow',             # Walking up/down stairs slowly
        #     'D06': 'Stairs_Fast',             # Walking up/down stairs quickly
        #     'D07': 'Sit_HalfHeight_Slow',     # Sit in half-height chair, slowly
        #     'D08': 'Sit_HalfHeight_Fast',     # Sit in half-height chair, quickly
        #     'D09': 'Sit_LowHeight_Slow',      # Sit in low-height chair, slowly
        #     'D10': 'Sit_LowHeight_Fast',      # Sit in low-height chair, quickly
        #     'D11': 'Collapse_Into_Chair',     # Sitting, trying to get up, collapse into chair
        #     'D12': 'Lying_Slow_SitAgain',     # Sitting, lying down slowly, sit again
        #     'D13': 'Lying_Fast_SitAgain',     # Sitting, lying down quickly, sit again
        #     'D14': 'Back_Lateral_Change',     # Back <-> lateral position change
        #     'D15': 'Standing_KneeBend',       # Standing, bending at knees, getting up
        #     'D16': 'Standing_NoKneeBend',     # Standing, bending without knees, getting up
        #     'D17': 'Car_InOut',               # Get into car, sit, get out
        #     'D18': 'Stumble_Walking',         # Stumble while walking
        #     'D19': 'Gentle_Jump',             # Gentle jump reaching for object
        # }

        # # Training-order groups from the corrected plan (Stage 4) -- informational,
        # # used only in the printed summary at the bottom of this file.
        # ACTIVITY_TRAINING_GROUPS = {
        #     1: ['D01', 'D02', 'D03', 'D04'],
        #     2: ['D05', 'D06', 'D15', 'D16'],
        #     3: ['D07', 'D08', 'D09', 'D10', 'D17'],
        #     4: ['D11', 'D12', 'D13', 'D14', 'D18', 'D19'],   # train last, validate separately
        # }

MIN_COLS = 9
# ===============================================================================


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def read_df(fpath: Path) -> pd.DataFrame:
    """Try three common SisFall separators; clean trailing semicolons."""
    df = None
    for kw in [dict(sep=r'[,\s]+', engine='python'),
               dict(sep=','),
               dict(sep=r'\s+', engine='python')]:
        try:
            df = pd.read_csv(fpath, header=None, comment='#', **kw)
            break
        except Exception:
            continue
    if df is None:
        raise IOError(f"Cannot parse {fpath}")
    try:
        df = df.map(lambda x: x.replace(';', '').strip()
                    if isinstance(x, str) else x)
    except Exception:
        pass
    return df.dropna(axis=1, how='all')


def normalize(arr: np.ndarray) -> np.ndarray:
    """
    Zero-mean + peak-normalise to [-1, +1].
    Bug-1 fix: called ONCE on the FULL signal, not per window (Step 1).
    """
    arr = arr.astype(float) - np.nanmean(arr)
    m   = np.nanmax(np.abs(arr))
    return arr / m if (np.isfinite(m) and m > 0) else arr


def get_context_slice(full_arr: np.ndarray,
                       t0: float, t1: float) -> Tuple[np.ndarray, float]:
    """
    Return (ctx_array, ctx_t0_seconds).
    Bug-2/4/5 fix: context drawn from SAME full_arr for overlapping windows.
    (3s context pad on both sides, kills edge bleedback.)
    """
    n      = len(full_arr)
    ctx_t0 = max(0.0, t0 - CONTEXT_S)
    ctx_t1 = min(n / FS, t1 + CONTEXT_S)
    i0     = int(np.floor(ctx_t0 * FS))
    i1     = int(np.ceil(ctx_t1 * FS))
    i0, i1 = max(0, i0), min(n, i1)
    return full_arr[i0:i1], i0 / FS


def build_segments(n_samples: int) -> List[Tuple[float, float]]:
    """Sliding window -- emits ONLY full-length windows (Bug-10 fix, Step 2)."""
    dur   = n_samples / FS
    start = float(TIME_START_S) if TIME_START_S is not None else 0.0
    end   = min(float(TIME_END_S), dur) if TIME_END_S is not None else dur
    if end <= start:
        return []
    w   = float(SPLIT_WINDOW_S)
    hop = w * (1.0 - float(SPLIT_OVERLAP))
    segs, s = [], start
    while s + w <= end + 1e-9:
        segs.append((round(s, 6), round(s + w, 6)))
        s += hop
    return segs


        # def activity_label_for(base: str) -> str:
        #     """Step 3: SisFall filename code -> activity label (Daily Living only)."""
        #     code = base[:3].upper()
        #     return ACTIVITY_LABELS.get(code, code)


def render_to_grayscale(render_fn) -> np.ndarray:
    """
    Render onto a borderless black-bg matplotlib figure, capture to an
    in-memory PNG buffer (no disk write), resize LANCZOS -> 224x224,
    and return gray_arr. Grayscale uses PIL's .convert('L') which applies
    ITU-R BT.601 luma:  Y = 0.299*R + 0.587*G + 0.114*B
    """
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor='black')
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('black')

    render_fn(ax)

    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.patch.set_visible(False)

    buf = io.BytesIO()
    fig.savefig(buf, dpi=DPI, bbox_inches=None, pad_inches=0,
                facecolor='black', format='png')
    plt.close(fig)
    buf.seek(0)

    colour_img = Image.open(buf).convert('RGB').resize(BASE_CELL_SIZE, Image.LANCZOS)
    gray_arr   = np.array(colour_img.convert('L'), dtype=np.uint8)
    return gray_arr


# ─────────────────────────────────────────────────────────────────────────────
#  CORE MATRIX COMPUTATIONS (per single-axis signal)
# ─────────────────────────────────────────────────────────────────────────────

def _cwt_matrix(full_sig: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """CWT magnitude, context-padded then cropped to [t0, t1].
    Full scale range (1..128) -- used as-is for the scalogram, and trimmed
    internally by _kurtogram_matrix() for the kurtogram (FIX A)."""
    ctx, ctx_t0 = get_context_slice(full_sig, t0, t1)
    coeffs, _   = pywt.cwt(ctx, CWT_SCALES, CWT_WAVELET, sampling_period=1.0 / FS)
    mag         = np.abs(coeffs)
    t_ctx    = ctx_t0 + np.arange(mag.shape[1]) / FS
    col_mask = (t_ctx >= t0 - 0.5 / FS) & (t_ctx <= t1 + 0.5 / FS)
    return mag[:, col_mask]


def _stft_matrix(full_sig: np.ndarray, t0: float, t1: float
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """STFT power (dB), context-padded, time-masked with Bug-4/5 synthetic-
    frame edge fixes so the image always spans [t0, t1]."""
    ctx, ctx_t0 = get_context_slice(full_sig, t0, t1)
    f, t_seg, Sxx = sps.spectrogram(
        ctx, fs=FS, window='hann',
        nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP,
        nfft=STFT_NFFT, scaling='spectrum')
    Sxx_db = 10.0 * np.log10(Sxx + 1e-12)
    t_abs  = t_seg + ctx_t0

    mask  = (t_abs >= t0 - 1e-9) & (t_abs <= t1 + 1e-9)
    t_win = t_abs[mask]
    S_win = Sxx_db[:, mask]

    if t_win.size > 0 and t_win[0] > t0 + 1e-9:        # Bug-4 fix
        t_win = np.insert(t_win, 0, t0)
        S_win = np.hstack([S_win[:, [0]], S_win])
    if t_win.size > 0 and t_win[-1] < t1 - 1e-9:        # Bug-5 fix
        t_win = np.append(t_win, t1)
        S_win = np.hstack([S_win, S_win[:, [-1]]])
    if t_win.size < 2:
        t_win = np.array([t0, t1])
        S_win = np.hstack([S_win, S_win]) if S_win.size else np.zeros((len(f), 2))
    return f, t_win, S_win


def _kurtogram_matrix(mag_win: np.ndarray) -> np.ndarray:
    """Sliding-window excess kurtosis per CWT scale row, computed ONLY over
    scales[KURT_MIN_SCALE_IDX:] (FIX A -- excludes noise-dominated top/high-
    frequency scales that were rendering as a flat white band).
    K near 0 = Gaussian/steady-state; K >> 0 = impulsive (fall impact);
    K << 0 = very regular/sinusoidal.
    Scale-invariant by construction: kurtosis(a*X) = kurtosis(X) for any
    positive constant a, so this is unaffected by normalize()'s per-file
    peak scaling."""
    trimmed = mag_win[KURT_MIN_SCALE_IDX:, :]
    n_scales, n_times = trimmed.shape
    ws        = max(3, KURT_WINDOW)
    positions = list(range(0, max(1, n_times - ws + 1), KURT_STEP))
    if not positions:
        positions = [0]
    K = np.zeros((n_scales, len(positions)))
    for si in range(n_scales):
        for pi, p in enumerate(positions):
            w = trimmed[si, p: p + ws]
            K[si, pi] = (stats.kurtosis(w, fisher=False, bias=False) - 3.0
                         if w.size >= 3 else 0.0)
    return K


# ─────────────────────────────────────────────────────────────────────────────
#  STEP A -- RGB SCALOGRAM   (X=R, Y=G, Z=B)  -- full CWT scale range
# ─────────────────────────────────────────────────────────────────────────────

def gen_rgb_scalogram(x_sig, y_sig, z_sig, t0, t1, vmin, vmax
                       ) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Returns (rgb_array, [mag_X, mag_Y, mag_Z]) -- mags are reused by the
    kurtogram so the CWT is never recomputed."""
    grays, mags = [], []
    for sig in (x_sig, y_sig, z_sig):
        mag = _cwt_matrix(sig, t0, t1)

        def render(ax, _mag=mag):
            ax.imshow(_mag, aspect='auto', origin='upper',
                      extent=[t0, t1, float(CWT_SCALES[-1]), float(CWT_SCALES[0])],
                      vmin=vmin, vmax=vmax, cmap=CMAP_SCALOGRAM,
                      interpolation='bilinear')
            ax.set_xlim(t0, t1)

        gray = render_to_grayscale(render)
        grays.append(gray); mags.append(mag)

    rgb_arr = np.stack(grays, axis=-1)   # R=X, G=Y, B=Z
    return rgb_arr, mags


# ─────────────────────────────────────────────────────────────────────────────
#  STEP B -- RGB SPECTROGRAM   (X=R, Y=G, Z=B)
# ─────────────────────────────────────────────────────────────────────────────

def gen_rgb_spectrogram(x_sig, y_sig, z_sig, t0, t1, vmin, vmax) -> np.ndarray:
    grays = []
    for sig in (x_sig, y_sig, z_sig):
        f, t_win, S_win = _stft_matrix(sig, t0, t1)

        def render(ax, _f=f, _t=t_win, _S=S_win):
            ax.pcolormesh(_t, _f, _S, shading='gouraud',
                          vmin=vmin, vmax=vmax, cmap=CMAP_SPECTROGRAM)
            ax.set_xlim(t0, t1)
            ax.set_ylim(0, FS / 2)

        grays.append(render_to_grayscale(render))

    return np.stack(grays, axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP C -- RGB KURTOGRAM   (X=R, Y=G, Z=B)  -- reuses CWT mags from Step A
#  FIX A: computed/rendered only over scales[KURT_MIN_SCALE_IDX:].
#  Colour Option 2: rendered ONCE per axis using CMAP_KURTOGRAM_MODEL
#  (cividis, monotonic luminance -- sign-preserving through grayscale).
# ─────────────────────────────────────────────────────────────────────────────

def gen_rgb_kurtogram(mags: List[np.ndarray], t0, t1, vmin, vmax) -> np.ndarray:
    grays = []
    y_top    = float(CWT_SCALES[-1])
    y_bottom = float(CWT_SCALES[KURT_MIN_SCALE_IDX])   # FIX A -- trimmed extent

    for mag in mags:
        K = _kurtogram_matrix(mag)

        def render(ax, _K=K):
            ax.imshow(_K, aspect='auto', origin='upper',   # Bug-7 fix
                      extent=[t0, t1, y_top, y_bottom],
                      vmin=vmin, vmax=vmax, cmap=CMAP_KURTOGRAM_MODEL,
                      interpolation='bilinear')
            ax.set_xlim(t0, t1)   # Bug-6 fix -- stretch to exactly t1

        grays.append(render_to_grayscale(render))

    return np.stack(grays, axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
#  MOSAIC ASSEMBLY -- generic over any row/col subset (FIX C)
# ─────────────────────────────────────────────────────────────────────────────

def build_mosaic(rgb_by_device_rep: Dict[str, Dict[str, np.ndarray]],
                  row_devices: List[str], col_reps: List[str],
                  variant_name: str) -> Image.Image:
    """
    Assemble one mosaic variant for one window. Shape is entirely driven by
    row_devices x col_reps -- canvas size and cell placement adapt
    automatically to whatever subset of devices/representations is passed.

    Each cached representation array is BASE_CELL_SIZE (224x224), computed
    once per window and shared across all variants. Here, each cell is
    resized (LANCZOS) from BASE_CELL_SIZE down to this specific variant's
    target size (RESIZE_TO[variant_name]) before pasting -- this is the only
    per-variant-specific step; the expensive CWT/STFT/kurtosis + matplotlib
    render is never repeated. 8px neutral gutters separate all cells.

    rgb_by_device_rep: {device: {representation: uint8 (224,224,3) array}}
    """
    cell_w, cell_h = RESIZE_TO[variant_name]
    g = MOSAIC_GUTTER_PX
    n_rows = len(row_devices)
    n_cols = len(col_reps)

    canvas_w = n_cols * cell_w + (n_cols - 1) * g
    canvas_h = n_rows * cell_h + (n_rows - 1) * g
    canvas = Image.new('RGB', (canvas_w, canvas_h), color=MOSAIC_GUTTER_COLOR)

    for r, device in enumerate(row_devices):
        for c, rep in enumerate(col_reps):
            arr = rgb_by_device_rep.get(device, {}).get(rep)
            if arr is None:
                continue
            cell_img = Image.fromarray(arr, mode='RGB')
            if cell_img.size != (cell_w, cell_h):
                cell_img = cell_img.resize((cell_w, cell_h), Image.LANCZOS)
            x = c * (cell_w + g)
            y = r * (cell_h + g)
            canvas.paste(cell_img, (x, y))

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
#  PER-DEVICE, PER-WINDOW PROCESSING  (computes all 3 reps ONCE per device)
# ─────────────────────────────────────────────────────────────────────────────

def process_device_window(
        device: str, x_full, y_full, z_full, t0: float, t1: float,
        bounds: Dict[str, Tuple],
        output_root: Path, stem: str) -> Dict[str, np.ndarray]:
    """
    Generate the 3 individual RGB representation arrays for ONE device at
    ONE window, IN MEMORY ONLY. These are reused by every mosaic variant --
    the CWT/STFT/kurtosis transforms are never recomputed per variant.
    """
    sc_rgb, mags = gen_rgb_scalogram(
        x_full, y_full, z_full, t0, t1, *bounds['scalogram'])
    sp_rgb = gen_rgb_spectrogram(
        x_full, y_full, z_full, t0, t1, *bounds['spectrogram'])
    ku_rgb = gen_rgb_kurtogram(
        mags, t0, t1, *bounds['kurtogram'])

    if SAVE_INDIVIDUAL_REPRESENTATIONS:
        reps_dir = output_root / REPRESENTATIONS_DIRNAME / device
        for name, arr in (('scalogram', sc_rgb), ('spectrogram', sp_rgb), ('kurtogram', ku_rgb)):
            d = reps_dir / name
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr, mode='RGB').save(d / f"{stem}.png")

    return {'scalogram': sc_rgb, 'spectrogram': sp_rgb, 'kurtogram': ku_rgb}


# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL COLOUR BOUNDS -- ADL-only sampling
# ─────────────────────────────────────────────────────────────────────────────

def compute_global_bounds(base_dir: Path, force_scan: bool = False) -> None:
    """
    Scan ADL-only files and compute robust vmin/vmax for scalogram,
    spectrogram, and kurtogram.

    IMPORTANT: only scans MOSAIC_ROW_DEVICES (currently Acc1+Gyro) -- NOT
    every key in DEVICES. Acc2 is intentionally excluded from the mosaics,
    so it must also be excluded here; otherwise its signal statistics would
    still skew the percentile bounds even though Acc2 never appears in any
    generated image.

    force_scan=True ignores USE_HARDCODED_BOUNDS and always performs the
    real scan -- this is what --bounds-only uses to get fresh numbers.
    """
    global GLOBAL_BOUNDS

    if USE_HARDCODED_BOUNDS and not force_scan:
        GLOBAL_BOUNDS = dict(HARDCODED_BOUNDS)
        print("[INFO] Using hardcoded bounds:", HARDCODED_BOUNDS)
        return

    print(f"[INFO] Scanning Daily Living files for global colour bounds "
        f"(devices={MOSAIC_ROW_DEVICES}) ...")

    sv, pv, kv = [], [], []

    raw_dir = base_dir / RAW_DATA_DIR
    adl_dir = raw_dir / 'Daily Living'

    adls = sorted(adl_dir.rglob('*.txt'))

    if not adls:
        print("[WARNING] No Daily Living files found for global bounds.")
        return

    for fpath in adls:
        try:
            df = read_df(fpath)

            if df.shape[1] < MIN_COLS:
                continue

            for device_name in MOSAIC_ROW_DEVICES:
                cols = DEVICES[device_name]

                if max(cols) >= df.shape[1]:
                    continue

                for ci in cols:
                    raw = pd.to_numeric(
                        df.iloc[:, ci],
                        errors='coerce'
                    ).dropna().values

                    if raw.size < 256:
                        continue

                    s = normalize(raw)
                    coeffs, _ = pywt.cwt(s, CWT_SCALES, CWT_WAVELET,
                                            sampling_period=1.0 / FS)
                    mag = np.abs(coeffs)
                    sv.append((float(np.nanpercentile(mag, 1)),
                                float(np.nanpercentile(mag, 99))))

                    _, _, Sxx = sps.spectrogram(
                        s, fs=FS, window='hann', nperseg=STFT_NPERSEG,
                        noverlap=STFT_NOVERLAP, nfft=STFT_NFFT, scaling='spectrum')
                    Sxx_db = 10.0 * np.log10(Sxx + 1e-12)
                    pv.append((float(np.nanpercentile(Sxx_db, 1)),
                                float(np.nanpercentile(Sxx_db, 99))))

                    # FIX A: kurtogram bounds computed on the SAME
                    # trimmed scale range that will actually be rendered.
                    K = _kurtogram_matrix(mag)
                    kv.append((float(np.nanpercentile(K, 1)),
                                float(np.nanpercentile(K, 99))))
        except Exception as e:
            warnings.warn(f"[BOUNDS] {fpath.name}: {e}")

    def agg(pairs, lo=2, hi=98):
        if not pairs:
            return (None, None)
        return (float(np.percentile([p[0] for p in pairs], lo)),
                float(np.percentile([p[1] for p in pairs], hi)))

    s_lo, s_hi = agg(sv)
    sp_lo, sp_hi = agg(pv)
    k_lo, k_hi = agg(kv)

    if k_lo is not None and k_hi is not None:
        k_abs = max(abs(k_lo), abs(k_hi))
        k_lo, k_hi = -k_abs, k_abs

    GLOBAL_BOUNDS = {
        'scalogram':   (s_lo, s_hi),
        'spectrogram': (sp_lo, sp_hi),
        'kurtogram':   (k_lo, k_hi),
    }
    print(f"  scalogram   : vmin={s_lo}  vmax={s_hi}")
    print(f"  spectrogram : vmin={sp_lo}  vmax={sp_hi}")
    print(f"  kurtogram   : vmin={k_lo}  vmax={k_hi}   "
          f"(scales[{KURT_MIN_SCALE_IDX}:] only -- FIX A)")
    print("[TIP] Copy above into HARDCODED_BOUNDS and set "
          "USE_HARDCODED_BOUNDS=True to skip this scan next time.")
    print("[NOTE] These bounds were computed from ADL files ONLY.")


# ─────────────────────────────────────────────────────────────────────────────
#  FILE PROCESSOR -- builds all devices x representations ONCE, then emits
#  every configured mosaic variant from that same in-memory result (FIX C).
# ─────────────────────────────────────────────────────────────────────────────

def process_file(file_path: Path, output_root: Path) -> None:
    fname = file_path.name
    base = file_path.stem
    # Expected structure:
    # SisFall_dataset/
    #   Daily Living/Activity/SAxx/raw.txt
    #   Fall/Activity/SAxx/raw.txt
    subject = file_path.parent.name
    activity = file_path.parent.parent.name
    cls = file_path.parent.parent.parent.name

    is_fall = cls.lower() == 'fall'

    try:
        df = read_df(file_path)
    except Exception as e:
        print(f"[ERROR] Read {file_path}: {e}"); return

    if df.shape[1] < MIN_COLS:
        print(f"[SKIP]  {file_path.name}: only {df.shape[1]} columns"); return

    segments = build_segments(len(df))
    if not segments:
        print(f"[SKIP]  {file_path.name}: no full-length segments"); return

    # ── normalise FULL signal once per axis per device (Step 1 / Bug-1) ───────
    # Only MOSAIC_ROW_DEVICES (Acc1+Gyro) -- NOT all of DEVICES. Acc2 is
    # excluded from every mosaic (build_mosaic never reads it), so computing
    # its scalogram/spectrogram/kurtogram here would render 9 matplotlib
    # figures per window (3 axes x 3 reps) for absolutely nothing.
    device_signals: Dict[str, Dict[str, np.ndarray]] = {}
    for device in MOSAIC_ROW_DEVICES:
        cols = DEVICES[device]
        if max(cols) >= df.shape[1]:
            continue
        full_norm: Dict[str, np.ndarray] = {}
        for i, axis in enumerate('XYZ'):
            raw = pd.to_numeric(df.iloc[:, cols[i]], errors='coerce').values
            raw = raw[np.isfinite(raw)]
            if raw.size < 8:
                continue
            full_norm[axis] = normalize(raw)
        if len(full_norm) < 3:
            print(f"[SKIP]  {base} {device}: fewer than 3 valid axes"); continue
        device_signals[device] = full_norm

    if not device_signals:
        print(f"[SKIP]  {base}: no usable devices"); return

    # One output folder per mosaic variant per class (FIX C)
    variant_folders: Dict[str, Path] = {}
    for variant_name in MOSAIC_VARIANTS:
        folder = output_root / variant_name / cls / activity / subject
        folder.mkdir(parents=True, exist_ok=True)
        variant_folders[variant_name] = folder
        folder.mkdir(parents=True, exist_ok=True)
        variant_folders[variant_name] = folder

    for t0, t1 in segments:
        ok = True
        for device, sig in device_signals.items():
            i0 = max(0, int(np.floor(t0 * FS)))
            i1 = min(len(sig['X']), int(np.ceil(t1 * FS)))
            if (i1 - i0) < 8:
                ok = False
                break
        if not ok:
            continue

        rgb_by_device_rep: Dict[str, Dict[str, np.ndarray]] = {}

        for device, sig in device_signals.items():
            ts   = f"_T{t0:.1f}-{t1:.1f}s" if len(segments) > 1 else ""
            stem = f"{base}_{device}{ts}"
            try:
                reps = process_device_window(
                    device=device,
                    x_full=sig['X'], y_full=sig['Y'], z_full=sig['Z'],
                    t0=t0, t1=t1,
                    bounds=GLOBAL_BOUNDS,
                    output_root=output_root, stem=stem)
                rgb_by_device_rep[device] = reps
            except Exception as e:
                print(f"[ERROR] {stem}: {e}")
                traceback.print_exc()

        if not rgb_by_device_rep:
            continue

        # ── emit every configured mosaic variant from this SAME window's
        #    already-computed representations -- no recomputation (FIX C) ──
        mts = f"_T{t0:.1f}-{t1:.1f}s" if len(segments) > 1 else ""
        mosaic_stem = f"{base}{mts}"
        for variant_name, col_reps in MOSAIC_VARIANTS.items():
            mosaic_img = build_mosaic(rgb_by_device_rep, MOSAIC_ROW_DEVICES, col_reps,
                                       variant_name)
            mosaic_img.save(variant_folders[variant_name] / f"{mosaic_stem}.png")

    print(f"[OK]   {base}")


# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(base_dir: Path) -> None:
    base_dir = base_dir.resolve()
    out_root = base_dir / OUTPUT_DIR
    out_root.mkdir(parents=True, exist_ok=True)

    raw_dir = base_dir / RAW_DATA_DIR
    if not raw_dir.exists():
        print(f"[ERROR] Raw data directory not found: {raw_dir}"); return

    # Find all raw .txt files recursively.
    # Expected:
    # Daily Living / Activity / SAxx / file.txt
    # Fall / Activity / SAxx / file.txt

    raw_files = sorted(raw_dir.rglob('*.txt'))

    if not raw_files:
        print(f"[ERROR] No .txt files found in {raw_dir}")
        return

    print(f"Found {len(raw_files)} raw file(s).  Output: {out_root}\n")

    compute_global_bounds(base_dir)
    print()

    total = 0

    for fpath in raw_files:
        print(f"  Processing: {fpath}")

        try:
            process_file(fpath, out_root)
            total += 1
        except Exception as e:
            print(f"[ERROR] {fpath.name}: {e}")
            traceback.print_exc()

    print(f"\n  Done.  Processed {total} file(s).  Output: {out_root}")
    for variant_name in MOSAIC_VARIANTS:
        print(f"  {variant_name} -> {out_root / variant_name}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    base = DEFAULT_BASE_DIR
    bounds_only = '--bounds-only' in sys.argv
    positional_args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if positional_args:
        arg = Path(positional_args[0])
        if arg.is_dir():
            base = arg

    print("=" * 78)
    print("  SisFall  ->  RGB Scalogram + Spectrogram + Kurtogram  ->  Dual Mosaic")
    print("=" * 78)
    print(f"  Base directory  : {base}")
    print(f"  Devices (mosaic): {', '.join(MOSAIC_ROW_DEVICES)}   "
          f"(Acc2 excluded -- redundant with Acc1)")
    print(f"  Scalogram       : Morlet CWT, scales 1-{CWT_SCALES[-1]}, cmap={CMAP_SCALOGRAM}")
    print(f"  Spectrogram     : STFT nperseg={STFT_NPERSEG}, cmap={CMAP_SPECTROGRAM}")
    print(f"  Kurtogram       : window={KURT_WINDOW}, step={KURT_STEP}, "
          f"cmap={CMAP_KURTOGRAM_MODEL}, scales[{KURT_MIN_SCALE_IDX}:] only (FIX A)")
    print(f"  Channel map     : (each image) X=R | Y=G | Z=B")

    if bounds_only:
        print()
        print("  Mode            : --bounds-only (scan + print vmin/vmax, no images written)")
        print()
        raw_dir = base.resolve() / RAW_DATA_DIR
        if not raw_dir.exists():
            print(f"[ERROR] Raw data directory not found: {raw_dir}")
        else:
            compute_global_bounds(base.resolve(), force_scan=True)
            print()
            print("[NEXT STEP] Copy the three tuples above into HARDCODED_BOUNDS,")
            print("            set USE_HARDCODED_BOUNDS = True, then re-run WITHOUT")
            print("            --bounds-only to generate the full image set using")
            print("            these fixed bounds (fast -- no re-scan needed).")
        sys.exit(0)

    print(f"  Mosaic variants :")
    for name, cols in MOSAIC_VARIANTS.items():
        print(f"      {name:<22s} rows={MOSAIC_ROW_DEVICES}  cols={cols}")
    print(f"  Mosaic gutter   : {MOSAIC_GUTTER_PX}px neutral")
    print(f"  Global bounds   : ADL-only sampling, "
          f"hardcoded={'ON' if USE_HARDCODED_BOUNDS else 'OFF'}")
    print(f"  Individual reps saved to disk : {SAVE_INDIVIDUAL_REPRESENTATIONS}")
    print(f"  Debug images    : REMOVED (not generated in this version)")
    print(f"  Sampling rate   : {FS} Hz")
    print(f"  Window          : {SPLIT_WINDOW_S}s  |  overlap {SPLIT_OVERLAP*100:.0f}%")
    print(f"  Base cell cache : {BASE_CELL_SIZE[0]}x{BASE_CELL_SIZE[1]} px (shared, computed once per window)")
    for variant_name, size in RESIZE_TO.items():
        canvas_w = 2 * size[0] + MOSAIC_GUTTER_PX
        canvas_h = len(MOSAIC_ROW_DEVICES) * size[1] + (len(MOSAIC_ROW_DEVICES) - 1) * MOSAIC_GUTTER_PX
        print(f"  Output size     : {variant_name:<22s} cell={size[0]}x{size[1]} px "
              f"-> canvas={canvas_w}x{canvas_h} px  (LANCZOS)")
    print()

    try:
        run_pipeline(base)
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
