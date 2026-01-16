import os
import re
import numpy as np
import matplotlib
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq

try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# USER CONFIG
# =========================
set_name = "Rest1"
set_folder = "Rest"

T0, T1 = 859, 983  # crop interval (seconds in the same timestamp units as file)
txt_file_path = "/Users/anshshetty/Library/Mobile Documents/com~apple~CloudDocs/ExoHand/Adhi Data/Rest/Rest1.txt"

out_dir = f"/Users/anshshetty/Library/Mobile Documents/com~apple~CloudDocs/ExoHand/Adhi Data/{set_folder}"
os.makedirs(out_dir, exist_ok=True)

low = 20.0
high = 450.0
smoothing_window_ms = 30

window_length = 0.1    # 100 ms
step_length   = 0.025  # 25 ms


# =========================
# HELPERS
# =========================
float_pat = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def extract_floats(line: str):
    """Return list of floats found in the line (robust to weird spacing/characters)."""
    return [float(x) for x in float_pat.findall(line)]

def load_two_channel_txt(path):
    """
    Robust loader for:
      - header/meta lines like '81'
      - data lines like: t ch1 ch2
      - or: t something ch1 ch2
    Strategy: time = first float, ch1/ch2 = last two floats.
    """
    times = []
    ch1 = []
    ch2 = []

    with open(path, "r") as f:
        for line in f:
            nums = extract_floats(line)
            if len(nums) < 3:
                # e.g. '81' or blank or junk -> ignore
                continue
            t = nums[0]
            a1 = nums[-2]
            a2 = nums[-1]
            times.append(t)
            ch1.append(a1)
            ch2.append(a2)

    times = np.asarray(times, dtype=float)
    ch1 = np.asarray(ch1, dtype=float)
    ch2 = np.asarray(ch2, dtype=float)

    if times.size < 2:
        raise ValueError("Not enough samples parsed. Check file format/path.")

    return times, ch1, ch2

def adc_to_volts(x):
    """
    Heuristic scaling:
      - if values look like 10-bit ADC counts -> convert to 0..5V
      - if already volts (<10) -> keep
      - else treat as millivolts
    """
    x = np.asarray(x, float)
    m = np.max(np.abs(x))
    if m <= 1035:
        return (x / 1023.0) * 5.0
    elif m < 10:
        return x.copy()
    else:
        return x / 1000.0

def despike_linear(x, z_thresh=6.0, win=0.2, fs=1000.0):
    """
    Sliding-window median/MAD spike detector, replace spikes by linear interpolation.
    Returns (x_clean, spike_mask).
    """
    x = np.asarray(x, float)
    n = x.size
    if n < 5:
        return x, np.zeros(n, dtype=bool)

    k = max(3, int(win * fs) | 1)  # odd
    half = k // 2

    med = np.empty(n, dtype=float)
    mad = np.empty(n, dtype=float)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        w = x[lo:hi]
        m = np.median(w)
        med[i] = m
        mad[i] = np.median(np.abs(w - m)) + 1e-12

    z = 0.6745 * (x - med) / mad
    bad = np.abs(z) > z_thresh

    if np.any(bad):
        good_idx = np.where(~bad)[0]
        bad_idx  = np.where(bad)[0]
        x_clean = x.copy()
        if good_idx.size > 1:
            x_clean[bad_idx] = np.interp(bad_idx, good_idx, x[good_idx])
            return x_clean, bad

    return x, bad

def preprocess_channel(x_adc, fs, low=20.0, high=450.0, smooth_ms=30):
    """
    Scale -> despike -> bandpass -> rectify -> moving-average envelope
    Returns: (raw_volts, filtered, envelope, spike_mask)
    """
    v = adc_to_volts(x_adc)

    v, spike_mask = despike_linear(v, fs=fs)

    nyq = fs / 2.0
    high_eff = min(high, 0.95 * nyq)
    low_eff  = max(5.0, min(low, 0.5 * high_eff))

    if low_eff >= high_eff or high_eff <= 10:
        vf = v.copy()
    else:
        b, a = butter(4, [low_eff/nyq, high_eff/nyq], btype="band")
        vf = filtfilt(b, a, v)

    rect = np.abs(vf)
    win = max(1, int(smooth_ms * fs / 1000.0))
    env = np.convolve(rect, np.ones(win)/win, mode="same")

    return v, vf, env, spike_mask


# =========================
# LOAD
# =========================
raw_time, adc1_all, adc2_all = load_two_channel_txt(txt_file_path)
print("Total samples parsed:", raw_time.size)
print("raw_time range:", float(raw_time.min()), "to", float(raw_time.max()))

# =========================
# CROP
# =========================
mask = (raw_time >= T0) & (raw_time <= T1)
if not np.any(mask):
    raise ValueError("No samples in [T0, T1]. Check T0/T1 or timestamp units.")

t_seg  = raw_time[mask]
x1_seg = adc1_all[mask]
x2_seg = adc2_all[mask]

# Enforce strictly increasing timestamps (dedupe)
t_unique, keep_idx = np.unique(t_seg, return_index=True)
t_seg  = t_unique
x1_seg = x1_seg[keep_idx]
x2_seg = x2_seg[keep_idx]

if t_seg.size < 2:
    raise ValueError("Too few samples after cropping/dedup.")

duration = float(t_seg[-1] - t_seg[0])
if duration <= 0:
    raise ValueError("Non-increasing timestamps after dedup.")

# Estimate sampling frequency robustly
dt = np.diff(t_seg)
dt_pos = dt[dt > 0]
fs_med  = 1.0 / np.median(dt_pos) if dt_pos.size else np.nan
fs_span = (t_seg.size - 1) / duration
fs = fs_med if (np.isfinite(fs_med) and fs_med > 0) else fs_span
fs = float(np.clip(fs, 100.0, 5000.0))  # clamp to sane EMG Fs

time = t_seg - t_seg[0]  # start at 0
print(f"Cropped: span={time[-1]:.2f}s, N={time.size}, Fs≈{fs:.2f} Hz")


# =========================
# PREPROCESS BOTH CHANNELS
# =========================
v1, f1, env1, spikes1 = preprocess_channel(x1_seg, fs, low=low, high=high, smooth_ms=smoothing_window_ms)
v2, f2, env2, spikes2 = preprocess_channel(x2_seg, fs, low=low, high=high, smooth_ms=smoothing_window_ms)

print("Removed spikes ch1:", int(spikes1.sum()), "ch2:", int(spikes2.sum()))
print("Max |V| ch1:", float(np.max(np.abs(v1))), "ch2:", float(np.max(np.abs(v2))))


# =========================
# FEATURE EXTRACTION (WINDOWED)
# =========================
window_size = max(1, int(window_length * fs))
step_size   = max(1, int(step_length   * fs))

features = []

for start in range(0, len(f1) - window_size + 1, step_size):
    w1 = f1[start:start+window_size]
    w2 = f2[start:start+window_size]
    Nw = len(w1)

    # channel 1 time-domain
    rms1 = np.sqrt(np.mean(w1**2))
    mav1 = np.mean(np.abs(w1))
    var1 = np.var(w1)
    wl1  = np.sum(np.abs(np.diff(w1)))
    max1 = np.max(np.abs(w1))

    thr1 = 0.01 * np.max(np.abs(w1)) + 1e-12
    zc1  = np.sum(((w1[:-1]*w1[1:]) < 0) & (np.abs(w1[:-1]-w1[1:]) > thr1))
    ssc1 = np.sum((((w1[1:-1]-w1[:-2]) * (w1[1:-1]-w1[2:])) > 0) &
                  (np.abs(w1[1:-1]-w1[:-2]) > thr1) &
                  (np.abs(w1[1:-1]-w1[2:]) > thr1))
    wamp_thr1 = 0.01 * np.max(np.abs(w1)) + 1e-12
    wamp1 = np.sum(np.abs(np.diff(w1)) > wamp_thr1)
    iemg1 = np.sum(np.abs(w1))

    # channel 1 freq-domain
    fft1 = np.abs(rfft(w1 * np.hanning(Nw)))
    frq  = rfftfreq(Nw, 1/fs)
    psd1 = fft1**2
    if np.sum(psd1) > 0:
        meanf1 = float(np.sum(frq * psd1) / np.sum(psd1))
        cdf1 = np.cumsum(psd1)
        medf1 = float(frq[np.searchsorted(cdf1, 0.5 * cdf1[-1])])
    else:
        meanf1 = 0.0
        medf1  = 0.0

    e1 = env1[start:start+window_size]
    env_mean1 = float(np.mean(e1))
    env_max1  = float(np.max(e1))

    # channel 2 time-domain
    rms2 = np.sqrt(np.mean(w2**2))
    mav2 = np.mean(np.abs(w2))
    var2 = np.var(w2)
    wl2  = np.sum(np.abs(np.diff(w2)))
    max2 = np.max(np.abs(w2))

    thr2 = 0.01 * np.max(np.abs(w2)) + 1e-12
    zc2  = np.sum(((w2[:-1]*w2[1:]) < 0) & (np.abs(w2[:-1]-w2[1:]) > thr2))
    ssc2 = np.sum((((w2[1:-1]-w2[:-2]) * (w2[1:-1]-w2[2:])) > 0) &
                  (np.abs(w2[1:-1]-w2[:-2]) > thr2) &
                  (np.abs(w2[1:-1]-w2[2:]) > thr2))
    wamp_thr2 = 0.01 * np.max(np.abs(w2)) + 1e-12
    wamp2 = np.sum(np.abs(np.diff(w2)) > wamp_thr2)
    iemg2 = np.sum(np.abs(w2))

    # channel 2 freq-domain
    fft2 = np.abs(rfft(w2 * np.hanning(Nw)))
    psd2 = fft2**2
    if np.sum(psd2) > 0:
        meanf2 = float(np.sum(frq * psd2) / np.sum(psd2))
        cdf2 = np.cumsum(psd2)
        medf2 = float(frq[np.searchsorted(cdf2, 0.5 * cdf2[-1])])
    else:
        meanf2 = 0.0
        medf2  = 0.0

    e2 = env2[start:start+window_size]
    env_mean2 = float(np.mean(e2))
    env_max2  = float(np.max(e2))

    # cross-channel features (often helpful for classification)
    env_ratio = env_mean1 / (env_mean2 + 1e-12)
    corr = np.corrcoef(w1, w2)[0, 1] if (np.std(w1) > 1e-12 and np.std(w2) > 1e-12) else 0.0

    # timestamps
    t_rel = float(time[start] + 0.5 * window_length)
    t_abs = float(t_seg[0] + t_rel)

    features.append([
        t_rel, t_abs,

        rms1, mav1, var1, wl1, max1, zc1, ssc1, wamp1, iemg1, meanf1, medf1, env_mean1, env_max1,
        rms2, mav2, var2, wl2, max2, zc2, ssc2, wamp2, iemg2, meanf2, medf2, env_mean2, env_max2,

        env_ratio, corr
    ])

features = np.asarray(features, dtype=np.float64)
print("Feature matrix shape:", features.shape)
print("First row:", features[0])


# =========================
# SAVE CSV
# =========================
out_csv = os.path.join(out_dir, f"emg_features_2ch_{set_name}_.csv")
header = (
    "t_rel_s,t_abs_s,"
    "rms_ch1,mav_ch1,var_ch1,wl_ch1,maxamp_ch1,zc_ch1,ssc_ch1,wamp_ch1,iemg_ch1,mean_freq_ch1,median_freq_ch1,env_mean_ch1,env_max_ch1,"
    "rms_ch2,mav_ch2,var_ch2,wl_ch2,maxamp_ch2,zc_ch2,ssc_ch2,wamp_ch2,iemg_ch2,mean_freq_ch2,median_freq_ch2,env_mean_ch2,env_max_ch2,"
    "env_ratio_ch1_over_ch2,corr_ch1_ch2"
)
np.savetxt(out_csv, features, delimiter=",", header=header, comments="")
print("Saved features ->", out_csv)


# =========================
# PLOT & SAVE PNG
# =========================
plt.figure(figsize=(12, 6))
plt.plot(time, v1,   label="Raw ch1 (V)", alpha=0.25)
plt.plot(time, f1,   label="Filtered ch1", alpha=0.55)
plt.plot(time, env1, label=f"Envelope ch1 (~{smoothing_window_ms} ms)", linewidth=2)

plt.plot(time, v2,   label="Raw ch2 (V)", alpha=0.25)
plt.plot(time, f2,   label="Filtered ch2", alpha=0.55)
plt.plot(time, env2, label=f"Envelope ch2 (~{smoothing_window_ms} ms)", linewidth=2)

plt.xlabel("Time (s)")
plt.ylabel("EMG (V)")
plt.grid(True)
plt.legend()

out_png = os.path.join(out_dir, f"emg_plot_2ch_{set_name}.png")
plt.savefig(out_png, dpi=180, bbox_inches="tight")
print("Saved plot ->", out_png)

plt.show()
