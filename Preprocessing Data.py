import matplotlib
import os 
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq

try:
    matplotlib.use("TkAgg")   # correct casing
except Exception:
    matplotlib.use("Agg") # allows for the plot to be visualised
import matplotlib.pyplot as plt  

# --- load ---
motion = "extend"
set_folder = "rest"
set_name = "rest"
T0, T1 = 2874.62, 2952.11
txt_file_path = f"/Users/anshshetty/Desktop/Exoskeleton Arm /testing data/{set_folder} /{set_name}.txt"
with open(txt_file_path, "r") as file:
    lines = file.readlines()
data_lines = []
for line in lines:
    parts = line.strip().split()
    if len(parts) == 2:
        try:
            float(parts[0]); float(parts[1])
            data_lines.append(parts)
        except:
            continue

adc_values = [float(pair[1]) for pair in data_lines]

print("No. of samples...", len(adc_values))

raw_time   = [float(p[0]) for p in data_lines]
adc_values = [float(p[1]) for p in data_lines]

print("Total samples in file:", len(adc_values))

# --- crop to screen recording interval ---
# seconds
raw_time = np.asarray(raw_time, dtype=float)
adc_all  = np.asarray(adc_values, dtype=float)

# --- crop to screen recording interval (between T0 and T1, in seconds) ---
mask = (raw_time >= T0) & (raw_time <= T1)   # Boolean mask: True if timestamp is within [T0, T1]
assert np.any(mask), "No samples in the selected interval."  # stop if nothing falls inside

raw_time_seg = raw_time[mask]    # keep only the cropped times
adc_array    = adc_all[mask]     # keep only the cropped ADC values

# --- rebuild uniform time axis for the cropped segment ---
t_seg = np.asarray(raw_time_seg, float)   # copy cropped times into t_seg
x_seg = np.asarray(adc_array,   float)   # copy cropped ADC values into x_seg


# (optional sanity)
print("T0,T1:", T0, T1)
print("raw_time_seg[0],[-1]:", float(t_seg[0]), float(t_seg[-1]))
print("Expected span:", T1 - T0)

# enforce strictly increasing timestamps
t_unique, keep_idx = np.unique(t_seg, return_index=True)
t_seg = t_unique
x_seg = x_seg[keep_idx]   # keep samples aligned

if t_seg.size < 2:
    raise ValueError("Too few samples after cropping/dedup.")
duration = float(t_seg[-1] - t_seg[0])
if duration <= 0:
    raise ValueError("Non-increasing timestamps; check T0/T1 and file.")

dt = np.diff(t_seg)
dt_pos = dt[dt > 0]                              # ignore zero-dt duplicates
fs_med  = 1.0 / np.median(dt_pos) if dt_pos.size else np.nan
fs_span = (t_seg.size - 1) / duration            # average rate from span
samplefreq = fs_med if (np.isfinite(fs_med) and fs_med > 0) else fs_span
samplefreq = float(np.clip(samplefreq, 100.0, 5000.0))   # clamp to sane EMG Fs

time = t_seg - t_seg[0]                          # x-axis starting at 0 s
print(f"Span: {time[-1]:.2f} s, N={time.size}, Fs≈{samplefreq:.2f} Hz")
# --- volts from CROPPED adc ---
adc_max = np.max(np.abs(x_seg))

if adc_max <= 1035:   # raw ADC counts
    real_voltage = (x_seg / 1023.0) * 5.0
elif adc_max < 10:    # already in volts
    real_voltage = x_seg.copy()
else:                 # probably millivolts
    real_voltage = x_seg/ 1000.0

print("Max |V| after scaling:", np.max(np.abs(real_voltage)))

# --- despike outliers ---
def despike_linear(x, z_thresh=6.0, win=0.2, fs=1000.0): #x - EMG signal, z_thresh- how many robust standard deviations away a point must be to count as a spike. Set to 6. win - sliding window size in seconds to locate a local median + deviation. fs - sampling frequency to convert window size into number of samples.  
    x = np.asarray(x, float) #ensures the emg signal is a float numpy array 
    n = x.size #stores the total number of samples (n)
    k = max(3, int(win * fs) | 1)  # odd length >=3, when you compute the window, it ensures its symmetrical around the window
    half = k // 2 #half the window length
    med = np.copy(x) #array to hold the local median 
    mad = np.full(n, np.nan) #array to hold the local median absolute deviation
    for i in range(n): #loop through every sample in the signal
        lo = max(0, i - half) #lower index of the window
        hi = min(n, i + half + 1) #upper index of the window
        w = x[lo:hi] #extract the local window of data
        m = np.median(w) #compute the median of the window
        med[i] = m #store the median for position i
        mad[i] = np.median(np.abs(w - m)) + 1e-12  #mad[i] = the median absolute deviation (MAD) in the local window. It’s the median of |w - m|, i.e. how spread out the samples are around the median m. If all values in the window are identical (e.g. a flat segment of the signal), then np.abs(w - m) is all zeros → MAD = 0.
    z = 0.6745 * (x - med) / mad #calculate z-score, How far is each sample away from the local median, measured in MAD units?
    bad = np.abs(z) > z_thresh #the point is more than 6 x MAD away from the local median
    if np.any(bad):                              # if there are any "bad" (spike) points in the signal
        good_idx = np.where(~bad)[0]             # indices of good points (where bad == False)
        bad_idx  = np.where(bad)[0]              # indices of spike points (where bad == True)

        if good_idx.size > 1:                    # only interpolate if we have at least 2 good points
            x_clean = x.copy()                   # make a copy of the original signal to edit

            # replace the spike values with linearly interpolated values
            # np.interp takes: (x-coords to fill, known x-coords, known y-values)
            # here: fill bad_idx using values from x[good_idx] at positions good_idx
            x_clean[bad_idx] = np.interp(bad_idx, good_idx, x[good_idx])

            return x_clean, bad                  # return the cleaned signal and the spike mask

# if no spikes were found, or not enough good points for interpolation
    return x, bad                                # just return the original signal and the mask
real_voltage, spike_mask = despike_linear(real_voltage, fs=samplefreq)
print("Removed spikes:", spike_mask.sum())

# --- band-pass filter ---
low = 20.0
high = 450.0
nyquistfreq = samplefreq / 2.0
high_eff = min(high, 0.95 * nyquistfreq)
low_eff  = max(5.0, min(low, 0.5 * high_eff))

if low_eff >= high_eff or high_eff <= 10:
    # Skip bandpass if invalid for this Fs
    filtered_voltage = real_voltage.copy()
else:
    b, a = butter(4, [low_eff/nyquistfreq, high_eff/nyquistfreq], btype='band')
    filtered_voltage = filtfilt(b, a, real_voltage)

# --- rectify & smooth ---
rectified_voltage = np.abs(filtered_voltage)

# smoothing window in ms
smoothing_window_ms = 30
smoothing_window_size = max(1, int(smoothing_window_ms * samplefreq / 1000.0))

smoothed_voltage = np.convolve(
    rectified_voltage, 
    np.ones(smoothing_window_size)/smoothing_window_size, 
    mode="same"
)

# --- segmentation / feature extraction ---
# --- segmentation / feature extraction ---
window_length = 0.1    # 100 ms windows
step_length   = 0.025  # 25 ms hop
window_size   = max(1, int(window_length * samplefreq))   # guard
step_duration = max(1, int(step_length   * samplefreq))   # guard

features = []
for start in range(0, len(filtered_voltage) - window_size + 1, step_duration):
    window = filtered_voltage[start : start + window_size]
    Nw = len(window)

    rms   = np.sqrt(np.mean(window**2))
    mav   = np.mean(np.abs(window))
    var   = np.var(window)
    wl    = np.sum(np.abs(np.diff(window)))
    maxamp= np.max(np.abs(window))

    thr = 0.01 * np.max(np.abs(window))
    zc  = np.sum(((window[:-1]*window[1:]) < 0) & (np.abs(window[:-1]-window[1:]) > thr))

    ssc = np.sum((( (window[1:-1]-window[:-2]) * (window[1:-1]-window[2:]) ) > 0) & (np.abs(window[1:-1]-window[:-2]) > thr) & (np.abs(window[1:-1]-window[2:]) > thr))

    wamp_thr = 0.01 * np.max(np.abs(window))
    wamp = np.sum(np.abs(np.diff(window)) > wamp_thr)

    iemg = np.sum(np.abs(window))

    fft_vals = np.abs(rfft(window * np.hanning(Nw)))
    fft_freq = rfftfreq(Nw, 1/samplefreq)
    psd = fft_vals**2
    mean_freq = np.sum(fft_freq * psd) / np.sum(psd) if np.sum(psd) > 0 else 0.0
    if np.sum(psd) > 0:
        cdf = np.cumsum(psd)
        median_freq = fft_freq[np.searchsorted(cdf, 0.5 * cdf[-1])]
    else:
        median_freq = 0.0
    # optional envelope features
    env_window = smoothed_voltage[start : start + window_size]
    env_mean   = np.mean(env_window)
    env_max    = np.max(env_window)
    t_rel = time[start] + 0.5 * window_length   # seconds, window center on the true time axis
    t_abs = raw_time_seg[0] + t_rel  

    features.append([t_rel, t_abs, rms, mav, var, wl, maxamp, zc, ssc, wamp, iemg, mean_freq, median_freq, env_mean, env_max])

features = np.array(features, dtype=np.float64)
print(features[:5])
out_csv = f"/Users/anshshetty/Desktop/Exoskeleton Arm /testing data/{set_folder} /emg_feature_efs_{motion}_{set_name}.csv"
header = "t_rel_s,t_abs_s,rms,mav,var,wl,maxamp,zc,ssc,wamp,iemg,mean_freq,median_freq,env_mean,env_max"
np.savetxt(out_csv, features, delimiter=",", header=header, comments="")
print(f"Saved features -> {out_csv}")

# --- visualise ---
plt.figure(figsize=(12,6))
plt.plot(time, real_voltage, label="Raw (V, biased)", alpha=0.3)
plt.plot(time, filtered_voltage, label="Filtered (20–450 Hz)", alpha=0.6)
plt.plot(time, smoothed_voltage, label=f"Envelope (~{smoothing_window_ms} ms)", linewidth=2)  # make label dynamic
plt.xlabel("Time(s)")
plt.ylabel("EMG (V)")
plt.legend()
plt.grid(True)


out_png = f"/Users/anshshetty/Desktop/Exoskeleton Arm /testing data/{set_folder} /efs_{motion}_{set_name}_emg_plot.png"
plt.savefig(out_png, dpi=180, bbox_inches="tight")
print(f"Saved plot -> {out_png}")

plt.show()