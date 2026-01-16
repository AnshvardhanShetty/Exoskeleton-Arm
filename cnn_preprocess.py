# cnn_preprocess.py
import os
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt

# ==============================
# CONFIG
# ==============================
ROOT_DIR = "/Users/anshshetty/Library/Mobile Documents/com~apple~CloudDocs/ExoHand/grabmyo"

# same 4 channels you used for HGBT
CHANNELS_TO_USE = [6, 13, 0, 3]

FS = 2048.0
LOWCUT = 20.0
HIGHCUT = 450.0
FILTER_ORDER = 4

WIN_S  = 0.200   # 200 ms window
STEP_S = 0.050   # 50 ms step

# downsample each window in time to this length
T_TARGET = 128

# gesture index → intent mapping (same as intent_dataset script)
INTENT_MAP = {
    1:  "close",
    2:  "close",
    3:  "close",
    4:  "close",
    16: "close",

    5:  "open",
    6:  "open",
    7:  "open",
    8:  "open",
    9:  "open",
    10: "open",
    15: "open",

    17: "rest",
}

INTENT_TO_IDX = {"rest": 0, "close": 1, "open": 2}

OUT_NPZ = os.path.join(ROOT_DIR, "grabmyo_cnn_envelopes.npz")


# ==============================
# UTILS
# ==============================
def find_all_dat_files(root):
    paths = []
    for dp, _, files in os.walk(root):
        for f in files:
            if f.endswith(".dat"):
                paths.append(os.path.join(dp, f))
    return sorted(paths)


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)
    from scipy.signal import butter
    b, a = butter(order, [low, high], btype="band")
    return b, a


def preprocess_envelope(emg32):
    """
    Grab 4 channels, bandpass, rectify, smooth → envelope (C, N)
    """
    chs = CHANNELS_TO_USE
    sig = emg32[chs, :]   # (C, N)
    C, N = sig.shape

    env = np.zeros_like(sig)
    b, a = butter_bandpass(LOWCUT, HIGHCUT, FS, FILTER_ORDER)

    # 50 ms moving average for envelope
    win = int(0.050 * FS)
    if win < 1:
        win = 1
    kernel = np.ones(win) / win

    for c in range(C):
        x = sig[c] - np.mean(sig[c])
        xf = filtfilt(b, a, x)
        rect = np.abs(xf)
        env[c] = np.convolve(rect, kernel, mode="same")

    return env  # (C, N)


def downsample_env(env_window, target_len=T_TARGET):
    """
    env_window: (C, T_raw) → (C, target_len)
    """
    C, T = env_window.shape
    idx = np.linspace(0, T - 1, target_len).astype(int)
    return env_window[:, idx]


# ==============================
# MAIN
# ==============================
def main():
    dat_files = find_all_dat_files(ROOT_DIR)
    print(f"Found {len(dat_files)} .dat files")

    all_X = []
    all_y = []
    all_parts = []

    win = int(WIN_S * FS)
    step = int(STEP_S * FS)

    for path in dat_files:
        fname = os.path.basename(path)
        parts = fname.replace(".dat", "").split("_")

        # expected: session1_subject5_gesture3_trial7.dat
        session = parts[0]
        participant = parts[1]   # "subject5"
        gesture_num = int(parts[2].replace("gesture", ""))
        trial = int(parts[3].replace("trial", ""))

        if gesture_num not in INTENT_MAP:
            # skip gestures we don't care about (wrist, pron/sup, etc.)
            continue

        intent_label = INTENT_MAP[gesture_num]
        intent_idx = INTENT_TO_IDX[intent_label]

        rec = wfdb.rdrecord(path.replace(".dat", ""))
        emg32 = rec.p_signal.T.astype(np.float64)  # (32, N)
        env = preprocess_envelope(emg32)          # (4, N)

        C, N = env.shape
        if N < win:
            continue

        for start in range(0, N - win + 1, step):
            end = start + win
            env_win = env[:, start:end]          # (4, win)
            env_ds  = downsample_env(env_win)    # (4, T_TARGET)

            all_X.append(env_ds.astype(np.float32))
            all_y.append(intent_idx)
            all_parts.append(participant)

    X = np.stack(all_X, axis=0)       # (N_samples, 4, T_TARGET)
    y = np.array(all_y, dtype=np.int64)
    participants = np.array(all_parts)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Participants:", np.unique(participants))

    np.savez_compressed(
        OUT_NPZ,
        X=X,
        y=y,
        participant=participants,
        intent_to_idx=np.array(list(INTENT_TO_IDX.items()), dtype=object),
    )
    print("Saved →", OUT_NPZ)


if __name__ == "__main__":
    main()
