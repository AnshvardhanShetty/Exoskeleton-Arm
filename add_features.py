import argparse, sys, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd

# Base columns already in your CSV (adjust if needed)
BASE_FEATS = [
    "rms","mav","var","wl","maxamp","zc","ssc","wamp","iemg",
    "mean_freq","median_freq","env_mean","env_max"
]

def _canon_label(s):
    return unicodedata.normalize("NFKC", str(s)).strip().lower()

def add_temporal_features(df: pd.DataFrame,
                          base_feats=BASE_FEATS,
                          time_col=None,
                          lags=(1,2,3),
                          roll_windows=(3,5,9),
                          ema_spans=(3,5,9)) -> pd.DataFrame:
    df = df.copy()

    # Canonicalise labels (defensive)
    if "label" in df.columns:
        df["label"] = df["label"].map(_canon_label)

    # Choose time column
    if time_col is None:
        time_col = "t_abs_s" if "t_abs_s" in df.columns else ("t_rel_s" if "t_rel_s" in df.columns else None)

    # Grouping: prefer explicit ids; else infer from t_rel_s resets
    grp_cols = [c for c in ["file","trial","set","recording_id"] if c in df.columns]
    if not grp_cols:
        if "t_rel_s" in df.columns:
            df["__seq__"] = (df["t_rel_s"].diff().fillna(0) < 0).cumsum()
        else:
            df["__seq__"] = 0
        grp_cols = ["__seq__"]

    # Sort within group by time so shifts/rolling are meaningful
    if time_col in df.columns:
        df = df.sort_values(grp_cols + [time_col]).reset_index(drop=True)
    else:
        df = df.sort_values(grp_cols).reset_index(drop=True)

    g = df.groupby(grp_cols, group_keys=False)

    # 1) Lags (past-only context)
    for f in base_feats:
        if f in df.columns:
            for L in lags:
                df[f"{f}_lag{L}"] = g[f].shift(L)

    # 2) Deltas / velocity of change
    for f in base_feats:
        if f in df.columns:
            df[f"{f}_d1"] = g[f].diff(1)
            df[f"{f}_d2"] = g[f].diff(2)
            df[f"{f}_pct"] = g[f].pct_change().replace([np.inf, -np.inf], np.nan)

    # 3) Rolling stats (captures plateau vs burst)
    for f in base_feats:
        if f in df.columns:
            for w in roll_windows:
                m = g[f].rolling(w, min_periods=max(1, w//2)).mean().reset_index(level=grp_cols, drop=True)
                s = g[f].rolling(w, min_periods=max(1, w//2)).std().reset_index(level=grp_cols, drop=True)
                df[f"{f}_roll{w}_mean"] = m
                df[f"{f}_roll{w}_std"]  = s

    # 4) EMA (smoother level)
    for f in base_feats:
        if f in df.columns:
            for span in ema_spans:
                df[f"{f}_ema{span}"] = g[f].apply(lambda s: s.ewm(span=span, adjust=False).mean())

    # 5) Plateau score: level / variability
    for f in ["env_mean","env_max","rms","mav"]:
        if f in df.columns:
            for w in roll_windows:
                lvl = df.get(f"{f}_roll{w}_mean")
                vol = df.get(f"{f}_roll{w}_std")
                if lvl is not None and vol is not None:
                    df[f"{f}_plateau{w}"] = lvl / (1e-9 + vol)

    return df

train_df = pd.read_csv("/Users/anshshetty/Desktop/Exoskeleton Arm /max_extend/emg_feature_efs_extend_max.csv")


train_df = add_temporal_features(train_df)


train_df.to_csv("/Users/anshshetty/Desktop/Exoskeleton Arm /max_extend/emg_feature_efs_extend_max.csv", index=False)

#df = pd.read_csv("/Users/anshshetty/Desktop/Exoskeleton Arm /max_extend/emg_feature_efs_extend_max.csv")

# remove columns with temporal-engineered patterns
#label_idx = df.columns.get_loc("label")

# Keep everything up to and including 'label'
#df_clean = df.iloc[:, :label_idx+1]


#df_clean.to_csv("/Users/anshshetty/Desktop/Exoskeleton Arm /extend_test/set 2/emg_feature_efs_extend_set 2_cleaned.csv", index=False)
#print("Saved:", df_clean.shape)