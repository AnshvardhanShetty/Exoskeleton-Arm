import pandas as pd
import numpy as np
import os
import json
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

ROOT = "/Users/anshshetty/Library/Mobile Documents/com~apple~CloudDocs/ExoHand/grabmyo"
DATA = os.path.join(ROOT, "grabmyo_intent_dataset.csv")


def group_features_by_channel(feature_cols):
    groups = {}
    for col in feature_cols:
        if col.startswith("ch"):
            prefix = col.split("_")[0]
            groups.setdefault(prefix, []).append(col)
    return groups


def load():
    df = pd.read_csv(DATA)
    parts = sorted(df["participant"].unique())
    print("Participants:", len(parts))
    return df, parts


def split(df, parts):
    n = len(parts)
    train_p = parts[:int(0.8*n)]
    val_p   = parts[int(0.8*n):int(0.9*n)]
    test_p  = parts[int(0.9*n):]

    train = df[df.participant.isin(train_p)].reset_index(drop=True)
    val   = df[df.participant.isin(val_p)].reset_index(drop=True)
    test  = df[df.participant.isin(test_p)].reset_index(drop=True)

    return train, val, test


def get_Xy(df, feature_cols=None):
    """
    If feature_cols is None, infer from df.
    Otherwise, use the provided feature_cols in that exact order.
    """
    meta = ["participant","session","gesture","gesture_name",
            "trial","t_rel_s","intent","intent_idx"]
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in meta]
    X = df[feature_cols].values
    y = df["intent_idx"].values
    return X, y, feature_cols


def augment_stroke_style(train_df, feature_cols, factor=0.5, n_copies=1, drop_prob=0.1):
    groups = group_features_by_channel(feature_cols)
    augmented_rows = []

    X = train_df[feature_cols].values
    y = train_df["intent_idx"].values

    for _ in range(n_copies):
        X_aug = X.copy()

        for prefix, cols in groups.items():
            idxs = [feature_cols.index(c) for c in cols]

            gain = np.random.normal(loc=1.0, scale=factor/3.0)
            median_abs = np.median(np.abs(X[:, idxs]))
            bias = np.random.normal(loc=0.0, scale=factor * median_abs * 0.1)

            X_aug[:, idxs] = X_aug[:, idxs] * gain + bias

            if np.random.rand() < drop_prob:
                X_aug[:, idxs] = 0.0

        noise_scale = factor * np.median(np.abs(X))
        X_aug += np.random.normal(0.0, noise_scale * 0.01, size=X_aug.shape)

        df_aug = pd.DataFrame(X_aug, columns=feature_cols)
        df_aug["intent_idx"] = y
        df_aug["augmented"] = 1
        augmented_rows.append(df_aug)

    train_df_copy = train_df.copy()
    train_df_copy["augmented"] = 0

    train_aug = pd.concat([train_df_copy] + augmented_rows, axis=0).reset_index(drop=True)
    return train_aug


def add_rest_quality_feature(df):
    env_cols = [c for c in df.columns if c.endswith("env_rms")]
    df["rest_activity"] = df[env_cols].sum(axis=1)
    return df


def train_model(X, y):
    weights = compute_sample_weight("balanced", y)
    clf = HistGradientBoostingClassifier(
        learning_rate=0.07,
        max_leaf_nodes=50,
        max_iter=300,
        random_state=42
    )
    clf.fit(X, y, sample_weight=weights)
    return clf


def evaluate(clf, X, y, split):
    y_pred = clf.predict(X)
    print(f"\n== {split} ==")
    print("Acc:", accuracy_score(y, y_pred))
    print(classification_report(y, y_pred))
    print(confusion_matrix(y, y_pred))


def main():
    df, parts = load()
    df = add_rest_quality_feature(df)

    train_df, val_df, test_df = split(df, parts)

    # 1) Infer feature_cols ONCE from the clean training set
    X_train_raw, y_train_raw, feature_cols = get_Xy(train_df, feature_cols=None)
    print(f"Num features: {len(feature_cols)}")

    # 2) Augment training set using that fixed feature_cols
    train_df_aug = augment_stroke_style(train_df, feature_cols,
                                        factor=0.5, n_copies=1, drop_prob=0.1)

    # 3) Build X/y for all splits using the SAME feature_cols
    X_train, y_train, _ = get_Xy(train_df_aug, feature_cols=feature_cols)
    X_val,   y_val,   _ = get_Xy(val_df,       feature_cols=feature_cols)
    X_test,  y_test,  _ = get_Xy(test_df,      feature_cols=feature_cols)

    clf = train_model(X_train, y_train)

    evaluate(clf, X_train, y_train, "TRAIN")
    evaluate(clf, X_val,   y_val,   "VAL")
    evaluate(clf, X_test,  y_test,  "TEST")

    joblib.dump(clf, os.path.join(ROOT, "intent_model.pkl"))
    print("Saved model.")

    # Build metadata dynamically from df so mapping is always correct
    intent_map = df[["intent","intent_idx"]].drop_duplicates().sort_values("intent_idx")
    intent_to_idx = {row["intent"]: int(row["intent_idx"]) for _, row in intent_map.iterrows()}
    idx_to_intent = {int(row["intent_idx"]): row["intent"] for _, row in intent_map.iterrows()}

    meta = {
        "feature_cols": feature_cols,
        "intent_to_idx": intent_to_idx,
        "idx_to_intent": {str(k): v for k, v in idx_to_intent.items()}
    }
    with open(os.path.join(ROOT,"intent_meta.json"),"w") as f:
        json.dump(meta, f, indent=4)
    print("Saved metadata.")


if __name__ == "__main__":
    main()
