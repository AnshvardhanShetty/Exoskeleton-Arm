import pandas as pd
import os

ROOT_DIR = "/Users/anshshetty/Library/Mobile Documents/com~apple~CloudDocs/ExoHand/grabmyo"

IN_CSV  = os.path.join(ROOT_DIR, "grabmyo_4ch_features_intentready.csv")
OUT_CSV = os.path.join(ROOT_DIR, "grabmyo_intent_dataset.csv")

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

    17: "rest"
}

def main():
    df = pd.read_csv(IN_CSV)
    print("Loaded:", df.shape)

    df["intent"] = df["gesture"].map(INTENT_MAP)

    before = df.shape[0]
    df = df[~df["intent"].isna()].reset_index(drop=True)
    after = df.shape[0]
    print(f"Kept {after}/{before} windows after intent mapping")

    # fixed, explicit mapping order:
    intent_to_idx = {"rest": 0, "close": 1, "open": 2}
    df["intent_idx"] = df["intent"].map(intent_to_idx)

    print("Intents present:", df["intent"].unique())
    df.to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_CSV)

if __name__ == "__main__":
    main()
