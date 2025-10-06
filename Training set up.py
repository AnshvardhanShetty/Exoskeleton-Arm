import os
import pandas as pd 

def load_all_csvs(root_folder):
    dataframes = []
    for subdirectory, _, files in os.walk(root_folder):
        for filename in files:
            if filename.endswith(".csv"):
                fpath = os.path.join(subdirectory, filename)
                df = pd.read_csv(fpath)
                dataframes.append(df)
    return pd.concat(dataframes, ignore_index = True)

train_folder = "/Users/anshshetty/Desktop/Exoskeleton Arm /training data"
train_df = load_all_csvs(train_folder)

# --- merge testing files ---
test_folder = "/Users/anshshetty/Desktop/Exoskeleton Arm /testing data"
test_df = load_all_csvs(test_folder)

# --- save outputs OUTSIDE the folders being scanned ---
out_dir = "/Users/anshshetty/Desktop/Exoskeleton Arm "   # one level up
train_out = os.path.join(out_dir, "training_data_flex.csv")
test_out  = os.path.join(out_dir, "testing_data_flex.csv")

train_df.to_csv(train_out, index=False)
test_df.to_csv(test_out, index=False)