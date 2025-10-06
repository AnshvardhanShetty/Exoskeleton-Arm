import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.utils import resample
import sys, os, time, hashlib
from pathlib import Path

print("\n===== RUN START =====")
print("PYTHON EXE:", sys.executable)
print("SCRIPT    :", __file__)

def md5(p):
    try:
        with open(p, "rb") as f: 
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        return f"<md5 error: {e}>"

print("SCRIPT MD5:", md5(__file__))

train_csv = "/Users/anshshetty/Desktop/Exoskeleton Arm /training_data_flex.csv"
test_csv  = "/Users/anshshetty/Desktop/Exoskeleton Arm /testing_data_flex.csv"

tp = Path(train_csv); sp = Path(test_csv)
print("TRAIN CSV exists?", tp.exists(), "→", tp.resolve() if tp.exists() else "MISSING")
print("TEST  CSV exists?", sp.exists(), "→", sp.resolve() if sp.exists() else "MISSING")
print("===== RUN SANITY OK =====\n")

train_df = pd.read_csv("/Users/anshshetty/Desktop/Exoskeleton Arm /training_data_flex.csv")
test_df = pd.read_csv("/Users/anshshetty/Desktop/Exoskeleton Arm /testing_data_flex.csv")
def has_temporal_cols(df):
    return df.columns.str.contains(r"lag|d1|d2|roll|ema|plateau", case=False).any()

print("[check] temporal in TRAIN? ", has_temporal_cols(train_df))
print("[check] temporal in TEST ? ", has_temporal_cols(test_df))
# If either is False, you’re not training on the engineered files you think you are.
drop = {"t_rel_s","t_abs_s","label","__seq__"}  # __seq__ if present

# pick the intersection + numeric only
feature_columns = [
    c for c in train_df.columns
    if c in test_df.columns
    and c not in drop
    and pd.api.types.is_numeric_dtype(train_df[c])
]
keep_train = train_df[feature_columns].notna().any(axis=1)
keep_test  = test_df[feature_columns].notna().any(axis=1)

print("[check] all-NaN rows dropped → train:", (~keep_train).sum(), " test:", (~keep_test).sum())

train_df = train_df.loc[keep_train].reset_index(drop=True)
test_df  = test_df.loc[keep_test].reset_index(drop=True)

print("RAW TRAIN COUNTS:\n", train_df["label"].value_counts(dropna=False))
print("RAW  TEST COUNTS:\n", test_df["label"].value_counts(dropna=False))
print("RAW TRAIN UNIQUES (repr):", list(map(repr, pd.unique(train_df["label"]))))



def downsample(df, label_col = "label", random_state=42):
    
    counts = df [label_col].value_counts() #counts the number of windows per class
    print("\n[downsample] incoming counts:\n", counts)           # DEBUG

    n_min = counts.min() #find the smalles class size
    parts = [] #prepares an emoty list to collect per-class dataframes after downsampling.
    for c, n in counts.items(): #loops through eaach class and its count n
        df_c = df[df[label_col] == c] #selects only the rows of the current class c in the iteration, df_c is the subset to keep or downsample
        if n > n_min: #if this class is larger than the smallest class.
            df_c = resample(
                df_c,
                replace = False,
                n_samples = n_min,
                random_state = random_state
            )
        print(f"[downsample] keeping {len(df_c)} rows of class '{c}'")  # DEBUG
        parts.append(df_c)
    out = (pd.concat(parts, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True))
    print("[downsample] outgoing counts:\n", out[label_col].value_counts(dropna=False))  # DEBUG
    return out
def clean_labels(df):
    df = df.copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    bad = df["label"].isin(["", "nan", "none", "null"])
    if bad.any():
        print(f"[clean_labels] dropping {bad.sum()} rows with empty labels")
        df = df[~bad]
    return df.reset_index(drop=True)

train_df = clean_labels(train_df)
test_df  = clean_labels(test_df)

# if you balance later, do it AFTER this cleaning
train_df_balanced = downsample(train_df, "label", 42)

y_train = train_df_balanced["label"].values
y_test  = test_df["label"].values

X_train = train_df_balanced[feature_columns].values
X_test = test_df[feature_columns].values

y_train = train_df_balanced["label"].astype(str).str.strip().str.lower().values
y_test  = test_df["label"].astype(str).str.strip().str.lower().values

le = LabelEncoder()
le.fit(pd.Index(y_train).union(pd.Index(y_test)))  # robust fit
y_train_enc = le.transform(y_train)
y_test_enc  = le.transform(y_test)
print("Encoder classes:", list(le.classes_))

print("Classes:", list(le.classes_))
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

def logreg(X_train, y_train_enc, X_test, y_test_enc, le):
    logreg_pipe = Pipeline([ #chains several steps together, the data goes through each step in order. Prevents having to preprocess manually as the pipeline takes care of it.
        ("imputer", SimpleImputer(strategy="median")), #if there are any missing values, replace it with the median of that column, median is robust to outliers.
        ("scaler", StandardScaler()), #scales the features using z-scale so that they are in equal units.
        ("clf", LogisticRegression( # applies logistic regression
            max_iter = 2000, #logistic regression is solved by an iterative optimisation algorithm, max_iter is the max. number of steps the optimiser is allowed to take. 
            class_weight = "balanced", #Basically gives each class an equal chance at prediction by increasing the loss penalty for mistakes on underrpresresnted classes. 
            multi_class = "auto", #uses softmax regression
            solver="lbfgs", #used to find weights
            C=1.0, #adds a penalty for big weights, ensuring they don't over grow to overfit
            n_jobs=None #How many CPU cores they use, 1 core is one independent worker inside the CPU.
        ))
    ])

    logreg_pipe.fit(X_train, y_train_enc)

    y_pred = logreg_pipe.predict(X_test)
    print(classification_report(y_test_enc, y_pred, target_names=le.classes_, digits=3))

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test_enc, y_pred, display_labels=le.classes_, normalize='true', cmap='Blues'
    )
    plt.title("Normalized Confusion Matrix (true rows)")
    plt.tight_layout()
    plt.show()

    acc = accuracy_score(y_test_enc, y_pred)
    print(acc)
    f1m = f1_score(y_test_enc, y_pred, average = "macro")
    print(f1m)
    cm = confusion_matrix(y_test_enc, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print("Confusion matrix (rows = true, cols = predicted):")
    print(cm_df)
    
    return logreg_pipe


#logreg_model = logreg(X_train, y_train_enc, X_test, y_test_enc, le)

def random_forest(X_train, y_train_enc, random_state = 42):
    clf = RandomForestClassifier(
        n_estimators = 200, #number of trees
        max_features = "sqrt", #random subset of features
        class_weight = "balanced", #handles imbalance
        random_state=random_state,
        n_jobs = -1 #use all cpu cores
    )
    clf.fit(X_train, y_train_enc)
    
    y_pred = clf.predict(X_test)
    print(classification_report(y_test_enc, y_pred, target_names=le.classes_, digits=3))

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test_enc, y_pred, display_labels=le.classes_, normalize='true', cmap='Blues'
    )
    plt.title("Normalized Confusion Matrix (true rows)")
    plt.tight_layout()
    plt.show()

    acc = accuracy_score(y_test_enc, y_pred)
    print(acc)
    f1m = f1_score(y_test_enc, y_pred, average = "macro")
    print(f1m)
    cm = confusion_matrix(y_test_enc, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print("Confusion matrix (rows = true, cols = predicted):")
    print(cm_df)
    
    return clf
    
rf_model = random_forest(X_train, y_train_enc, random_state=42)
imp = rf_model.feature_importances_
top_idx = np.argsort(imp)[::-1][:20]

for i in top_idx:
    print(feature_columns[i], imp[i])