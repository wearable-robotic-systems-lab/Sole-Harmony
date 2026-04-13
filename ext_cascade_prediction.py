"""
Sole-HARmony — Cascade Classifier (External Validation)

Applies the trained models (SD, SS, WS, sAD) to evaluate the hierarchical XGBoost cascade
classifier on unseen subjects. Loads pre-trained models, performs multi-stage
classification, and outputs final activity predictions along with
evaluation metrics.

Input:  FeatureTables/*_features.csv (test subjects) + XGBoost models
Output: CSV file with predictions + confusion matrix
"""

import os
from sklearn.metrics import (
    confusion_matrix, balanced_accuracy_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np
import pandas as pd
import json
import xgboost as xgb
from xgboost import XGBClassifier
from scipy.stats import mode


#########################################################################################
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
FEATURES_DIR = os.path.join(SCRIPT_DIR, "FeatureTables")
DROP_COLS = ["label_Cam", "subject","t_window_ms","session"]
SAVE_DIR = os.path.join(SCRIPT_DIR, "cascade_predictions")

MODEL_DIRS = {
    "SD":  os.path.join(SCRIPT_DIR, "ext_XGBoost_SD"),
    "SS":  os.path.join(SCRIPT_DIR, "ext_XGBoost_SS"),
    "WS":  os.path.join(SCRIPT_DIR, "ext_XGBoost_WS"),
    "sAD": os.path.join(SCRIPT_DIR, "ext_XGBoost_sAD"),
}

# LOAD MODELS
model_SD  = XGBClassifier()
model_SS  = XGBClassifier()
model_WS  = XGBClassifier()
model_sAD = XGBClassifier()

model_SD.load_model(os.path.join(MODEL_DIRS["SD"],  f"XGB_SD.json"))
with open(os.path.join(MODEL_DIRS["SD"],"XGB_SD_meta.json"), "r") as f:
    meta_SD = json.load(f)
best_iter_SD = meta_SD["best_iteration"]
model_SS.load_model(os.path.join(MODEL_DIRS["SS"],  f"XGB_SS.json"))
with open(os.path.join(MODEL_DIRS["SS"],"XGB_SS_meta.json"), "r") as f:
    meta_SS = json.load(f)
best_iter_SS = meta_SS["best_iteration"]
model_WS.load_model(os.path.join(MODEL_DIRS["WS"],  f"XGB_WS.json"))
with open(os.path.join(MODEL_DIRS["WS"],"XGB_WS_meta.json"), "r") as f:
    meta_WS = json.load(f)
best_iter_WS = meta_WS["best_iteration"]
model_sAD.load_model(os.path.join(MODEL_DIRS["sAD"], f"XGB_sAD.json"))
with open(os.path.join(MODEL_DIRS["sAD"],"XGB_sAD_meta.json"), "r") as f:
    meta_sAD = json.load(f)
best_iter_sAD = meta_sAD["best_iteration"]


# ============================================================
# LOADING UTILITIES
# ============================================================

def parse_subject_from_filename(path):
    # expects something like: C000_1658420327_features.csv → "C000"
    base = os.path.basename(path)
    return base.split("_")[0]

def parse_subject_session_from_filename(path):
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]       
    parts = stem.split("_")

    subject = parts[0]                         # C001
    session = parts[1] if len(parts) > 1 else "unknown"   # 1658420327

    return subject, session

def load_files_for_prefix(prefix_range, suffix):
    files = glob.glob(os.path.join(FEATURES_DIR, f"*{suffix}"))
    dfs = []

    for f in files:
        subject, session = parse_subject_session_from_filename(f)
        if subject not in prefix_range:
            continue

        df = pd.read_csv(f)
        subject, session = parse_subject_session_from_filename(f)
        df["subject"] = subject
        df["session"] = session
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# ============================================================
# SS TEMPORAL SMOOTHING
# ============================================================

def smooth_labels_in_segment(indices, labels, neighbors):
    indices = np.array(indices)
    labels = np.array(labels, dtype=int)
    smoothed = labels.copy()

    # --- find segment boundaries (gaps > 1) ---
    gaps = np.where(np.diff(indices) > 1)[0]
    segment_starts = np.insert(gaps + 1, 0, 0)
    segment_ends   = np.append(gaps, len(indices) - 1)

    # --- smooth each segment independently ---
    for start, end in zip(segment_starts, segment_ends):
        seg = labels[start:end+1]

        for i in range(len(seg)):
            s = max(0, i - neighbors)
            e = min(len(seg), i + neighbors + 1)
            # safe mode extraction
            m = mode(seg[s:e], keepdims=True).mode
            val = m[0] if hasattr(m, "__len__") else m
            smoothed[start + i] = int(val)

    return smoothed

# ============================================================
# CASCADE CLASSIFIER (WITH SS SMOOTHING)
# ============================================================

def classify_subject_cascade(X_test, model_SD, model_SS, model_WS, model_sAD, 
                             best_iter_SD, best_iter_SS, best_iter_WS, best_iter_sAD):

    final_pred = np.full(len(X_test), -1)

    ## -- Static vs dynamic prediction -- ##
    X_SD = X_test
    dtest = xgb.DMatrix(X_SD)

    booster = model_SD.get_booster()
    P_pred_SD = booster.predict(
        dtest,
        iteration_range=(0, best_iter_SD)
    )
    pred_SD = (P_pred_SD > 0.50).astype(int)

    # Final Indices
    idx_static = np.where(pred_SD == 0)[0]
    idx_dynamic = np.where(pred_SD == 1)[0]

    # Static → Sitting vs Standing
    if len(idx_static) > 0:
        X_SS = X_test
        d_static = xgb.DMatrix(X_SS[idx_static, :])

        booster = model_SS.get_booster()
        p_ss = booster.predict(
            d_static,
            iteration_range=(0, best_iter_SS)
        )
        pred_SS = (p_ss > 0.50).astype(int)

        # Smooth
        smoothed_pred_SS = smooth_labels_in_segment(idx_static, pred_SS, neighbors=4) #6

        final_pred[idx_static] = smoothed_pred_SS  # Sitting=0, Standing=1

    # Dynamic → Walking vs Stairs
    if len(idx_dynamic) > 0:
        d_dynamic = xgb.DMatrix(X_test[idx_dynamic,:])
        booster = model_WS.get_booster()
        p_ws = booster.predict(
            d_dynamic,
            iteration_range=(0, best_iter_WS)
        )
        pred_WS = (p_ws > 0.50).astype(int)

        # walking and stairs
        idx_walking = idx_dynamic[pred_WS == 0]
        idx_stairs  = idx_dynamic[pred_WS == 1]

        final_pred[idx_walking] = 2  # Walking = 2

        if len(idx_stairs) > 0:
            d_stairs = xgb.DMatrix(X_test[idx_stairs,:])
            booster = model_sAD.get_booster()
            p_sad = booster.predict(
                d_stairs,
                iteration_range=(0, best_iter_sAD)
            )
            pred_sAD = (p_sad > 0.50).astype(int)

            final_pred[idx_stairs] = 3 + pred_sAD  # 3 = ascending, 4 = descending

    return final_pred 

# Load DATA
test_subjects  = {f"C0{i:02d}" for i in range(11, 14)}  # C011–C014
df_test      = load_files_for_prefix(test_subjects,  "_features.csv")

# ## Include only features from IMU or FSR
# cols_to_keep = df_test.columns[df_test.columns.str.contains("Acc|Gyr|Cor")]
# df_test = df_test[list(cols_to_keep)+ ["label_Cam", "subject","t_window_ms","session"]] # + ["label_Cam", "subject","t_window_ms","session"]
## 

X_test_all = df_test.drop(columns=DROP_COLS).values.astype(np.float32)
y_test_all = df_test['label_Cam'].astype(int).values

y_pred_all = classify_subject_cascade(X_test_all, model_SD, model_SS, model_WS, model_sAD,
                                      best_iter_SD, best_iter_SS, best_iter_WS, best_iter_sAD) #

cm_all = confusion_matrix(y_test_all, y_pred_all, labels=[0, 1, 2, 3, 4])
cm_normalized_all = cm_all.astype('float') / cm_all.sum(axis=1, keepdims=True)
cm_percent_all = np.round(cm_normalized_all * 100, 1)  # Convert to %

# --- Activity names ---
activity_names = ['Sitting', 'Standing', 'Walking', 'Stairs-Down', 'Stairs-Up']

# # Report
acc_all       = balanced_accuracy_score(y_test_all, y_pred_all)
macro_f1 = f1_score(y_test_all, y_pred_all, average="macro")

# --- Print Balanced Accuracy ---
print(f"\n Total Balanced Accuracy")
print(f"    Balanced Accuracy: {acc_all:.3f}")
print(f"    Macro F1:          {macro_f1:.3f}")

# --- Plot ---
plt.figure(figsize=(6, 5))
sns.heatmap(cm_percent_all, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=activity_names,
            yticklabels=activity_names,
            cbar_kws={'label': 'Percentage (%)'})

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Total Confusion Matrix (% per class)")
plt.tight_layout()
plt.show()


# === Total Confusion Matrix ===
print("\n🧾 Total Confusion Matrix:")
print(cm_all)

# === SAVE PREDICTIONS TO EXCEL ===========================================
# Build dataframe with results
df_results = pd.DataFrame({
    "subject": df_test["subject"].values,
    "session": df_test["session"].values,
    "y_true": y_test_all,
    "y_pred": y_pred_all
})

os.makedirs(SAVE_DIR, exist_ok=True)
save_path = os.path.join(SAVE_DIR,"ext_XGB_cascade_prediction.csv")
df_results.to_csv(save_path, index=False)

print(f"\n📁 Saved prediction results to: {save_path}")
print(df_results.head())

