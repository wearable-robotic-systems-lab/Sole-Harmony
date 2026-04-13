"""
Sole-HARmony — LOOCV Cascade Classifier

Applies subject-specific models from leave-one-subject-out cross-validation
to evaluate the hierarchical XGBoost cascade (SD, SS, WS, sAD).
Generates per-session predictions, aggregates results across subjects,
and reports overall performance with a normalized confusion matrix.

Input:  FeatureTables/*_features.csv (test subjects) + subject-specific XGBoost models
Output: CSV file with predictions + confusion matrix
"""


import os
import glob
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

from scipy.stats import mode
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score

# ============================================================
# CONFIG
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
FEATURES_DIR = os.path.join(SCRIPT_DIR, "FeatureTables")
SAVE_DIR = os.path.join(SCRIPT_DIR,"cascade_predictions")
os.makedirs(SAVE_DIR, exist_ok=True)


MODEL_DIRS = {
    "SD":  os.path.join(SCRIPT_DIR, "loocv_XGBoost_SD"),
    "SS":  os.path.join(SCRIPT_DIR, "loocv_XGBoost_SS"),
    "WS":  os.path.join(SCRIPT_DIR, "loocv_XGBoost_WS"),
    "sAD": os.path.join(SCRIPT_DIR, "loocv_XGBoost_sAD"),
}

CLASS_NAMES = [
    "Sitting",
    "Standing",
    "Walking",
    "Stairs Down",
    "Stairs Up"
]

LABEL_COL = "label_Cam"


# ============================================================
# SS TEMPORAL SMOOTHING
# ============================================================

def smooth_labels_in_segment(indices, labels, neighbors=3):
    indices = np.asarray(indices)
    labels = np.asarray(labels, dtype=int)
    smoothed = labels.copy()

    if len(indices) == 0:
        return smoothed

    gaps = np.where(np.diff(indices) > 1)[0]
    segment_starts = np.insert(gaps + 1, 0, 0)
    segment_ends   = np.append(gaps, len(indices) - 1)

    for start, end in zip(segment_starts, segment_ends):
        seg = labels[start:end+1]

        for i in range(len(seg)):
            s = max(0, i - neighbors)
            e = min(len(seg), i + neighbors + 1)
            m = mode(seg[s:e], keepdims=True).mode
            smoothed[start + i] = int(m[0])

    return smoothed


# ============================================================
# LOADING UTILITIES
# ============================================================

def parse_subject_from_filename(path):
    base = os.path.basename(path)
    return base.split("_")[0]

def parse_subject_session_from_filename(path):
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]          
    parts = stem.split("_")

    subject = parts[0]                         
    session = parts[1] if len(parts) > 1 else "unknown"   

    return subject, session


def load_files_for_prefix(prefix_range, suffix):
    files = glob.glob(os.path.join(FEATURES_DIR, f"*{suffix}"))
    dfs = []

    for f in files:
        subject, session = parse_subject_session_from_filename(f)
        if subject not in prefix_range:
            continue

        df = pd.read_csv(f)

        if "label_AP" in df.columns:
            df = df.dropna(subset=["label_AP"])

        df = df.assign(
            subject=subject,
            session=session,
            source_file=os.path.basename(f)
        )   

        dfs.append(df)

    if not dfs:
        raise RuntimeError("No matching CSV files found")

    return pd.concat(dfs, ignore_index=True).copy()


# ============================================================
# CASCADE CLASSIFIER (WITH SS SMOOTHING)
# ============================================================

def classify_subject_cascade(X, model_SD, model_SS, model_WS, model_sAD, 
                             best_iter_SD, best_iter_SS, best_iter_WS, best_iter_sAD, ss_neighbors):

    final_pred = np.full(len(X), -1, dtype=int)
    dtest = xgb.DMatrix(X)

    # ---- Static vs Dynamic ----
    booster = model_SD.get_booster()
    p_sd = booster.predict(
        dtest,
        iteration_range=(0, best_iter_SD)
    )
    pred_sd = (p_sd > 0.50).astype(int)

    idx_static  = np.where(pred_sd == 0)[0]
    idx_dynamic = np.where(pred_sd == 1)[0]

    # ---- Sitting vs Standing (STATIC ONLY) ----
    if len(idx_static) > 0:
        d_static = xgb.DMatrix(X[idx_static])
        booster = model_SS.get_booster()
        p_ss = booster.predict(
            d_static,
            iteration_range=(0, best_iter_SS)
        )
        pred_ss_raw = (p_ss > 0.50).astype(int)

        # 🔹 TEMPORAL SMOOTHING
        pred_ss = smooth_labels_in_segment(
            indices=idx_static,
            labels=pred_ss_raw,
            neighbors=ss_neighbors
        )

        final_pred[idx_static] = pred_ss   # 0 Sitting, 1 Standing

    # ---- Walking vs Stairs ----
    if len(idx_dynamic) > 0:
        d_dyn = xgb.DMatrix(X[idx_dynamic])
        booster = model_WS.get_booster()
        p_ws = booster.predict(
            d_dyn,
            iteration_range=(0, best_iter_WS)
        )
        pred_ws = (p_ws > 0.50).astype(int)

        idx_walking = idx_dynamic[pred_ws == 0]
        idx_stairs  = idx_dynamic[pred_ws == 1]

        final_pred[idx_walking] = 2

        # ---- Stair ascent vs descent ----
        if len(idx_stairs) > 0:
            d_stairs = xgb.DMatrix(X[idx_stairs])
            booster = model_sAD.get_booster()
            p_sad = booster.predict(
                d_stairs,
                iteration_range=(0, best_iter_sAD)
            )
            pred_sad = (p_sad > 0.50).astype(int)
            final_pred[idx_stairs] = 3 + pred_sad

    return final_pred


# ============================================================
# LOAD DATA
# ============================================================

print("Loading data...")

train_subjects = {f"C{i:03d}" for i in range(1, 11)}   # C001–C010
df_all = load_files_for_prefix(train_subjects, "_features.csv")

df_all = df_all.loc[:, ~df_all.columns.str.startswith("label_AP")]

############
# # Include only features from IMU or FSR
# cols_to_keep = df_all.columns[df_all.columns.str.contains("Acc|Gyr|Cor")]

# df_all = df_all[list(cols_to_keep)+ ["label_Cam","subject","t_window_ms","session"]] # + ["label_Cam","subject","t_window_ms"]
############

# ============================================================
# MAIN EVALUATION LOOP
# ============================================================

cms = []
ba_list = []
f1_list = []

y_true_all = []
y_pred_all = []
subjects = sorted(df_all["subject"].unique())
for subject in subjects:
    print(f"\n========== Evaluating {subject} ==========")

    df_subj = df_all[df_all["subject"] == subject].reset_index(drop=True)

    model_SD  = XGBClassifier()
    model_SS  = XGBClassifier()
    model_WS  = XGBClassifier()
    model_sAD = XGBClassifier()

    model_SD.load_model(os.path.join(MODEL_DIRS["SD"],  f"XGB_SD_{subject}.json"))
    with open(os.path.join(MODEL_DIRS["SD"], f"XGB_SD_{subject}_meta.json"), "r") as f:
        meta_SD = json.load(f)
    best_iter_SD = meta_SD["best_iteration"]
    model_SS.load_model(os.path.join(MODEL_DIRS["SS"],  f"XGB_SS_{subject}.json"))
    with open(os.path.join(MODEL_DIRS["SS"],  f"XGB_SS_{subject}_meta.json"), "r") as f:
        meta_SS = json.load(f)
    best_iter_SS = meta_SS["best_iteration"]
    model_WS.load_model(os.path.join(MODEL_DIRS["WS"],  f"XGB_WS_{subject}.json"))
    with open(os.path.join(MODEL_DIRS["WS"],  f"XGB_WS_{subject}_meta.json"), "r") as f:
        meta_WS = json.load(f)
    best_iter_WS = meta_WS["best_iteration"]
    model_sAD.load_model(os.path.join(MODEL_DIRS["sAD"], f"XGB_sAD_{subject}.json"))
    with open(os.path.join(MODEL_DIRS["sAD"], f"XGB_sAD_{subject}_meta.json"), "r") as f:
        meta_sAD = json.load(f)
    best_iter_sAD = meta_sAD["best_iteration"]

    
    # per-session evaluation + saving
    for session, df_test in df_subj.groupby("session"):
        y_test = df_test[LABEL_COL].astype(int).values
        X_test = df_test.drop(
            columns=[LABEL_COL, "subject", "session", "source_file", "t_window_ms"],
            errors="ignore"
        ).values.astype(np.float32)

        y_pred = classify_subject_cascade(
            X_test, model_SD, model_SS, model_WS, model_sAD, 
            best_iter_SD, best_iter_SS, best_iter_WS, best_iter_sAD, ss_neighbors=6
        )

        # metrics (optional per session)
        cm = confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4])
        cms.append(cm)
        ba_list.append(balanced_accuracy_score(y_test, y_pred))
        f1_list.append(f1_score(y_test, y_pred, average="macro"))

        # save per session
        out_df = pd.DataFrame({
            "subject": subject,
            "session": session,
            "y_true": y_test,
            "y_pred": y_pred
        })

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

        out_file = os.path.join(SAVE_DIR, f"Predictions_{subject}_{session}.csv")
        out_df.to_csv(out_file, index=False)
        print(f"💾 Saved predictions to: {out_file}")

# ============================================================
# AGGREGATE + NORMALIZED CONFUSION MATRIX 
# ============================================================

cm_total = confusion_matrix(y_true_all, y_pred_all, labels=[0,1,2,3,4])
cm_norm = cm_total / cm_total.sum(axis=1, keepdims=True) * 100
acc_all = balanced_accuracy_score(y_true_all, y_pred_all)
macro_f1 = f1_score(y_true_all, y_pred_all, average="macro")

print("\n========== OVERALL RESULTS ==========")
print(f"Balanced Accuracy: {acc_all:.3f}")
print(f"Macro F1:          {macro_f1:.3f}")
print(cm_total)

plt.figure(figsize=(7,6))
im = plt.imshow(cm_norm, interpolation="nearest", vmin=0, vmax=100, cmap=plt.cm.Oranges)
plt.colorbar(im, label="Proportion")

plt.xticks(range(5), CLASS_NAMES, rotation=45, ha="right")
plt.yticks(range(5), CLASS_NAMES)

plt.title("Normalized Confusion Matrix – Cascade Classifier")
plt.xlabel("Predicted")
plt.ylabel("True")

for i in range(5):
    for j in range(5):
        val = cm_norm[i, j]
        plt.text(
            j, i, f"{val:.1f}",
            ha="center", va="center",
            color="white" if val > 50 else "black"
        )

plt.tight_layout()
plt.show()

