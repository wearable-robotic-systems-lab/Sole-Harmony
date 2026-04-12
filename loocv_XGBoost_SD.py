"""
Sole-HARmony — LOOCV (Static vs Dynamic)

Performs leave-one-subject-out cross-validation (LOSO-CV) on 10 subjects.
For each fold:
- trains model on 9 subjects
- validates (80/20 split per subject)
- tests on held-out subject
- saves subject-specific model

Input:  FeatureTables/*_features.csv
Output: Trained model (JSON) + performance metrics

"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (confusion_matrix,
                             balanced_accuracy_score, f1_score)
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.metrics import make_scorer
from sklearn.model_selection import GroupKFold
import os, glob, json
from xgboost import DMatrix
from xgboost.callback import TrainingCallback




# ============================================================
# CONFIG
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
FEATURES_DIR = os.path.join(SCRIPT_DIR, "FeatureTables")

SAVE_DIR = "loocv_XGBoost_SD"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# LOADING UTILITIES
# ============================================================
def parse_subject_from_filename(path):
    base = os.path.basename(path)
    return base.split("_")[0]


def load_files_for_prefix(prefix_range, suffix):
    files = glob.glob(os.path.join(FEATURES_DIR, f"*{suffix}"))
    pick = [f for f in files if parse_subject_from_filename(f) in prefix_range]
    if not pick:
        return None

    dfs = []
    for f in pick:
        df = pd.read_csv(f)

        # remove AP NaNs
        if "label_AP" in df.columns:
            df = df.dropna(subset=["label_AP"])

        df["subject"] = parse_subject_from_filename(f)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)



# ============================================================
# CALLBACK FOR EARLY STOPPING USING macro F1 score
# ============================================================
class macroF1Logger(TrainingCallback):
    def __init__(self, X_val, y_val, patience=50, print_every=20):
        self.X_val_dm = DMatrix(X_val)
        self.y_val = y_val
        self.f1_per_iter = []
        self.best_score = 0
        self.best_iteration = 0
        self.patience = patience
        self.print_every = print_every
        self.wait = 0
        self.min_delta = 1e-2      
        self.smooth_k = 3 

    def after_iteration(self, model, epoch, evals_log):
        y_pred_raw = model.predict(self.X_val_dm)
        if len(y_pred_raw.shape) > 1:
            y_pred = np.argmax(y_pred_raw, axis=1)
        else:
            # Binary classification: threshold at 0.5
            y_pred = (y_pred_raw > 0.5).astype(int)

        score = f1_score(self.y_val, y_pred, average="macro")
        self.f1_per_iter.append(score)

        # ---- smoothing ----
        if len(self.f1_per_iter) >= self.smooth_k:
            score_eval = np.mean(self.f1_per_iter[-self.smooth_k:])
        else:
            score_eval = score

        if (epoch + 1) % self.print_every == 0:
            print(f"[{epoch+1}] Macro-F1: {score_eval:.5f}")

        # ---- early stopping with min_delta ----
        if score_eval > self.best_score + self.min_delta:
            self.best_score = score_eval
            self.best_iteration = epoch
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(
                    f"Early stopping at iteration {epoch} "
                    f"(no Macro-F1 improvement > {self.min_delta} "
                    f"for {self.patience} rounds)"
                )
                return True  # stop training
        return False



# ============================================================
# LOAD FULL DATA
# ============================================================
print("Loading data...")
train_subjects = {f"C{i:03d}" for i in range(1, 11)}  
df_all = load_files_for_prefix(train_subjects, "_features.csv") 


df_all = df_all.loc[:, ~df_all.columns.str.startswith(('label_AP'))]

############
# # Include only features from IMU or FSR
# cols_to_keep = df_all.columns[df_all.columns.str.contains("Acc|Gyr|Cor")]

# df_all = df_all[list(cols_to_keep)+ ["label_Cam","subject","t_window_ms"]] # + ["label_Cam","subject","t_window_ms"]
############

df_all['label_Cam'] = df_all['label_Cam'].map(
    {0: 0, 1: 0, 2: 1, 3: 1, 4: 1}
)


# ============================================================
# SPLIT INTO SUBJECT
# ============================================================
subjects = sorted(df_all["subject"].unique())[:10]
subject_dfs = [df_all[df_all["subject"] == s].reset_index(drop=True) for s in subjects]
num_subjects = len(subject_dfs)

print("Using subjects:", subjects)

# ============================================================
# HELPERS
# ============================================================
def split_xy(df):
    X = df.drop(columns=["label_Cam", "subject","t_window_ms"]).astype(np.float32).values
    y = df["label_Cam"].values
    return X, y


# ============================================================
# MAIN LOSO FUNCTION
# ============================================================
def eval_sbj(subject_dfs, test_idx):

    print(f"\n========== TEST SUBJECT {subjects[test_idx]} ==========")

    np.random.seed(42)

    remaining_subjects = [i for i in range(num_subjects) if i != test_idx]

    df_train_list = []
    df_val_list   = []
    group_labels  = []

    for sbj_idx in remaining_subjects:
        df_sbj = subject_dfs[sbj_idx]

        # Shuffle rows within subject
        idx = np.random.permutation(len(df_sbj))
        split_point = int(0.8 * len(df_sbj))

        train_idx = idx[:split_point]
        val_idx   = idx[split_point:]

        df_train_sbj = df_sbj.iloc[train_idx]
        df_val_sbj   = df_sbj.iloc[val_idx]

        df_train_list.append(df_train_sbj)
        df_val_list.append(df_val_sbj)

        # Group labels for training only
        group_labels.extend([sbj_idx] * len(df_train_sbj))

    df_test = subject_dfs[test_idx]
    df_train = pd.concat(df_train_list, ignore_index=True)
    df_val   = pd.concat(df_val_list,   ignore_index=True)
    group_labels = np.array(group_labels)

    X_train, y_train = split_xy(df_train)
    X_val,   y_val   = split_xy(df_val)
    X_test,  y_test  = split_xy(df_test)

    parameters = {
    'max_depth': Integer(2,5,prior='uniform'),
    'learning_rate': Real(0.01,0.1,prior='log-uniform'),
    'gamma': Integer(3,6,prior='uniform'),
    'reg_alpha': Real(0.1,2,prior='log-uniform'),
    'reg_lambda': Real(0.05,1.5,prior='log-uniform'),
    'scale_pos_weight': Integer(5,8,prior='uniform'),
    'colsample_bytree' : Real(0.2,1,prior='log-uniform'), 
    'min_child_weight': Integer(1, 50, prior='uniform')
    }

    scorer = make_scorer(f1_score)

    optimal_params = BayesSearchCV(
            XGBClassifier(objective = 'binary:logistic',
                                    random_state=42,
                                    n_estimators=300,
                                    ),
            parameters,
            scoring = scorer,
            n_iter = 60,
            n_jobs = -1,
            cv = GroupKFold(n_splits=3)
        )


    optimal_params.fit(X_train, y_train, groups=group_labels, verbose = 10)
    best_params = optimal_params.best_params_
    print("Best params:", best_params)

    # ==== Final model ====
    f1_logger = macroF1Logger(X_val, y_val)

    model = XGBClassifier(
        objective='binary:logistic',
        n_estimators=300,
        random_state=42,
        callbacks=[f1_logger],
        **best_params
    )

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=20)

    imp = model.feature_importances_
    columns = df_train.drop(columns=["label_Cam", "subject","t_window_ms"]).columns
    print("\n\U0001f50d Top Feature Importances:")
    for i in np.argsort(imp)[::-1][:10]:
        print(f"{columns[i]:<30} Importance: {imp[i]:.6f}")
    
    # BEST ITERATION
    best_iter = f1_logger.best_iteration + 1
    filename_booster = f"XGB_SD_{subjects[test_idx]}.json"
    save_path = os.path.join(SAVE_DIR, filename_booster)
    model.get_booster().save_model(save_path)

    meta = {
        "best_iteration": int(best_iter)
    }
    filename_meta = f"XGB_SD_{subjects[test_idx]}_meta.json"
    save_path_meta = os.path.join(SAVE_DIR, filename_meta)
    with open(save_path_meta, "w") as f:
        json.dump(meta, f, indent=2)


    # ==== Test ====
    booster = model.get_booster()
    dtest = DMatrix(X_test)
    y_prob = booster.predict(
        dtest,
        iteration_range=(0, best_iter)
    )
    y_pred = (y_prob > 0.50).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    ba = balanced_accuracy_score(y_test, y_pred)
    print(f"📌 Subject {subjects[test_idx]} — BA = {ba:.3f}")
    print("\nConfusion Matrix for subject", subjects[test_idx])
    print(cm)

    return {"subject": subjects[test_idx], "cm": cm, "ba": ba}



# ============================================================
# RUN LOSO AND SAVE EVERYTHING INTO ONE WORKBOOK
# ============================================================
feature_names = subject_dfs[0].drop(columns="label_Cam").columns
results = []

for i in range(num_subjects):
    res = eval_sbj(subject_dfs, i) 
    results.append(res)


# ============================================================
# PRINT COMBINED METRICS ONLY
# ============================================================
all_cm = sum([r["cm"] for r in results])
all_ba = np.mean([r["ba"] for r in results])

print("\n======= OVERALL RESULTS =======\n")
print("Combined Confusion Matrix:\n", all_cm)
print(f"\nMean Balanced Accuracy: {all_ba:.3f}")
