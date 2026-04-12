"""
Sole-HARmony — Sitting vs Standing (XGBoost classifier)

Author: Laura De Marzi (Stevens Institute of Technology, WRSLab)

Trains a binary XGBoost classifier (sitting vs standing) using extracted
insole features. Includes Bayesian hyperparameter optimization, early stopping
based on macro-F1, and evaluation on unseen subjects.

Input:  FeatureTables/*.csv
Output: Trained model (JSON) + performance metrics
"""


import os
import glob
import pandas as pd, json
import numpy as np
from xgboost.callback import TrainingCallback
from xgboost import DMatrix
from sklearn.metrics import make_scorer, balanced_accuracy_score, classification_report, confusion_matrix,  f1_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold

# # -----------------------
# Config
# ----------------------- 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
FEATURES_DIR = os.path.join(SCRIPT_DIR, "FeatureTables")
DROP_COLS = ["label_Cam", "subject","t_window_ms"] 

# Create directory for saving models
os.makedirs("XGBoost_SittingStanding", exist_ok=True)

def parse_subject_from_filename(path):
    # expects something like: C000_1658420327_features.csv → "C000"
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
        df["subject"] = parse_subject_from_filename(f)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

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
                    f"⏹️ Early stopping at iteration {epoch} "
                    f"(no Macro-F1 improvement > {self.min_delta} "
                    f"for {self.patience} rounds)"
                )
                return True  # stop training
        return False
    

# Load DATA
print("Loading data...")
train_subjects = {f"C{i:03d}" for i in range(1, 11)}   # C001–C010
test_subjects  = {f"C0{i:02d}" for i in range(11, 14)}  # C011–C013
df_train_all = load_files_for_prefix(train_subjects, "_features.csv")
df_test      = load_files_for_prefix(test_subjects,  "_features.csv")


df_train_all = df_train_all[df_train_all['label_Cam'].isin([0,1])].reset_index(drop=True)
df_test = df_test[df_test['label_Cam'].isin([0, 1])].reset_index(drop=True)

df_train_all = df_train_all.loc[:, ~df_train_all.columns.str.startswith(('label_AP'))]

# ## Include only features from IMU or FSR
# cols_to_keep = df_train_all.columns[df_train_all.columns.str.contains("Acc|Gyr|Cor")]

# df_train_all = df_train_all[list(cols_to_keep)+ ["label_Cam","subject","t_window_ms"]] # + ["label_Cam","subject","t_window_ms"]
# df_test = df_test[list(cols_to_keep)+ ["label_Cam","subject","t_window_ms"]] # + ["label_Cam","subject","t_window_ms"]
# ############

# VALIDATION SPLIT 
LOW_LOADER = "C002" # C002 for features_09 (before C001)
all_subjects = df_train_all['subject'].unique()

df_train_list = []
df_val_list = []

# Ensure reproducibility
np.random.seed(42)

for sbj, df_sbj in df_train_all.groupby("subject"):
    # Shuffle the indices of this subject
    idx = np.random.permutation(len(df_sbj))
    
    # Compute split index (80%)
    split_point = int(0.8 * len(df_sbj))
    
    train_idx = idx[:split_point]
    val_idx = idx[split_point:]
    
    df_train_list.append(df_sbj.iloc[train_idx])
    df_val_list.append(df_sbj.iloc[val_idx])

# Combine all subjects
df_train = pd.concat(df_train_list, ignore_index=True)
df_val   = pd.concat(df_val_list, ignore_index=True)

X_train = df_train.drop(columns=DROP_COLS).values.astype(np.float32)
y_train = df_train['label_Cam'].astype(int).values
groups = df_train['subject'].values


X_val = df_val.drop(columns=DROP_COLS).values.astype(np.float32)
y_val = df_val['label_Cam'].astype(int).values


X_test = df_test.drop(columns=DROP_COLS).values.astype(np.float32)
y_test = df_test['label_Cam'].astype(int).values


# ASSIGN MORE WEIGTH TO C002
sample_weight = np.ones(len(y_train))
sample_weight[df_train['subject'].values == LOW_LOADER] = 9.0   # increase importance
print(f"Assigned HIGH sample weight to {LOW_LOADER}")


print("Bayesian search...")
# BAYESIAN OPTIMIZATION

parameters = {
    'max_depth': Integer(2,5,prior='uniform'),
    'learning_rate': Real(0.01,0.1,prior='log-uniform'),
    'gamma': Integer(2,5,prior='uniform'),
    'reg_alpha': Real(0.1,3,prior='log-uniform'),
    'reg_lambda': Real(0.1, 3,prior='log-uniform'),
    'scale_pos_weight': Real(7,10,prior='log-uniform'), 
    'min_child_weight': Integer(1, 50, prior='uniform'), 
    'colsample_bytree' : Real(0.2,1,prior='log-uniform'), 
    }

scorer = make_scorer(f1_score, average="macro")

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

optimal_params.fit(X_train, y_train,
                       groups=groups,
                       sample_weight=sample_weight,
                       verbose = 10,
                       )


best = optimal_params.best_params_
print("Best params:", optimal_params.best_params_, "CV score:", optimal_params.best_score_)
f1_logger = []
f1_logger = macroF1Logger(X_val, y_val)

model = XGBClassifier(
objective='binary:logistic',
n_estimators=300,
random_state=42,
**best,
callbacks=[f1_logger]
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)],   sample_weight=sample_weight,
 verbose=20) 

imp = model.feature_importances_
columns = df_train.drop(columns=DROP_COLS).columns 
print("\n\U0001f50d Top Feature Importances:")
for i in np.argsort(imp)[::-1][:10]:
    print(f"{columns[i]:<30} Importance: {imp[i]:.6f}")


# save model
booster = model.get_booster()
booster.save_model("XGBoost_SittingStanding/XGB_SS.json")

# save best iteration
best_iter = f1_logger.best_iteration + 1
meta = {
    "best_iteration": int(best_iter)
}
with open("XGBoost_SittingStanding/XGB_SS_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

######################################### TESTING ###################################
dtest = DMatrix(X_test)
y_prob = booster.predict(
    dtest,
    iteration_range=(0, best_iter)
)
y_pred = (y_prob > 0.50).astype(int)

print("\n\U0001f9ea Overall Test Results — Combined Subjects")
print(f" Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")
print(pd.DataFrame(classification_report(y_test, y_pred, digits=3, output_dict=True)).T)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# ---- Per-subject results on test ----
print("\n🧪 Per-Subject Results (C1*):")
test_subjects = df_test["subject"].astype(str).values
unique_subj = np.unique(test_subjects)

per_subj_rows = []
for sbj in unique_subj:
    mask = (test_subjects == sbj)
    y_true_s = y_test[mask]
    y_pred_s = y_pred[mask]
    acc_s = balanced_accuracy_score(y_true_s, y_pred_s)
    print("Confusion Matrix:\n", confusion_matrix(y_true_s, y_pred_s))
    per_subj_rows.append({"subject": sbj, "n_windows": int(mask.sum()), "balanced_acc": acc_s})

per_subj_df = pd.DataFrame(per_subj_rows).sort_values("subject").reset_index(drop=True)
print(per_subj_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


