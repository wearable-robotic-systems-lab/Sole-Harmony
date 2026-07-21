"""
ext_CNNbiLSTM.py

Trains a CNN-BiLSTM human activity recognition (HAR) model on the
Sole-HARmony insole data using the full IMU + FSR channel set, with a
disk-backed per-subject cache so repeated runs skip re-parsing the raw
.mat files.

Pipeline (two stages):
    Stage 1 - Grid search: for every hyperparameter config, run stratified
    k-fold CV on the training subjects. The selection metric is macro F1
    (val_f1); balanced accuracy (val_ba) is also computed and reported
    alongside it, but does not influence model/config selection. Results
    for every config are written to --search-results-csv.

    Stage 2 - Final fit: take the config with the best mean CV F1, fit it
    once on an 80/20 train/val split of the training subjects, evaluate on
    the held-out test subjects, and write model weights, loss/metric
    curves, and prediction CSVs to --output-root, plus a one-row summary
    to --final-results-csv.

Per-subject caching:
    Windowed (X, Y, Meta) arrays for each subject are cached to
    --cache-dir as .npy files and reloaded via mmap_mode="r" on later
    runs, so re-running the script doesn't require re-parsing the MATLAB
    session files for subjects already processed. Use --rebuild-cache to
    force a rebuild (e.g. after a preprocessing change).


Usage:
    python ext_CNNbiLSTM.py \
        --dataset-root /path/to/SoleHARmony_Dataset \
        --output-root ./DL_IMUFSR \
        --search-results-csv HPARAM_SEARCH_fsr.csv \
        --final-results-csv RESULTS_imufsr.csv

    # Force cache rebuild after changing preprocessing:
    python ext_CNNbiLSTM.py --rebuild-cache

"""

import argparse
import bisect
import gc
import os
import random

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

torch.multiprocessing.set_sharing_strategy("file_system")

FS = 270.0  # insole sampling frequency (Hz)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_DATASET_ROOT = os.path.join(ROOT_DIR, "SoleHARmony_Dataset")
DEFAULT_CACHE_DIR = os.path.join(SCRIPT_DIR, "subject_cache_imufsr")

DEFAULT_TRAIN_SUBJECTS = [f"C{i:03d}" for i in range(1, 11)]  # C001-C010
DEFAULT_TEST_SUBJECTS = [f"C{i:03d}" for i in range(11, 14)]  # C011-C013

# Hyperparameter grid searched over (cnn_filters, cnn_kernel, lstm_hidden,
# lstm_layers, dropout). Restrict with --quick-test for a fast smoke test.
HYPERPARAM_GRID = {
    "cnn_filters": [32, 64],
    "cnn_kernel": [3, 5],
    "lstm_hidden": [64, 128],
    "lstm_layers": [1, 2],
    "dropout": [0.1, 0.2],
}

# Module-level cache settings; overridden from CLI args in main().
CACHE_DIR = DEFAULT_CACHE_DIR
REBUILD_CACHE = False


def build_base_config(args: argparse.Namespace) -> dict:
    return {
        "n_splits_train": args.n_splits,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "dropout": 0.5,
        "cnn_filters": 64,
        "cnn_kernel": 5,
        "lstm_hidden": 128,
        "lstm_layers": 1,
        "bidirectional": True,
        "patience": args.patience,
        "max_epochs": args.max_epochs,
        "random_state": args.seed,
        "early_stopping_mode": "max",  # "max" for val F1, "min" for val loss
    }


# =========================
# -------- Dataset --------
# =========================


class MotionDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X).float()  # (N, C, T)
        self.Y = torch.from_numpy(Y).long()  # (N,)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# =========================
# -------- Model ----------
# =========================


class CNNLSTMNet(nn.Module):
    """CNN feature extractor + BiLSTM temporal model for windowed IMU+FSR data."""

    def __init__(
        self,
        n_channels,
        num_classes,
        cnn_filters=64,
        cnn_kernel=5,
        lstm_hidden=128,
        lstm_layers=1,
        bidirectional=True,
        dropout=0.5,
    ):
        super().__init__()

        k = cnn_kernel
        p = k // 2

        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, cnn_filters, kernel_size=k, padding=p),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(cnn_filters, cnn_filters, kernel_size=k, padding=p),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,  # (B, T', F)
            bidirectional=bidirectional,
        )

        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, C, T)
        feat = self.cnn(x)  # (B, F, T')
        feat = feat.permute(0, 2, 1)  # (B, T', F)

        _, (h_n, _) = self.lstm(feat)

        if self.bidirectional:
            h_forward = h_n[-2, :, :]
            h_backward = h_n[-1, :, :]
            last = torch.cat([h_forward, h_backward], dim=1)  # (B, 2H)
        else:
            last = h_n[-1, :, :]  # (B, H)

        return self.classifier(last)


class EarlyStopping:
    """Tracks the best validation metric and restores the best model state.

    mode="max" monitors a metric where higher is better (val F1 here).
    mode="min" monitors a metric where lower is better (val loss).
    """

    def __init__(self, patience=10, mode="max", delta=0.0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, metric, model):
        if self.best_score is None:
            self.best_score = metric
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return

        if self.mode == "max":
            improvement = metric > self.best_score + self.delta
        else:
            improvement = metric < self.best_score - self.delta

        if improvement:
            self.best_score = metric
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# =========================
# -------- Train / Eval ---
# =========================


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y_batch).sum().item()
        total += X_batch.size(0)

    return total_loss / total, total_correct / total


def eval_one_epoch(model, loader, criterion, device):
    """Returns (val_loss, val_balanced_accuracy, val_macro_f1).

    Macro F1 is the model-selection / early-stopping metric; balanced
    accuracy is computed for reporting only.
    """
    model.eval()
    total_loss, total = 0.0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            total += X_batch.size(0)

            y_true.append(y_batch.cpu().numpy())
            y_pred.append(logits.argmax(dim=1).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    val_loss = total_loss / total
    val_ba = balanced_accuracy_score(y_true, y_pred)
    val_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return val_loss, val_ba, val_f1


def collect_preds_and_labels(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            preds = model(X_batch).argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    return np.concatenate(all_labels, axis=0), np.concatenate(all_preds, axis=0)


# =========================
# -------- Loading ---------
# =========================


def _mat_struct_to_dict(obj):
    """Recursively convert a scipy MATLAB struct object into a plain dict."""
    out = {}
    for name in obj._fieldnames:
        val = getattr(obj, name)
        if isinstance(val, sio.matlab.mio5_params.mat_struct):
            out[name] = _mat_struct_to_dict(val)
        else:
            out[name] = val
    return out


def load_insole_mat(file_name: str) -> dict | None:
    """Load a DataStruct.mat session file, handling both classic (<v7.3)
    and HDF5-based (v7.3+) MATLAB file formats."""
    try:
        raw = sio.loadmat(file_name, squeeze_me=True, struct_as_record=False, mat_dtype=True)
        if "DataStruct" not in raw:
            return None
        return _mat_struct_to_dict(raw["DataStruct"])
    except NotImplementedError:
        pass  # v7.3 file; fall through to h5py loader

    with h5py.File(file_name, "r") as f:
        key = next(k for k in f.keys() if not k.startswith("#"))
        grp = f[key]

        def read_grp(g):
            out = {}
            for k, v in g.items():
                out[k] = np.squeeze(np.array(v)) if isinstance(v, h5py.Dataset) else read_grp(v)
            return out

        return read_grp(grp)


def unify_insole_structure(D: dict) -> dict:
    """Extract timestamps and stack IMU + FSR channels into per-sensor arrays."""
    return {
        "t": np.asarray(D["t_ms"], dtype=float),
        "lin_acc": np.column_stack([D["lin_acc_x"], D["lin_acc_y"], D["lin_acc_z"]]),
        "gyr": np.column_stack([D["gyr_x"], D["gyr_y"], D["gyr_z"]]),
        "fsr": np.column_stack(
            [
                D["fsr_Hallux"],
                D["fsr_Toes"],
                D["fsr_Met1"],
                D["fsr_Met3"],
                D["fsr_Met5"],
                D["fsr_Arch"],
                D["fsr_HeelL"],
                D["fsr_HeelR"],
            ]
        ),
    }


def load_from_mat(mat_path: str):
    """Load one session file and return (labels_df, left_insole, right_insole)."""
    data = load_insole_mat(mat_path)
    if data is None:
        return None, None, None

    arr = np.squeeze(np.array(data["labelsCam"]))
    # MATLAB may store labels as (3, N) instead of (N, 3)
    if arr.ndim == 2 and arr.shape[0] == 3 and arr.shape[1] != 3:
        arr = arr.T

    labels = (
        pd.DataFrame(arr, columns=["label", "start_t_ms", "end_t_ms"])
        .sort_values("start_t_ms")
        .reset_index(drop=True)
    )

    DL = unify_insole_structure(data["InsoleL"])
    DR = unify_insole_structure(data["InsoleR"])

    return labels, DL, DR


# =========================
# ---- Preprocessing -------
# =========================


def impute_broken_fsr(F: np.ndarray) -> np.ndarray:
    """Fill fully-NaN FSR channels using anatomically motivated donor
    channels (contralateral heel/forefoot sensor, or the mean of the
    remaining forefoot channels).

    F: (T, 8) array, columns = Hallux, Toes, Met1, Met3, Met5, Arch, HeelL, HeelR.
    """
    F = F.copy()
    fsr_names = ["Hallux", "Toes", "Met1", "Met3", "Met5", "Arch", "HeelL", "HeelR"]

    for i in range(F.shape[1]):
        if not np.all(np.isnan(F[:, i])):
            continue  # not fully broken, skip

        name = fsr_names[i]
        print(f"    Imputing {name} (fully NaN)...")

        if name == "HeelL":
            donor = F[:, 7]  # HeelR
        elif name == "HeelR":
            donor = F[:, 6]  # HeelL
        elif name == "Hallux":
            donor = F[:, 1]  # Toes
        elif name == "Toes":
            donor = F[:, 0]  # Hallux
        elif name in ("Met1", "Met3", "Met5", "Arch"):
            good = [j for j in range(8) if not np.all(np.isnan(F[:, j]))]
            F[:, i] = np.nanmean(F[:, good], axis=1) if good else 0.0
            continue
        else:
            continue

        if np.all(np.isnan(donor)):
            print(f"    Warning: donor channel for {name} also NaN, using zeros")
            F[:, i] = 0.0
        else:
            F[:, i] = donor

    return F


def butter_lowpass_filter_array(x, cutoff_hz, fs_hz, order=4):
    x = np.asarray(x)
    if x.size == 0:
        return x
    nyq = 0.5 * fs_hz
    wn = min(cutoff_hz / nyq, 0.999)
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, x)


def global_filter_and_normalize(data: dict | None) -> dict | None:
    """Low-pass filter (10 Hz cutoff) and normalize each sensor group:
    IMU channels are per-channel z-scored; FSR channels are imputed for
    fully-broken sensors, NaN-filled per-channel with the channel median,
    then normalized by a shared 99.9th-percentile value across all FSR
    channels (so relative loading between sensors is preserved).
    """
    if data is None:
        return None

    for key in ("lin_acc", "gyr"):
        if key not in data:
            continue
        A = data[key]
        Af = np.column_stack([butter_lowpass_filter_array(A[:, i], 10, FS) for i in range(A.shape[1])])
        Af = (Af - Af.mean(axis=0)) / (Af.std(axis=0) + 1e-8)
        data[key] = Af

    if "fsr" in data:
        F = impute_broken_fsr(data["fsr"].copy())

        for i in range(F.shape[1]):
            chan = F[:, i]
            if np.any(np.isnan(chan)):
                finite = np.isfinite(chan)
                if finite.sum() >= 2:
                    chan[~finite] = np.median(chan[finite])
                else:
                    chan[:] = 0
            F[:, i] = chan

        Ff = np.column_stack([butter_lowpass_filter_array(F[:, i], 10, FS) for i in range(F.shape[1])])

        finite = np.isfinite(Ff)
        if finite.sum() >= 2:
            p99 = np.nanpercentile(Ff[finite], 99.9)
            Fnorm = Ff / p99 if p99 > 0 else np.zeros_like(Ff)
        else:
            Fnorm = np.zeros_like(Ff)

        data["fsr"] = Fnorm

    return data


# =========================
# ---- Windows & Labels ----
# =========================


def get_effective_overlap(labels_df, DL, DR, side):
    cam_start = labels_df["start_t_ms"].iloc[0]
    cam_end = labels_df["end_t_ms"].iloc[-1]

    starts, ends = [cam_start], [cam_end]
    if side in ("left", "both") and DL is not None:
        starts.append(int(DL["t"][0]))
        ends.append(int(DL["t"][-1]))
    if side in ("right", "both") and DR is not None:
        starts.append(int(DR["t"][0]))
        ends.append(int(DR["t"][-1]))

    return max(starts), min(ends)


def build_windows(labels_df, DL, DR, side, window_sec, overlap_sec):
    start_eff, end_eff = get_effective_overlap(labels_df, DL, DR, side)
    w_ms = int(window_sec * 1000)
    step = int((window_sec - overlap_sec) * 1000)
    return [(t, t + w_ms) for t in range(int(start_eff), int(end_eff - w_ms) + 1, step)]


def window_label_if_constant(labels_df, t0, t1):
    """Return the label for [t0, t1) only if a single, non-ignored label
    covers the entire window; otherwise return None to drop the window."""
    overlapping = labels_df[~((labels_df["end_t_ms"] <= t0) | (labels_df["start_t_ms"] >= t1))]
    if overlapping.empty:
        return None

    unique_labels = overlapping["label"].unique()
    if len(unique_labels) == 1 and unique_labels[0] != -1:
        return int(unique_labels[0])

    return None


# =========================================================================
# ---- Disk-backed subject cache (memory-mapped, not fully loaded in RAM) --
# =========================================================================


def _cache_paths_for_subject(sbj: str):
    base = os.path.join(CACHE_DIR, sbj)
    return base + "_X.npy", base + "_Y.npy", base + "_Meta.npy"


def load_subject_cache(sbj: str):
    """Returns memory-mapped X/Y (mmap_mode='r') and a fully-loaded Meta
    array for a cached subject, or None if not cached / REBUILD_CACHE is set."""
    if REBUILD_CACHE:
        for p in _cache_paths_for_subject(sbj):
            if os.path.exists(p):
                os.remove(p)
        return None

    xp, yp, mp = _cache_paths_for_subject(sbj)
    if not (os.path.exists(xp) and os.path.exists(yp) and os.path.exists(mp)):
        return None
    try:
        X = np.load(xp, mmap_mode="r")
        Y = np.load(yp, mmap_mode="r")
        Meta = np.load(mp, allow_pickle=True)
        print(f"    Memory-mapped cached subject {sbj} from {xp} (no full RAM load)")
        return X, Y, Meta
    except Exception as e:
        print(f"    Failed to load cache for {sbj}: {e}")
        return None


def save_subject_cache(sbj: str, X: np.ndarray, Y: np.ndarray, Meta: np.ndarray):
    xp, yp, mp = _cache_paths_for_subject(sbj)
    try:
        np.save(xp, X)
        np.save(yp, Y)
        np.save(mp, Meta)
        print(f"    Saved cache for subject {sbj} -> {xp}")
    except Exception as e:
        print(f"    Failed to save cache for {sbj}: {e}")


def data_prepare_for_subject(dataset_root: str, sbj: str):
    """Load and window all sessions for one subject. Returns memory-mapped
    X/Y (building + caching them first if not already cached)."""
    cached = load_subject_cache(sbj)
    if cached is not None:
        return cached

    sbj_path = os.path.join(dataset_root, sbj)
    X_all, Y_all, Meta_all = [], [], []

    for folder in sorted(os.listdir(sbj_path)):  # e.g. C001_1658420327
        folder_path = os.path.join(sbj_path, folder)
        session_file = os.path.join(folder_path, "DataStruct.mat")

        if not os.path.exists(session_file):
            print(f"No DataStruct.mat found in {folder}, skipping...")
            continue

        print(f"\nProcessing {folder}...")

        labels, DL, DR = load_from_mat(session_file)
        if labels is None or DL is None or DR is None:
            continue

        parts = folder.split("_", 1)
        subj_id = parts[0]
        session_id = parts[1] if len(parts) > 1 else folder

        DL = global_filter_and_normalize(DL)
        DR = global_filter_and_normalize(DR)

        windows = build_windows(labels, DL, DR, side="both", window_sec=3, overlap_sec=1.5)
        target_len = int(3 * FS)

        X_subj, Y_subj, Meta_subj = [], [], []

        for t0, t1 in windows:
            lbl = window_label_if_constant(labels, t0, t1)
            if lbl is None:
                continue

            tL, tR = DL["t"], DR["t"]
            if t0 < tL[0] or t1 > tL[-1] or t0 < tR[0] or t1 > tR[-1]:
                continue

            i0L = bisect.bisect_left(tL, t0)
            i0R = bisect.bisect_left(tR, t0)
            if i0L + target_len > len(tL) or i0R + target_len > len(tR):
                continue

            x_left = np.concatenate(
                [
                    DL["lin_acc"][i0L : i0L + target_len],
                    DL["gyr"][i0L : i0L + target_len],
                    DL["fsr"][i0L : i0L + target_len],
                ],
                axis=1,
            )
            x_right = np.concatenate(
                [
                    DR["lin_acc"][i0R : i0R + target_len],
                    DR["gyr"][i0R : i0R + target_len],
                    DR["fsr"][i0R : i0R + target_len],
                ],
                axis=1,
            )
            x_window = np.concatenate([x_left, x_right], axis=1)  # (T, 2C)

            X_subj.append(x_window.astype(np.float32))
            Y_subj.append(lbl)
            Meta_subj.append((subj_id, session_id))

        if not X_subj:
            continue

        X_subj = np.stack(X_subj, axis=0).transpose(0, 2, 1)  # (N, 2C, T)
        X_all.append(X_subj)
        Y_all.append(np.array(Y_subj))
        Meta_all.extend(Meta_subj)

    X_concat = np.concatenate(X_all)
    Y_concat = np.concatenate(Y_all)
    Meta_array = np.array(Meta_all)

    save_subject_cache(sbj, X_concat, Y_concat, Meta_array)

    # Free the in-RAM arrays and reload as memmaps so every subject -- fresh
    # or cached -- behaves identically from here on.
    del X_concat, Y_concat
    gc.collect()
    xp, yp, _ = _cache_paths_for_subject(sbj)
    X_mm = np.load(xp, mmap_mode="r")
    Y_mm = np.load(yp, mmap_mode="r")
    return X_mm, Y_mm, Meta_array


def data_prepare(dataset_root: str, subject_list: list[str]):
    """Builds/loads per-subject caches, then materializes the concatenated
    array for the given subject_list into RAM."""
    X_all, Y_all, Meta_all = [], [], []

    for sbj in subject_list:
        X, Y, Meta = data_prepare_for_subject(dataset_root, sbj)
        print(f"    {sbj}: {X.shape[0]} windows, shape {X.shape} (memmapped)")
        X_all.append(np.asarray(X))  # materialize this subject's data into RAM
        Y_all.append(np.asarray(Y))
        Meta_all.append(np.asarray(Meta))

    return np.concatenate(X_all), np.concatenate(Y_all), np.concatenate(Meta_all)


# =========================
# ---- Hyperparam search ---
# =========================


def run_kfold_for_config(X, Y, config, device, fold_splits, num_workers):
    """Run stratified k-fold CV for one hyperparameter config.

    Selection metric: mean macro F1 across folds (kfold_mean_val_f1).
    Balanced accuracy is also tracked per fold, from the same best epoch
    that F1 was measured at, purely for reporting (kfold_mean_val_ba).
    """
    dataset = MotionDataset(X, Y)
    n_channels = X.shape[1]
    num_classes = len(np.unique(Y))

    use_cuda = device.type == "cuda"
    loader_kwargs = {
        "batch_size": config["batch_size"],
        "num_workers": num_workers,
        "pin_memory": use_cuda,
    }

    fold_val_f1s, fold_val_bas, fold_train_accs = [], [], []

    for fold, (train_idx, val_idx) in enumerate(fold_splits, start=1):
        print(f"  Fold {fold}/{len(fold_splits)}...")

        train_loader = DataLoader(Subset(dataset, train_idx), shuffle=True, **loader_kwargs)
        val_loader = DataLoader(Subset(dataset, val_idx), shuffle=False, **loader_kwargs)

        model = CNNLSTMNet(
            n_channels=n_channels,
            num_classes=num_classes,
            cnn_filters=config["cnn_filters"],
            cnn_kernel=config["cnn_kernel"],
            lstm_hidden=config["lstm_hidden"],
            lstm_layers=config["lstm_layers"],
            bidirectional=config["bidirectional"],
            dropout=config["dropout"],
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        early_stopper = EarlyStopping(patience=config["patience"], mode=config["early_stopping_mode"])

        best_val_f1, best_val_ba, best_train_acc = 0.0, 0.0, 0.0

        for epoch in range(1, config["max_epochs"] + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_ba, val_f1 = eval_one_epoch(model, val_loader, criterion, device)

            print(
                f"    Epoch {epoch:03d}: train_acc={train_acc:.3f}, val_f1={val_f1:.3f}, "
                f"val_ba={val_ba:.3f}, train_loss={train_loss:.3f}, val_loss={val_loss:.3f}, "
                f"ES counter={early_stopper.counter}/{early_stopper.patience}"
            )

            metric_for_es = val_f1 if config["early_stopping_mode"] == "max" else val_loss
            early_stopper(metric_for_es, model)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_ba = val_ba
                best_train_acc = train_acc

            if early_stopper.early_stop:
                print(f"    Early stop at epoch {epoch} (no improvement for {early_stopper.patience} epochs)")
                break

        fold_val_f1s.append(best_val_f1)
        fold_val_bas.append(best_val_ba)
        fold_train_accs.append(best_train_acc)

        del model, optimizer, criterion, train_loader, val_loader
        if use_cuda:
            torch.cuda.empty_cache()

    mean_val_f1 = float(np.mean(fold_val_f1s))
    mean_val_ba = float(np.mean(fold_val_bas))
    mean_train_acc = float(np.mean(fold_train_accs))

    print(
        f"{config['n_splits_train']}-fold: Train ACC = {mean_train_acc:.4f}, "
        f"Val F1 = {mean_val_f1:.4f}, Val BA = {mean_val_ba:.4f}"
    )

    result = {
        "kfold_mean_val_f1": mean_val_f1,
        "kfold_mean_val_ba": mean_val_ba,
        "kfold_mean_train_acc": mean_train_acc,
    }
    for i, v in enumerate(fold_val_f1s):
        result[f"kfold_val_f1_fold_{i}"] = float(v)
    for i, v in enumerate(fold_val_bas):
        result[f"kfold_val_ba_fold_{i}"] = float(v)
    for i, v in enumerate(fold_train_accs):
        result[f"kfold_train_acc_fold_{i}"] = float(v)

    return result


def train_single_model_and_test(
    X_train_array,
    Y_train_array,
    X_test_array,
    Y_test_array,
    Meta_train,
    Meta_test,
    config,
    device,
    save_path,
    num_workers,
    val_ratio=0.2,
):
    """Fit one model on an 80/20 train/val split of the training subjects
    (selecting the best epoch by val F1), evaluate on the held-out test
    subjects, and save model/plots/predictions to save_path."""

    indices = np.arange(len(Y_train_array))
    idx_tr, idx_val = train_test_split(indices, test_size=val_ratio, stratify=Y_train_array, random_state=42)

    X_tr, X_val = X_train_array[idx_tr], X_train_array[idx_val]
    y_tr, y_val = Y_train_array[idx_tr], Y_train_array[idx_val]
    Meta_tr, Meta_val = Meta_train[idx_tr], Meta_train[idx_val]

    train_ds = MotionDataset(X_tr, y_tr)
    val_ds = MotionDataset(X_val, y_val)
    test_ds = MotionDataset(X_test_array, Y_test_array)

    use_cuda = device.type == "cuda"
    loader_kwargs = {
        "batch_size": config["batch_size"],
        "num_workers": num_workers,
        "pin_memory": use_cuda,
    }

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    n_channels = X_train_array.shape[1]
    num_classes = len(np.unique(Y_train_array))

    model = CNNLSTMNet(
        n_channels=n_channels,
        num_classes=num_classes,
        cnn_filters=config["cnn_filters"],
        cnn_kernel=config["cnn_kernel"],
        lstm_hidden=config["lstm_hidden"],
        lstm_layers=config["lstm_layers"],
        bidirectional=config["bidirectional"],
        dropout=config["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    early_stopper = EarlyStopping(patience=config["patience"], mode=config["early_stopping_mode"])

    best_val_f1, best_val_ba = 0.0, 0.0
    best_train_acc = best_val_loss = best_train_loss = 0.0
    best_epoch = 0

    train_loss_list, train_acc_list = [], []
    val_loss_list, val_f1_list, val_ba_list = [], [], []

    for epoch in range(1, config["max_epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_ba, val_f1 = eval_one_epoch(model, val_loader, criterion, device)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_f1_list.append(val_f1)
        val_ba_list.append(val_ba)

        print(
            f"[Epoch {epoch:03d}] Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
            f"Val Loss={val_loss:.4f} F1={val_f1:.4f} BA={val_ba:.4f} | "
            f"ES counter={early_stopper.counter}/{early_stopper.patience}"
        )

        metric_for_es = val_f1 if config["early_stopping_mode"] == "max" else val_loss
        early_stopper(metric_for_es, model)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_ba = val_ba
            best_train_acc = train_acc
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_epoch = epoch

        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch} (no improvement for {early_stopper.patience} epochs)")
            break

    if early_stopper.best_state is not None:
        model.load_state_dict(early_stopper.best_state)
        print(f"Restored best model with val_f1 = {early_stopper.best_score:.4f}")

    os.makedirs(save_path, exist_ok=True)

    model_save_path = os.path.join(save_path, "model.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved best model to: {model_save_path}")

    epochs = range(1, len(train_loss_list) + 1)
    best_epoch_to_plot = best_epoch if best_epoch > 0 else len(train_loss_list)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss_list, label="Train Loss")
    plt.plot(epochs, val_loss_list, label="Val Loss")
    plt.axvline(best_epoch_to_plot, color="red", linestyle="--", label=f"Best Val @ {best_epoch_to_plot}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "loss_curve.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_acc_list, label="Train Acc")
    plt.plot(epochs, val_f1_list, label="Val F1")
    plt.plot(epochs, val_ba_list, label="Val BA")
    plt.axvline(best_epoch_to_plot, color="red", linestyle="--", label=f"Best Val @ {best_epoch_to_plot}")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("F1 / BA Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "metric_curve.png"))
    plt.close()

    print(f"Saved loss_curve.png and metric_curve.png to: {save_path}")

    test_loss, test_ba, test_f1 = eval_one_epoch(model, test_loader, criterion, device)
    print(f"[TEST] Loss={test_loss:.4f} BA={test_ba:.4f} F1={test_f1:.4f}")

    train_y_true, train_y_pred = collect_preds_and_labels(model, train_loader, device)
    val_y_true, val_y_pred = collect_preds_and_labels(model, val_loader, device)
    test_y_true, test_y_pred = collect_preds_and_labels(model, test_loader, device)

    train_df = pd.DataFrame(
        {"subject": Meta_tr[:, 0], "session": Meta_tr[:, 1], "y_true": train_y_true, "y_pred": train_y_pred}
    )
    val_df = pd.DataFrame(
        {"subject": Meta_val[:, 0], "session": Meta_val[:, 1], "y_true": val_y_true, "y_pred": val_y_pred}
    )
    test_df = pd.DataFrame(
        {"subject": Meta_test[:, 0], "session": Meta_test[:, 1], "y_true": test_y_true, "y_pred": test_y_pred}
    )

    train_pred_path = os.path.join(save_path, "train_predictions.csv")
    val_pred_path = os.path.join(save_path, "val_predictions.csv")
    test_pred_path = os.path.join(save_path, "test_predictions.csv")

    train_df.to_csv(train_pred_path, index=False)
    val_df.to_csv(val_pred_path, index=False)
    test_df.to_csv(test_pred_path, index=False)

    print(f"Saved train/val/test predictions to:\n  {train_pred_path}\n  {val_pred_path}\n  {test_pred_path}")

    del model, optimizer, criterion, train_loader, val_loader, test_loader
    if use_cuda:
        torch.cuda.empty_cache()

    return {
        # --- selection metric (F1) ---
        "best_train_acc": float(best_train_acc),
        "best_val_f1": float(best_val_f1),
        "test_f1": float(test_f1),
        # --- reported metric (BA) ---
        "best_val_ba": float(best_val_ba),
        "test_ba": float(test_ba),
        # --- loss ---
        "best_train_loss": float(best_train_loss),
        "best_val_loss": float(best_val_loss),
        "test_loss": float(test_loss),
        # --- model & figures ---
        "model_path": model_save_path,
        "loss_curve": os.path.join(save_path, "loss_curve.png"),
        "metric_curve": os.path.join(save_path, "metric_curve.png"),
    }


# =========================
# -------- CLI -------------
# =========================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3-subject holdout DL runner: IMU+FSR CNN-BiLSTM, F1-selected grid search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, help="Path to SoleHARmony_Dataset directory.")
    p.add_argument(
        "--output-root",
        default=os.path.join(SCRIPT_DIR, "DL_IMUFSR"),
        help="Directory to write the final model/plot/prediction outputs.",
    )
    p.add_argument(
        "--search-results-csv", default="HPARAM_SEARCH_fsr.csv", help="Path to write per-config grid search results."
    )
    p.add_argument(
        "--final-results-csv", default="RESULTS_imufsr.csv", help="Path to write the final best-config test results."
    )
    p.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help="Per-subject cache directory. Do not reuse a cache dir from a "
        "different feature set (e.g. the LOOCV FSR pipeline) -- shapes are "
        "not checked on load.",
    )
    p.add_argument("--rebuild-cache", action="store_true", help="Delete and rebuild subject caches from .mat files.")
    p.add_argument(
        "--train-subjects",
        nargs="+",
        default=DEFAULT_TRAIN_SUBJECTS,
        help="Subject IDs used for k-fold CV and the 80/20 train/val split.",
    )
    p.add_argument("--test-subjects", nargs="+", default=DEFAULT_TEST_SUBJECTS, help="Held-out test subject IDs.")
    p.add_argument("--n-splits", type=int, default=3, help="Number of stratified k-folds.")
    p.add_argument("--max-epochs", type=int, default=30, help="Max training epochs per fold / final fit.")
    p.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=16, help="DataLoader worker processes.")
    p.add_argument("--val-ratio", type=float, default=0.2, help="Held-out validation fraction for the final fit.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a single hyperparameter config with 2 max epochs, for a fast pipeline smoke test.",
    )
    return p.parse_args()


def main():
    global CACHE_DIR, REBUILD_CACHE

    args = parse_args()

    CACHE_DIR = args.cache_dir
    REBUILD_CACHE = args.rebuild_cache
    os.makedirs(CACHE_DIR, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Cache dir: {CACHE_DIR}")
    print(f"  Rebuild cache: {REBUILD_CACHE}")
    print("=" * 60)

    print("Preparing training data...")
    X_train, Y_train, Meta_train = data_prepare(args.dataset_root, args.train_subjects)
    print("Preparing test data...")
    X_test, Y_test, Meta_test = data_prepare(args.dataset_root, args.test_subjects)

    base_config = build_base_config(args)
    if args.quick_test:
        base_config["max_epochs"] = min(2, args.max_epochs)

    kf = StratifiedKFold(n_splits=base_config["n_splits_train"], shuffle=True, random_state=args.seed)
    fold_splits = list(kf.split(X_train, Y_train))

    grid = HYPERPARAM_GRID
    filters_opts = grid["cnn_filters"][:1] if args.quick_test else grid["cnn_filters"]
    kernel_opts = grid["cnn_kernel"][:1] if args.quick_test else grid["cnn_kernel"]
    hidden_opts = grid["lstm_hidden"][:1] if args.quick_test else grid["lstm_hidden"]
    layer_opts = grid["lstm_layers"][:1] if args.quick_test else grid["lstm_layers"]
    dropout_opts = grid["dropout"][:1] if args.quick_test else grid["dropout"]

    # ---- Stage 1: grid search (CV only, no test-set training here) ----
    search_results = []

    for f in filters_opts:
        for k in kernel_opts:
            for h in hidden_opts:
                for l in layer_opts:
                    for dr in dropout_opts:
                        config = base_config.copy()
                        config.update(cnn_filters=f, cnn_kernel=k, lstm_hidden=h, lstm_layers=l, dropout=dr)

                        folder_name = f"F{f}_K{k}_H{h}_L{l}_DR{dr}"
                        print(f"\n[SEARCH] config = {folder_name}")

                        kfold_result = run_kfold_for_config(
                            X_train, Y_train, config, device, fold_splits, num_workers=args.num_workers
                        )
                        gc.collect()
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                        search_results.append(
                            {
                                "cnn_filters": f,
                                "cnn_kernel": k,
                                "lstm_hidden": h,
                                "lstm_layers": l,
                                "dropout": dr,
                                "config_tag": folder_name,
                                **kfold_result,
                            }
                        )

                        # Write incrementally so a partial run isn't lost.
                        pd.DataFrame(search_results).to_csv(args.search_results_csv, index=False)

    search_df = pd.DataFrame(search_results)

    # ---- Stage 2: pick the best config by mean CV F1, fit once, test once ----
    best_row = search_df.loc[search_df["kfold_mean_val_f1"].idxmax()]
    best_tag = best_row["config_tag"]
    print(
        f"\n{'=' * 60}\nBest config by inner-CV F1: {best_tag} "
        f"(kfold_mean_val_f1={best_row['kfold_mean_val_f1']:.4f}, "
        f"kfold_mean_val_ba={best_row['kfold_mean_val_ba']:.4f})\n{'=' * 60}"
    )

    best_config = base_config.copy()
    best_config.update(
        cnn_filters=int(best_row["cnn_filters"]),
        cnn_kernel=int(best_row["cnn_kernel"]),
        lstm_hidden=int(best_row["lstm_hidden"]),
        lstm_layers=int(best_row["lstm_layers"]),
        dropout=float(best_row["dropout"]),
    )

    save_path = os.path.join(args.output_root, best_tag)

    final_metrics = train_single_model_and_test(
        X_train,
        Y_train,
        X_test,
        Y_test,
        Meta_train,
        Meta_test,
        best_config,
        device,
        save_path=save_path,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    final_result = {
        "config_tag": best_tag,
        "cnn_filters": best_config["cnn_filters"],
        "cnn_kernel": best_config["cnn_kernel"],
        "lstm_hidden": best_config["lstm_hidden"],
        "lstm_layers": best_config["lstm_layers"],
        "dropout": best_config["dropout"],
        "kfold_mean_val_f1": best_row["kfold_mean_val_f1"],
        "kfold_mean_val_ba": best_row["kfold_mean_val_ba"],
        **final_metrics,
    }
    pd.DataFrame([final_result]).to_csv(args.final_results_csv, index=False)

    print("\nFinal (best-CV-F1) config test results:")
    print(f"  Test F1 = {final_result['test_f1']:.4f}")
    print(f"  Test BA = {final_result['test_ba']:.4f}")


if __name__ == "__main__":
    main()
