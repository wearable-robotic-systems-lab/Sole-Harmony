"""
Sole-HARmony feature extraction pipeline

Author: Laura De Marzi (Stevens Institute of Technology, WRSLab)

Extracts time- and frequency-domain features from multimodal insole data
(IMU + FSR) using sliding windows.

Input:  DataStruct.mat (per session)
Output: FeatureTables/{session}_features.csv
"""

import bisect
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
from scipy.stats import entropy
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import scipy.io as sio
import h5py


def load_insole_mat(file_name):
    """
    Loads MATLAB file saved as:
        save('DataStruct.mat', 'Data')

    with fields:
        DataStruct.InsoleL
        DataStruct.InsoleR
        DataStruct.labelsCam
        DataStruct.labelsAP
    """

    # ---- HDF5 loader for v7.3 ----
    try:
        with h5py.File(file_name, "r") as f:
            # Autodetect group name
            keys = list(f.keys())   # e.g. ['#refs#', 'DataStruct']
            struct_key = None
            for k in keys:
                if not k.startswith("#"):
                    struct_key = k
                    break

            if struct_key is None:
                print(f"❌ No valid struct found in {file_name}")
                return None

            grp = f[struct_key]

            # Minimal HDF reader: convert datasets recursively
            def read_hdf_group(g):
                out = {}
                for k, v in g.items():
                    if isinstance(v, h5py.Dataset):
                        arr = np.array(v)
                        arr = np.squeeze(arr)
                        out[k] = np.array(arr)
                    elif isinstance(v, h5py.Group):
                        out[k] = read_hdf_group(v)
                return out

            data = read_hdf_group(grp)

        # Convert labels to DataFrames
        for lbl_name in ("labelsCam", "labelsAP"):
            if lbl_name in data:
                arr = np.array(data[lbl_name])
                arr = np.squeeze(arr)

                if arr.ndim == 2 and arr.shape[0] == 3 and arr.shape[1] != 3:
                    arr = arr.T

                data[lbl_name] = pd.DataFrame(
                    arr, columns=["label", "start_t_ms", "end_t_ms"]
                )

        return data
    
    except Exception as e:
        print(f" HDF5 loader failed: {e}")
        return None

    


def load_label_and_data(path_mat):
    data = load_insole_mat(path_mat)
    if data is None:
        print(f" Skipping {session} (no valid .mat file).")
        return None, None, None, None

    DL = unify_insole_structure(data["InsoleL"])
    DR = unify_insole_structure(data["InsoleR"])

    labels_cam = data["labelsCam"]
    labels_AP  = data.get("labelsAP", None)

    return labels_cam, labels_AP, DL, DR

def unify_insole_structure(D):
    """
    Convert MATLAB InsoleL/InsoleR struct to a unified Python format:

    {
        't'       : array (N,)
        'lin_acc' : Nx3 matrix
        'gyr'     : Nx3 matrix
        'fsr'     : Nx8 matrix
    }
    """
    out = {}

    # time
    out['t'] = np.array(D['t_ms'], dtype=float)

    # linear acceleration 
    out['lin_acc'] = np.column_stack([
        D['lin_acc_x'], D['lin_acc_y'], D['lin_acc_z']
    ])

    # gyro
    out['gyr'] = np.column_stack([
        D['gyr_x'], D['gyr_y'], D['gyr_z']
    ])

    # FSRs
    fsr_cols = ["fsr_Hallux", "fsr_Toes", "fsr_Met1", "fsr_Met3",
                "fsr_Met5", "fsr_Arch", "fsr_HeelL", "fsr_HeelR"]
    out['fsr'] = np.column_stack([D[k] for k in fsr_cols])

    return out



# =========================
# ---- Preprocessing ------
# =========================

def butter_lowpass_filter_array(x, cutoff_hz, fs_hz, order=4):
    x = np.array(x)
    if x.size == 0: return x
    nyq = 0.5 * fs_hz
    Wn = cutoff_hz / nyq
    Wn = min(Wn, 0.999)
    b,a = butter(order, Wn, btype='low')
    return filtfilt(b,a,x)


def global_filter_and_normalize(data):
    if data is None: return None

    # Determine sampling frequency
    t = np.array(data["t"], dtype=float)
    dt = np.median(np.diff(t))
    global FS
    FS = 1000.0/dt
    print(f"Sampling frequency: {FS}")

    # LPF linear acceleration
    A = data['lin_acc']
    Af = np.column_stack([butter_lowpass_filter_array(A[:,i], 10, FS)
                            for i in range(A.shape[1])])
    data['lin_acc'] = Af

    # LPF gyro
    G = data['gyr']
    Gf = np.column_stack([butter_lowpass_filter_array(G[:,i], 10, FS)
                            for i in range(G.shape[1])])
    data['gyr'] = Gf

    # LPF pressure + normalization
    F = data['fsr'].copy()
    # Replace NaNs per FSR channel with that channel's median
    for i in range(F.shape[1]):
        chan = F[:, i]
        if np.any(np.isnan(chan)):
            finite = np.isfinite(chan)
            if finite.sum() >= 2:
                median_val = np.median(chan[finite])
                chan[~finite] = median_val
            else:
                chan[:] = np.nan  
        F[:, i] = chan
    Ff = np.column_stack([butter_lowpass_filter_array(F[:,i], 10, FS)
                            for i in range(F.shape[1])])

    # Sum of all FSRs
    Psum = np.nansum(Ff, axis=1)
    invalid_rows = np.all(np.isnan(Ff), axis=1)
    Psum[invalid_rows] = np.nan

    # Normalize only over finite values MIN-MAX
    finite = np.isfinite(Psum)

    if np.sum(finite) >= 2:
        p99 = np.nanpercentile(Psum[finite], 99.9)
        if p99 > 0:
            Psum = np.clip(Psum, None, p99)
            Pnorm = Psum / p99
        else:
            Pnorm = np.full_like(Psum, np.nan)
    else:
        Pnorm = np.full_like(Psum, np.nan)

    data['Pnorm'] = Pnorm

    # Heel and front FSR signal
    heel = np.nanmean(Ff[:, 5:8], axis=1)
    front = np.nanmean(Ff[:, 0:5], axis=1)

    combined = np.hstack([heel[np.isfinite(heel)],
                      front[np.isfinite(front)]])
    
    if combined.size >= 2:
        p99 = np.nanpercentile(combined, 99.9)
        if p99 > 0:
            heel = np.clip(heel, None, p99)
            heel_norm  = heel  / p99
            front = np.clip(front, None, p99)
            front_norm = front / p99
        else:
            heel_norm  = np.full_like(heel,np.nan)
            front_norm = np.full_like(front,np.nan)
    else:
        heel_norm  = np.full_like(heel,np.nan)
        front_norm = np.full_like(front,np.nan)

    data['Pheel'] = heel_norm
    data['Pfront'] = front_norm

    return data


# =========================
# ---- Feature helpers ----
# =========================

def fill_nan_median(x):
    x = np.array(x, float, copy=True)  
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return np.zeros_like(x)
    med = np.median(x[finite])
    x[~finite] = med
    return x

def features_1d(sig):
    sig = fill_nan_median(sig)
    L = len(sig)
    out = dict(mean=0.0, median=0.0, max=0.0, min=0.0, var=0.0, domF=0.0, pow=0.0, ent=0.0)

    if L < 3: return out

    # TD
    out['mean'] = float(np.mean(sig))
    out['median'] = float(np.median(sig))
    out['max']  = float(np.max(sig))
    out['min']  = float(np.min(sig))
    out['var']  = float(np.var(sig))


    # FD using Welch + FS=270    
    freqs, psd = welch(sig, fs=FS, nperseg=min(256, L))
    s = np.sum(psd)
    if np.isfinite(s) and s > 0:
        idx = np.argmax(psd)
        domF = freqs[idx] if np.isfinite(freqs[idx]) else 0.0
        out['domF'] = float(domF)
        out['pow']  = float(s)

        p = psd / s
        if np.all(np.isfinite(p)) and np.any(p > 0):
            out['ent'] = float(entropy(p))
        else:
            out['ent'] = 0.0

    return out


def extract_signal_features(sig, name):
    f = features_1d(sig)
    return {f"{name}_mean": f['mean'],
            f"{name}_median": f['median'],
            f"{name}_max" : f['max'],
            f"{name}_min" : f['min'],
            f"{name}_var" : f['var'],
            f"{name}_domF": f['domF'],
            f"{name}_pow" : f['pow'],
            f"{name}_ent" : f['ent']}


def corrYZ(acc, name):
    if acc is None or len(acc) < 3:
        return {f"{name}_CorrYZ": 0.0}
    y = fill_nan_median(acc[:,1])
    z = fill_nan_median(acc[:,2])
    r = np.corrcoef(y, z)[0,1]
    if not np.isfinite(r):
        r = 0.0
    return {f"{name}_CorrYZ": r}

def extract_bilateral_fsr_features(PL,PR):
    if PL is None or PR is None:
        return {}
    L = min(len(PL), len(PR))
    PL = PL[:L]
    PR = PR[:L]

    TotP = PL + PR
    Max  = np.maximum(PL, PR)
    eps  = 1e-6
    Asym = np.abs(PL - PR) / (PL + PR + eps)

    return {
        "TotP_max_max"    : float(np.max(Max)),
        "Asym_median"     : float(np.nanmedian(Asym)),
        "TotP_max_75"     : float(np.percentile(Max, 75)),
        "TotP_max_25"     : float(np.percentile(Max, 25))
    }



# =========================
# ---- Windows & Labels ---
# =========================
def build_windows(labels_table, window_sec, overlap_sec):
    w_ms    = int(window_sec * 1000)
    step_ms = int((window_sec - overlap_sec) * 1000)

    Start = labels_table["start_t_ms"].iloc[0]
    End   = labels_table["end_t_ms"].iloc[-1]

    return [(t, t + w_ms) for t in range(int(Start), int(End - w_ms) + 1, step_ms)]


def window_label_if_constant(labels_df, t0, t1):
    # all segments overlapping window
    overlapping = labels_df[
        ~((labels_df["end_t_ms"] <= t0) | (labels_df["start_t_ms"] >= t1))
    ]

    unique_labels = overlapping["label"].unique()

    if len(unique_labels) == 1 and unique_labels[0] != -1:
        return unique_labels[0]
    return None

def ap_label_majority(labels_df, t0, t1):
    """
    Return the most common ActiPal label inside the window [t0, t1].

    - If no overlapping segments → None
    - If any overlapping segment has NaN label → None
    - If multiple labels → return the mode
    """
    # Find all overlapping label rows
    overlapping = labels_df[
        ~((labels_df["end_t_ms"] <= t0) | (labels_df["start_t_ms"] >= t1))
    ]

    if overlapping.empty:
        return None

    # extract labels
    lbls = overlapping["label"].values

    # if any label is NaN → window label is None
    if np.isnan(lbls).any():
        return None

    # compute frequency
    vals, counts = np.unique(lbls, return_counts=True)
    mode_idx = np.argmax(counts)

    return int(vals[mode_idx])




# =========================
# ---- Feature Extract ----
# =========================

def extract_all_features(labels_cam, labels_ap, DL, DR, window_sec, overlap_sec):
    windows = build_windows(labels_cam, window_sec, overlap_sec)

    def process(t0, t1):
        cam_lbl = window_label_if_constant(labels_cam, t0, t1)
        if cam_lbl is None:
            return None

        row = {'label_Cam': int(cam_lbl), 't_window_ms': int(t0)}
        features_added = 0

        # Add AP majority label
        if labels_ap is not None:
            ap_lbl = ap_label_majority(labels_ap, t0, t1)
            row["label_AP"] = ap_lbl

        PL = None
        PR = None
        for tag, data in (('L',DL),('R',DR)):
            if data is None: continue

            t = data['t']
            if t0 < t[0] or t1 > t[-1]: continue
            i0 = bisect.bisect_left(t,t0)
            i1 = bisect.bisect_left(t,t1)
            if i1<=i0: continue

            # PRESS (normalized)
            if 'Pnorm' in data:
                Pseg = data['Pnorm'][i0:i1]
                row.update(extract_signal_features(Pseg, f"{tag}_Press"))
                if tag == 'L':
                    PL = Pseg
                else:
                    PR = Pseg
                features_added += 1

            if 'Pheel' in data:
                heel_seg = data['Pheel'][i0:i1]
                row.update(extract_signal_features(heel_seg, f"{tag}_Hell_Press"))

                features_added += 1

            if 'Pfront' in data:
                front_seg = data['Pfront'][i0:i1]
                row.update(extract_signal_features(front_seg, f"{tag}_Front_Press"))

                features_added += 1

            if 'lin_acc' in data:
                LA = data['lin_acc'][i0:i1]
                if LA.shape[0] > 0:
                    ax, ay, az = LA[:,0], LA[:,1], LA[:,2]
                    row.update(extract_signal_features(np.sqrt(ax*ax + ay*ay), f"{tag}_AccXY"))
                    row.update(extract_signal_features(az, f"{tag}_AccZ"))
                    row.update(extract_signal_features(np.sqrt(ay*ay + az*az), f"{tag}_AccYZ"))
                    row.update(corrYZ(LA, f"{tag}"))
                    features_added += 1

            if 'gyr' in data:
                gx = data['gyr'][i0:i1, 0]
                if gx.size > 0:
                    row.update(extract_signal_features(gx, f"{tag}_Gyr"))
                    features_added += 1

        if features_added == 0:
            return None
        
        row.update(extract_bilateral_fsr_features(PL, PR))

        return row

    res = Parallel(n_jobs=-1,backend='loky')(
        delayed(process)(t0,t1) for (t0,t1) in tqdm(windows)
    )
    res = [r for r in res if r is not None]
    return pd.DataFrame(res)


# =========================
# --------- MAIN ----------
# =========================

if __name__ == "__main__":

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    input_root  = os.path.join(SCRIPT_DIR, "SoleHARmony_Dataset")
    output_root = os.path.join(SCRIPT_DIR, "FeatureTables")

    os.makedirs(output_root, exist_ok=True)

    print("INPUT ROOT:", input_root)
    print("Exists?", os.path.exists(input_root))
    print("Contents:", os.listdir(input_root) if os.path.exists(input_root) else "N/A")

    for sbj_folder in os.listdir(input_root):
        subject_path = os.path.join(input_root, sbj_folder)
        if not os.path.isdir(subject_path):
                continue

        for session in os.listdir(subject_path):
            session_folder = os.path.join(subject_path, session)
            if not os.path.isdir(session_folder):
                continue

            session_file = os.path.join(session_folder, "DataStruct.mat")  
            output_file  = os.path.join(output_root, f"{session}_features.csv")

            # Skip if features already exist
            if os.path.exists(output_file):
                print(f" Features already exist for {session}, skipping...")
                continue
            # Skip if no session file
            if not os.path.exists(session_file):
                print(f" No sessionData.mat found in {session}, skipping...")
                continue

            print(f"\n Processing {session}...")

            # Load data
            labels_Cam, labels_AP, DL, DR = load_label_and_data(session_file)
            if labels_Cam is None:
                print(f" Skipping {session} (no valid .mat file).")
                continue

            # Filter + normalize
            DL = global_filter_and_normalize(DL)
            DR = global_filter_and_normalize(DR)

            # Extract features
            df = extract_all_features(labels_Cam, labels_AP, DL, DR, window_sec=3, overlap_sec=1.5)

            # Save output
            df.to_csv(output_file, index=False)
            print(f" Saved → {output_file}   ({len(df)} windows)")
    
    print("\n All sessions processed!")
