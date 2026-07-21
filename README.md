# Sole-HARmony Dataset & Benchmark
 
Sole-HARmony (**Sole**-based **H**uman **A**ctivity **R**ecognition via
**M**ultimodal **O**bservation of **N**aturalistic mobilit**Y**) is a
long-duration, free-living, multimodal dataset with frame-level
ground-truth annotations for real-world human activity recognition (HAR).
It was collected using bilateral instrumented insoles (FSRs + IMU) worn by 13 healthy adults over five days of unsupervised,
everyday activity, with a body-worn camera providing frame-level ground
truth. The dataset is intended to support the development and validation
of HAR and digital mobility algorithms under realistic, free-living
conditions — a setting under-represented in existing wearable HAR
datasets, which are typically collected in controlled or semi-controlled
environments.
 
The purpose of this repository is to provide the code used to validate the dataset,
this include using a hierarchical XGBoost classifier and a CNN-BiLSTM deep
learning classifier for 5-class activity recognition, across IMU-only,
FSR-only, and fused (IMU+FSR) sensor configurations. We also include, for
illustrative purposes, example MATLAB scripts showing how activity labels
were derived from BORIS annotation files and synchronized with the
wearable sensor data during dataset creation.
 
<!-- For more information, see the dataset paper:
 
> De Marzi, L., Liu, K.-C., Duong, T., Zanotto, D. Sole-HARmony: A
> long-duration, free-living, multimodal dataset with frame-level
> ground-truth annotations for real-world human activity recognition.
> *Scientific Data* (2026). Dataset: https://doi.org/10.5281/zenodo.19242395
-->
 
## Dataset summary
 
- **Participants:** 13 healthy adults (4 female, 9 male; age 23.3 ± 4.6 y,
  weight 74.5 ± 19.1 kg, height 172.07 ± 5.8 cm). IRB approval: Stevens
  Institute of Technology (IRB ID 2021-034(N)).
- **Protocol:** 4–5 hours/day of unsupervised, free-living recording over
  5 days per participant (4.41 ± 0.78 h/day), for a total of **286.57
  hours**. No minimum activity durations or step counts were enforced, to
  preserve ecological validity.
- **Sensors (per insole, bilateral):**
  - 8-cell FSR array (IEE S.A., Luxembourg) — Hallux, Toes, Met1, Met3,
    Met5, Arch, HeelL, HeelR
  - 9-DOF IMU (Yost Labs Inc.) mounted under the medial arch — tri-axial
    accelerometer + gyroscope used (magnetometer not used)
  - All signals sampled at **270 Hz**
  - Ground-truth video from a downward-facing, body-worn camera (BOBLOV D1
    Mini), mounted at the waist on the dominant-leg side via a custom
    mirror attachment
  - 10 of 13 participants also wore an activPAL3 micro (thigh-mounted) for
    cross-device comparison
- **Activity classes (from video annotation):** Sitting, Standing, Walking,
  Stairs Down, Stairs Up, Undefined. Distribution across the dataset:
  Sitting 210.4 h (73.4%), Standing 34.7 h (12.1%), Walking 34.6 h (12.1%),
  Stairs Down 1.1 h (0.38%), Stairs Up 1.0 h (0.35%), Undefined 4.8 h
  (1.69%).
- **Annotation:** Manual, frame-level annotation of body-camera video using
  BORIS (open-source event-logging software).
- **Synchronization:** Peak-based alignment using an initial triple
  foot-stomp event per session, matched between video and insole/activPAL
  acceleration peaks. 
- **Data quality:** ~3.17% of FSR samples were invalidated (NaN) due to
  sensor artifacts, concentrated in the left insole's arch/hallux channels;
  IMU signals were unaffected. Per-session data-quality notes are provided
  in each session's metadata file.
 
## Data Records
 
All data are hosted at https://doi.org/10.5281/zenodo.19242395, organized
hierarchically by participant and session:
 
```
Dataset/
├── Demographics.xlsx              # age, gender, height, weight per participant
├── C001/
│   ├── C001_1658420327/           # session = subject ID + UNIX timestamp
│   │   ├── meta.json               # subject ID, timestamp, duration, sampling
│   │   │                           #   rate, sensor units, data-quality notes,
│   │   │                           #   label definitions
│   │   ├── DataStruct.mat          # synchronized wearable data + labels
│   │   └── C001_1658420327_AP.datx # raw activPAL file (if worn)
│   └── C001_1658503124/
├── C002/ ...
└── C013/ ...
```
 
**`DataStruct.mat` contents:**
 
- `labelsCam`, `labelsAP` — N×3 matrices (label, start time, end time).
  Camera labels: 0=Sitting, 1=Standing, 2=Walking, 3=Stairs Down,
  4=Stairs Up, -1=Undefined. ActivPAL labels (10 subjects only):
  0=Sitting, 1=Standing, 2=Stepping.
- `InsoleL`, `InsoleR` — structs with synchronized M×1 vectors: `t_ms`,
  quaternion orientation (`quat_w/x/y/z`), linear acceleration
  (`lin_acc_x/y/z`), angular velocity (`gyr_x/y/z`), raw acceleration
  (`raw_acc_x/y/z`), and 8 FSR channels (`fsr_Hallux`, `fsr_Toes`,
  `fsr_Met1`, `fsr_Met3`, `fsr_Met5`, `fsr_Arch`, `fsr_HeelL`,
  `fsr_HeelR`). Axes: X = medio-lateral, Y = anteroposterior (heel-to-toe),
  Z = normal to the insole surface.
 
## Methods (benchmark pipeline)
 
### Stage 1: Feature extraction (XGBoost pipeline only)
 
For XGBoost, a **3-second sliding window with 50% overlap** is used, and
**118 time- and frequency-domain features** are extracted per window,
including: acceleration magnitude projected onto horizontal (X–Y) and
vertical (Z) axes; angular velocity about the mediolateral (X) axis;
summed and regional (hindfoot/forefoot) FSR signals (session-wise
99th-percentile normalized); a left–right plantar pressure asymmetry
metric; percentile features of the max(left, right) normalized FSR sum;
and the within-window Pearson correlation between Y- and Z-axis
acceleration. Implemented in `FeatureExtraction.py`, which loads
`DataStruct.mat` session files and outputs per-window `.csv` feature
tables.
 
### Stage 2: Activity classification
 
```
                Static vs. Dynamic
                 /              \
      Sitting vs. Standing   Walking vs. Stairs
                                    \
                          Stairs Down vs. Stairs Up
```
 
**Hierarchical XGBoost.** Four cascaded binary XGBoost classifiers, trained
on the extracted feature `.csv` tables:
- `loocv_XGBoost_*.py` (4 scripts, one per binary stage) —
  leave-one-out cross-validation (LOOCV) across the 10 activPAL-equipped
  "training subjects." Each fold trains on 9 subjects (with an internal
  80/20 split: 80% for Bayesian hyperparameter optimization, 20% for
  early-stopping validation) and tests on the held-out subject.
- `ext_XGBoost_*.py` (4 scripts, one per binary stage) —
  model trained on all 10 training subjects, evaluated on the 3
  remaining, fully unseen subjects (external validation, ~29% of the
  dataset).
 
**CNN-BiLSTM.** Raw, normalized sensor signals are segmented into the same
3-s/50%-overlap windows, passed through two Conv1D–BatchNorm–ReLU–MaxPool
blocks, then a BiLSTM layer; concatenated forward/backward hidden states
feed a fully-connected classification head.
- `loocv_CNNbiLSTM.py` — LOOCV across the 10 training subjects. Memory-mapped
  per-subject `.npy` caching (`MultiSubjectDataset`) avoids full-dataset
  concatenation.
- `ext_CNNbiLSTM.py` — model trained on all 10 training
  subjects, evaluated on the 3 held-out subjects.
 
Both model families are trained separately on three sensor configurations
— **IMU+FSR**, **IMU-only**, and **FSR-only** — to quantify each
modality's contribution.
 
### Stage 3: Prediction

**Hierarchical XGBoost.**
- `loocv_cascade_prediction.py` — applies the 4×10 LOOCV models to their
  respective held-out subjects and aggregates confusion matrices.
- `ext_cascade_prediction.py` — applies the 4 holdout models to the
  3 unseen test subjects.


 
## Implementation
 
### Environment setup
 
Note: models were originally trained on a Linux
workstation with an AMD Ryzen 7 9700X and an NVIDIA RTX PRO 4500 Blackwell
GPU.
 
### Data preparation
 
The MATLAB scripts used during dataset creation (BORIS-annotation-to-label
conversion and sensor/label synchronization --- `sync_cam_insole.m`, `expandCamTable.m`) are also included in this
repository just for transparency and illustrative purpose.
Users will start directly from the released `DataStruct.mat` session files.

 
```bash
$ python FeatureExtraction.py 
```
 
This loads each session's `DataStruct.mat`, applies the 3-s/50%-overlap
windowing, and exports the 118-feature `.csv` table plus aligned window
labels used by the XGBoost pipeline.
 
### Model training
 
**XGBoost — LOOCV (10 training subjects):**
```bash
$ python loocv_XGBoost_SD.py    # static vs dynamic
$ python loocv_XGBoost_SS.py    # sitting vs standing 
$ python loocv_XGBoost_WS.py    # walking vs stairs
$ python loocv_XGBoost_sAD.py   # stair ascending vs stair descending 
```
 
**XGBoost — external holdout (train 10 / test 3):**
```bash
$ python ext_XGBoost_SD.py    # static vs dynamic
$ python ext_XGBoost_SS.py    # sitting vs standing
$ python ext_XGBoost_WS.py    # walking vs stairs
$ python ext_XGBoost_sAD.py   # stair ascending vs stair descending
```
 
**CNN-BiLSTM:**
```bash
# LOOCV across 10 training subjects, includes grid search + 3-fold CV
$ python loocv_CNNbiLSTM.py --subjects C001,C002,...,C0XX 
 
# final model: train on 10, test on 3 unseen, includes hyperparameter search
$ python ext_CNNbiLSTM.py 
```

 
### Cascade prediction / evaluation
**XGBoost** 
```bash
$ python loocv_cascade_prediction.py
$ python ext_cascade_prediction.py 
```

 
## Usage notes & limitations
 
- The dataset comprises healthy young adults only (21±1 y in the technical
  validation cohort); models trained on it may not generalize directly to
  clinical populations or other age groups.
- ~3.17% of FSR samples (mostly left-insole arch/hallux) are invalidated
  (NaN) due to sensor artifacts — check each session's `meta.json` for
  affected channels before use.
- ~1.81% of the dataset is labeled "Undefined" (camera obstruction,
  privacy pauses, or activities outside the 5-class taxonomy, e.g.
  running).
- Only 5 locomotor activity classes are annotated (Sitting, Standing,
  Walking, Stairs Down, Stairs Up); other activities (running, shuffling,
  ramps) are not distinguished.
 
## Data availability
 
https://doi.org/10.5281/zenodo.19242395
 
## References
 
