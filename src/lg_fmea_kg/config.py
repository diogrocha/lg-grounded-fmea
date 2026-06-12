"""Acquisition constants and ground-truth labels for XJTU-SY.

The XJTU-SY run-to-failure bearing dataset was acquired at a sampling
rate of 25.6 kHz, with 1.28 s snapshots taken once per minute. The
fault-mode label and observed lifetime (in minutes) for each bearing
follow Wang et al. (2020), Table II.

Reference:
    B. Wang, Y. Lei, N. Li, N. Li, "A Hybrid Prognostics Approach for
    Estimating Remaining Useful Life of Rolling Element Bearings,"
    IEEE Transactions on Reliability, 69(1), 401-412, 2020.
"""

# Acquisition geometry
FS = 25_600                      # sampling frequency in Hz
SNAPSHOT_SEC = 1.28              # duration of each recorded snapshot in seconds
N_SAMPLES = int(FS * SNAPSHOT_SEC)   # samples per snapshot (32768)

# Failure-mode label and observed lifetime per bearing (Wang et al. 2020, Table II).
# Working conditions: 1_x = 35 Hz / 12 kN, 2_x = 37.5 Hz / 11 kN, 3_x = 40 Hz / 10 kN.
BEARING_LABELS = {
    "Bearing1_1": ("outer_race", 123),
    "Bearing1_2": ("outer_race", 161),
    "Bearing1_3": ("outer_race", 158),
    "Bearing1_4": ("cage", 122),
    "Bearing1_5": ("outer_race", 52),
    "Bearing2_1": ("inner_race", 491),
    "Bearing2_2": ("outer_race", 161),
    "Bearing2_3": ("cage", 533),
    "Bearing2_4": ("outer_race", 42),
    "Bearing2_5": ("outer_race", 339),
    "Bearing3_1": ("outer_race", 2538),
    "Bearing3_2": ("cage", 2496),
    "Bearing3_3": ("inner_race", 371),
    "Bearing3_4": ("inner_race", 1515),
    "Bearing3_5": ("outer_race", 114),
}

# Mapping from physical fault mode to integer class label used by the classifier.
# Healthy is assigned dynamically by the LG zone (see dataset.assign_labels).
FAULT_TO_CLASS = {
    "inner_race": 1,
    "outer_race": 2,
    "cage": 3,
    "rolling_element": 3,
}

# Human-readable class names, index aligned with the integer labels above.
CLASS_NAMES = ["healthy", "inner_race", "outer_race", "cage/RE"]
