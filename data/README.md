# Dataset

This project validates on the **XJTU-SY run-to-failure bearing dataset**.
The dataset is not redistributed here; download it from the official source
and point the pipeline at the extracted folder.

## Obtaining XJTU-SY

The dataset was released by the Institute of Design Science and Basic
Component at Xi'an Jiaotong University (XJTU) and the Changxing Sumyoung
Technology Co. (SY). It accompanies:

> B. Wang, Y. Lei, N. Li, N. Li, "A Hybrid Prognostics Approach for
> Estimating Remaining Useful Life of Rolling Element Bearings,"
> IEEE Transactions on Reliability, 69(1), 401-412, 2020.

The public mirror distributes the data as several multi-part `.rar`
archives. Download all parts into one folder and extract `part01`; the
remaining parts are picked up automatically:

```bash
sudo apt-get install -y unrar          # or 'brew install unrar' on macOS
unrar x XJTU-SY_Bearing_Datasets.part01.rar data/XJTU-SY/
```

If you run on Google Colab, use the notebook in `notebooks/` instead, which
mounts Google Drive and performs the extraction inside the runtime.

## Expected layout

After extraction the pipeline only needs to find folders named like
`BearingX_Y`, each containing the per-minute snapshot files. Discovery is
tolerant of wrapper directories and separator variants, so any of these
work:

```
data/XJTU-SY/
  35Hz12kN/Bearing1_1/1.csv ...
  37.5Hz11kN/Bearing2_1/1.csv ...
  40Hz10kN/Bearing3_1/1.csv ...
```

Each snapshot is a 1.28 s recording at 25.6 kHz (32768 samples), with two
channels (horizontal and vertical vibration). The pipeline reads the first
channel by default.

## Working conditions

| Subset | Condition       | Bearings        |
|--------|-----------------|-----------------|
| wc1    | 35 Hz / 12 kN   | Bearing1_1..1_5 |
| wc2    | 37.5 Hz / 11 kN | Bearing2_1..2_5 |
| wc3    | 40 Hz / 10 kN   | Bearing3_1..3_5 |

The main experiment runs on **wc2**. The cross-condition experiment trains
on wc2 and tests on wc1 and/or wc3 when those are present.

## Quick check without the real data

To verify the code runs without downloading anything, use the synthetic
smoke test (signals are not physically meaningful and must not be reported):

```bash
python run.py --demo
```
