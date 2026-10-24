# Viral Video Predictor

A small PyTorch model that predicts whether a YouTube-style video will go viral,
using only **pre-upload features** (no views/likes/comments/shares — those leak the answer).

Achieves **AUC 0.899 / F1 0.670 / accuracy 0.853** on held-out test data.

---

## What it does

Given metadata about a video **before it's uploaded**, predict `viral` (0/1):

| Feature | Type | Notes |
|---|---|---|
| `category` | categorical | Beauty, Tech, Fitness, Cooking, ... |
| `duration_sec` | numeric | video length in seconds |
| `title_length` | numeric | characters |
| `thumbnail_face` | binary | 1 if a face appears in the thumbnail |
| `upload_hour` | numeric | 0–23 |
| `upload_day` | categorical | Mon–Sun |
| `tags_count` | numeric | number of tags |
| `description_length` | numeric | characters |
| `subscriber_count` | numeric | channel size at upload time |

**Deliberately excluded** (leaky — only known *after* a video goes viral):
`views`, `likes`, `comments`, `shares`.

---

## Results

Trained on 12,000 rows (80/20 not-viral/viral split), 15% of train carved out as val.

| Metric | @ threshold 0.5 | @ tuned threshold 0.714 |
|---|---|---|
| AUC | 0.899 | 0.899 |
| Accuracy | 0.803 | **0.853** |
| F1 | 0.636 | **0.670** |

**Per-class (test @ 0.714):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Not viral (0) | 0.934 | 0.878 | 0.905 | 1920 |
| Viral (1) | 0.606 | 0.750 | 0.670 | 480 |

**Known limitation:** precision on the viral class is ~0.61 — roughly 4 in 10 "viral"
predictions are wrong. This is close to the information ceiling of the available
features (only `subscriber_count` and `duration_sec` carry meaningful signal).

---

## Project structure

```
TY-VIRAL-vid/
├── data/
│   ├── raw/
│   │   ├── viral_videos.csv          # original
│   │   └── viral_videos_clean.csv    # after clean.ipynb (leaky cols dropped)
│   ├── datasheet.md
│   └── readme.md
├── clean.ipynb     # data cleaning + EDA
├── data.py         # load, split, encode, DataLoaders
├── model.py        # ViralNet (MLP)
├── train.py        # training loop + threshold sweep
├── predict.py      # CLI batch prediction
├── ui.py           # PySide6 GUI
├── model.pt        # saved checkpoint (weights + encoders + threshold)
├── predictions.csv # output of predict.py
├── requirements.txt
└── README.md
```

---

## Setup

```powershell
pip install -r requirements.txt
```

`requirements.txt`:

```
torch>=2.3
pandas>=2.0
numpy>=1.26
scikit-learn>=1.4
PySide6>=6.6
```

---

## Usage

### 1. Clean the raw data

Open `clean.ipynb` and run all cells. Produces
`data/raw/viral_videos_clean.csv` (leaky columns dropped).

### 2. Train

```powershell
python train.py
```

- 40 epochs, AdamW, cosine LR, `pos_weight` for class imbalance.
- Saves best-val checkpoint to `model.pt`, including the **tuned threshold**
  found by sweeping the validation PR curve.

### 3. Batch prediction (CLI)

```powershell
python predict.py --csv data/raw/viral_videos_clean.csv
```

Writes `predictions.csv` with `viral_proba` + `viral_pred` columns.

### 4. Interactive UI

```powershell
python ui.py
```

PySide6 window with inputs for all 9 features, a Predict button, and a
color-coded verdict (green "Not viral" / red "🔥 VIRAL") showing the probability
and the threshold used.

---

## How it works

**Preprocessing** (`data.py`):
- Numeric features z-scored using **train-only** mean/std.
- Categoricals one-hot encoded using **train-only** vocab
  (unseen values at inference → all-zeros row, no crash).
- Binary features pass through as-is.

Encoders are saved inside the checkpoint, so `predict.py` and `ui.py` apply
identical transforms to new data.

**Model** (`model.py`): 3-layer MLP — 19 → 64 → 32 → 1 logit — with ReLU
and dropout 0.2. Small on purpose; a larger model overfits instantly on
12K rows with 2 informative features.

**Training** (`train.py`):
- Loss: `BCEWithLogitsLoss` with `pos_weight = n_neg / n_pos` ≈ 4.0,
  to counter the 80/20 class imbalance.
- Optimizer: AdamW (lr 1e-3, weight decay 1e-4), cosine schedule over 40 epochs.
- Best-val checkpoint restored, then a PR-curve sweep picks the F1-maximizing
  threshold on validation — this is where the +0.034 F1 comes from
  (0.636 → 0.670).

**Why the threshold matters:** the model is trained with `pos_weight`
so its raw probability output is calibrated as if classes were balanced.
At evaluation time with a 20% positive rate, threshold 0.5 under-predicts
the minority class. Sweeping gives 0.714, which trades some recall for
much better precision.

---

## Ceiling — why AUC is stuck at ~0.90

Correlation of each feature with `viral`:

| Feature | Corr with `viral` |
|---|---|
| `subscriber_count` | **0.414** |
| `thumbnail_face` | 0.057 |
| `duration_sec` | **−0.156** |
| everything else | ≈ 0 |

Only two features carry real signal. The model has converged:
val AUC is flat from epoch 25 to 40 (0.901 → 0.903), and the test AUC (0.899)
matches validation — no further training capacity will help.

To break past ~0.90 AUC you would need either:
- **New features** (log-subs, interactions, duration buckets — see "Future work"), or
- **More real data** with a less deterministic label.

The label itself appears to be a synthetic rule (uniform category distribution,
sequential `video_id`s, 20% base rate), so the ceiling is probably set by
noise in the label generator, not by the model.

---

## Future work

- **Feature engineering** (highest leverage, ~20 min):
  `log1p(subscriber_count)`, `log_subs × thumbnail_face`,
  `tags_per_minute`, `duration_bucket`, `hour_bucket`.
  Estimated gain: AUC 0.90 → 0.92, F1 0.67 → 0.72.
- **XGBoost baseline** — confirm the MLP is at the ceiling (or switch if not).
- **Threshold-aware training** — use focal loss or tune the loss during training
  instead of post-hoc.
- **UI batch tab** — drag in a CSV, see all predictions in a table.

---

## Files at a glance

| File | Purpose |
|---|---|
| `clean.ipynb` | EDA, string normalization, NaN/outlier handling, drop leaky cols |
| `data.py` | CSV → DataLoaders, train-only encoders, `_transform` for inference |
| `model.py` | `ViralNet` — 3-layer MLP, 19 → 64 → 32 → 1 |
| `train.py` | 40-epoch loop, best-val checkpoint, PR-curve threshold sweep |
| `predict.py` | Loads `model.pt`, predicts on any CSV with matching columns |
| `ui.py` | PySide6 GUI for single-video input |
| `model.pt` | Checkpoint: `model_state`, `in_dim`, `encoders`, `threshold`, `test_metrics` |

---

## License

MIT — see [LICENSE](LICENSE) for the full text.

Copyright (c) 2025 Sepehr Taji