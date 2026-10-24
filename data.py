import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

CSV_PATH = "data/raw/viral_videos_clean.csv"
TARGET = "viral"

# Leaky columns — known only AFTER a video is uploaded. Must not be features.
LEAKY = ["views", "likes", "comments", "shares"]

NUMERIC = ["duration_sec", "title_length", "upload_hour",
           "tags_count", "description_length", "subscriber_count"]
BINARY = ["thumbnail_face"]
CATEGORICAL = ["category", "upload_day"]


class ViralDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def _fit_encoders(df: pd.DataFrame) -> dict:
    """Fit category vocab + numeric mean/std on TRAIN only."""
    cats = {c: sorted(df[c].unique().tolist()) for c in CATEGORICAL}
    num_means = df[NUMERIC].mean()
    num_stds = df[NUMERIC].std().replace(0, 1.0)
    return {"cats": cats, "means": num_means, "stds": num_stds}


def _transform(df: pd.DataFrame, enc: dict) -> torch.Tensor:
    """Build a float tensor with all features."""
    cols = []

    # numeric -> standardized
    num = (df[NUMERIC] - enc["means"]) / enc["stds"]
    cols.append(num.values)

    # binary -> as-is
    cols.append(df[BINARY].values.astype("float32"))

    # categorical -> one-hot using fitted vocab (unknown -> all zeros)
    for c in CATEGORICAL:
        vocab = enc["cats"][c]
        onehot = torch.zeros(len(df), len(vocab))
        idx = {v: i for i, v in enumerate(vocab)}
        for j, val in enumerate(df[c].tolist()):
            if val in idx:
                onehot[j, idx[val]] = 1.0
        cols.append(onehot.numpy())

    X = np.concatenate([c if c.ndim == 2 else c.reshape(-1, 1) for c in cols], axis=1)
    return torch.tensor(X, dtype=torch.float32)


def load_data(batch_size: int = 64, seed: int = 42):
    df = pd.read_csv(CSV_PATH)
    df = df.drop(columns=["video_id"] + LEAKY, errors="ignore")
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    df["category"] = df["category"].astype(str)
    df["upload_day"] = df["upload_day"].astype(str)
    df["thumbnail_face"] = df["thumbnail_face"].astype(int)
    for c in NUMERIC:
        df[c] = df[c].fillna(df[c].median())

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df[TARGET], random_state=seed
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.15, stratify=train_df[TARGET], random_state=seed
    )

    enc = _fit_encoders(train_df)

    def make(df_, shuffle):
        X = _transform(df_, enc)
        y = torch.tensor(df_[TARGET].values, dtype=torch.float32)
        return DataLoader(ViralDataset(X, y), batch_size=batch_size, shuffle=shuffle)

    return make(train_df, True), make(val_df, False), make(test_df, False), enc