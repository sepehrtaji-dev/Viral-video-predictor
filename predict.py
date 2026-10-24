"""
Usage:
    python predict.py --csv data/raw/viral_videos.csv
"""
import argparse

import pandas as pd
import torch

from data import NUMERIC, BINARY, CATEGORICAL, TARGET, _transform
from model import ViralNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ckpt", default="model.pt")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    model = ViralNet(ckpt["in_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    df = pd.read_csv(args.csv)
    df["category"] = df["category"].astype(str)
    df["upload_day"] = df["upload_day"].astype(str)
    df["thumbnail_face"] = df["thumbnail_face"].astype(int)
    for c in NUMERIC:
        df[c] = df[c].fillna(df[c].median())

    X = _transform(df, ckpt["encoders"]).to(DEVICE)
    with torch.no_grad():
        proba = torch.sigmoid(model(X)).cpu().numpy()
    df["viral_proba"] = proba
    df["viral_pred"] = (proba >= 0.5).astype(int)

    print(df[["viral_proba", "viral_pred"] + ([TARGET] if TARGET in df else [])].head(10))
    df[["viral_proba", "viral_pred"]].to_csv("predictions.csv", index=False)
    print("saved -> predictions.csv")


if __name__ == "__main__":
    main()