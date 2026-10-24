"""
Usage:
    python predict.py --csv data/raw/viral_videos_clean.csv
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
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    model = ViralNet(ckpt["in_dim"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    thr = float(ckpt.get("threshold", 0.5))
    print(f"loaded {args.ckpt} | threshold = {thr:.3f}")

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
    df["viral_pred"] = (proba >= thr).astype(int)

    show_cols = ["viral_proba", "viral_pred"] + ([TARGET] if TARGET in df else [])
    print(df[show_cols].head(10))

    # summary
    print(f"\npredicted viral: {int(df['viral_pred'].sum()):,} / {len(df):,} "
          f"({df['viral_pred'].mean():.1%})")
    if TARGET in df:
        from sklearn.metrics import classification_report, roc_auc_score
        print("\nvs ground truth on this file:")
        print(f"  AUC = {roc_auc_score(df[TARGET], df['viral_proba']):.3f}")
        print(classification_report(df[TARGET], df["viral_pred"], digits=3))

    df[["viral_proba", "viral_pred"]].to_csv(args.out, index=False)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()