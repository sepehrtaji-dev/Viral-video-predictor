from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_recall_curve, classification_report,
)

from data import load_data
from model import ViralNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 40
LR = 1e-3
BATCH = 64
CKPT = Path("checkpoints\\model.pt")


def run_epoch(model, loader, loss_fn, opt=None):
    train = opt is not None
    model.train(train)
    total_loss, all_logits, all_y = 0.0, [], []
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits = model(X)
        loss = loss_fn(logits, y)
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_loss += loss.item() * len(y)
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())
    logits = torch.cat(all_logits)
    y = torch.cat(all_y)
    proba = torch.sigmoid(logits).numpy()
    pred = (proba >= 0.5).astype(int)
    return {
        "loss": total_loss / len(loader.dataset),
        "auc": roc_auc_score(y.numpy(), proba),
        "acc": accuracy_score(y.numpy(), pred),
        "f1": f1_score(y.numpy(), pred),
    }


def collect_probs(model, loader):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            probs.append(torch.sigmoid(model(X)).cpu())
            labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def find_best_threshold(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """Returns (best_threshold, best_f1, f1_at_0.5)."""
    prec, rec, thresh = precision_recall_curve(labels, probs)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    best_i = int(f1s.argmax())
    best_thresh = float(thresh[best_i]) if best_i < len(thresh) else 0.5

    # f1 at 0.5 for comparison
    pred_05 = (probs >= 0.5).astype(int)
    f1_05 = float(f1_score(labels, pred_05))

    return best_thresh, float(f1s[best_i]), f1_05


def main():
    train_dl, val_dl, test_dl, enc = load_data(batch_size=BATCH)

    in_dim = next(iter(train_dl))[0].shape[1]
    model = ViralNet(in_dim).to(DEVICE)
    print(f"device={DEVICE}  in_dim={in_dim}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    # handle class imbalance
    labels = torch.cat([y for _, y in train_dl])
    pos_weight = torch.tensor(
        [(labels == 0).sum() / (labels == 1).sum()], device=DEVICE
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_auc, best_state = -1.0, None
    for epoch in range(1, EPOCHS + 1):
        tr = run_epoch(model, train_dl, loss_fn, opt)
        va = run_epoch(model, val_dl, loss_fn)
        sched.step()
        marker = ""
        if va["auc"] > best_auc:
            best_auc = va["auc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = " *"
        print(f"ep {epoch:02d} | train loss {tr['loss']:.4f} auc {tr['auc']:.3f} "
              f"| val loss {va['loss']:.4f} auc {va['auc']:.3f} f1 {va['f1']:.3f}{marker}")

    # ---- restore best & find optimal threshold on val ----
    model.load_state_dict(best_state)

    val_probs, val_labels = collect_probs(model, val_dl)
    best_thresh, best_f1, f1_05 = find_best_threshold(val_probs, val_labels)
    print(f"\nthreshold sweep on val:")
    print(f"  best F1 = {best_f1:.3f} @ threshold = {best_thresh:.3f}")
    print(f"  F1 @ 0.5 = {f1_05:.3f}")

    # ---- final test eval (at both thresholds) ----
    test_probs, test_labels = collect_probs(model, test_dl)

    def report(probs, labels, thresh, tag):
        pred = (probs >= thresh).astype(int)
        return {
            "tag": tag,
            "threshold": thresh,
            "loss": None,  # computed below
            "auc": float(roc_auc_score(labels, probs)),
            "acc": float(accuracy_score(labels, pred)),
            "f1": float(f1_score(labels, pred)),
        }

    te_05 = report(test_probs, test_labels, 0.5, "test@0.5")
    te_best = report(test_probs, test_labels, best_thresh, f"test@{best_thresh:.3f}")

    # compute test loss once
    model.eval()
    with torch.no_grad():
        total, n = 0.0, 0
        for X, y in test_dl:
            X, y = X.to(DEVICE), y.to(DEVICE)
            total += loss_fn(model(X), y).item() * len(y)
            n += len(y)
    test_loss = total / n

    print(f"\nTEST loss {test_loss:.4f}")
    print(f"  @0.5 :        auc {te_05['auc']:.3f}  acc {te_05['acc']:.3f}  f1 {te_05['f1']:.3f}")
    print(f"  @{best_thresh:.3f}: auc {te_best['auc']:.3f}  acc {te_best['acc']:.3f}  f1 {te_best['f1']:.3f}")

    # classification report at best threshold
    print(f"\nclassification report (test @ threshold {best_thresh:.3f}):")
    print(classification_report(
        test_labels, (test_probs >= best_thresh).astype(int), digits=3
    ))

    # ---- save ----
    torch.save({
        "model_state": best_state,
        "in_dim": in_dim,
        "encoders": enc,
        "threshold": best_thresh,
        "test_metrics": {
            "loss": test_loss,
            "auc": te_best["auc"],
            "acc": te_best["acc"],
            "f1": te_best["f1"],
            "threshold": best_thresh,
        },
    }, CKPT)
    print(f"saved -> {CKPT}")


if __name__ == "__main__":
    main()