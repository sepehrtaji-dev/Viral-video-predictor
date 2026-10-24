from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

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


def main():
    train_dl, val_dl, test_dl, enc = load_data(batch_size=BATCH)

    in_dim = next(iter(train_dl))[0].shape[1]
    model = ViralNet(in_dim).to(DEVICE)
    print(f"device={DEVICE}  in_dim={in_dim}  params={sum(p.numel() for p in model.parameters()):,}")

    # handle class imbalance
    labels = torch.cat([y for _, y in train_dl])
    pos_weight = torch.tensor([(labels == 0).sum() / (labels == 1).sum()], device=DEVICE)
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

    model.load_state_dict(best_state)
    te = run_epoch(model, test_dl, loss_fn)
    print(f"\nTEST | loss {te['loss']:.4f} auc {te['auc']:.3f} acc {te['acc']:.3f} f1 {te['f1']:.3f}")

    torch.save({
        "model_state": best_state,
        "in_dim": in_dim,
        "encoders": enc,
        "test_metrics": te,
    }, CKPT)
    print(f"saved -> {CKPT}")


if __name__ == "__main__":
    main()