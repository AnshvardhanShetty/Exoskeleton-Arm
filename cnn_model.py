# cnn_train.py
import os
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = "/Users/anshshetty/Library/Mobile Documents/com~apple~CloudDocs/ExoHand/grabmyo"
DATA_NPZ = os.path.join(ROOT, "grabmyo_cnn_envelopes.npz")

MODEL_PATH = os.path.join(ROOT, "cnn_intent_model.pt")
META_PATH  = os.path.join(ROOT, "cnn_intent_meta.json")


# ==========================
# DATASET
# ==========================
class EMGDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, C, T), y: (N,)
        self.X = torch.from_numpy(X)  # float32
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==========================
# MODEL
# ==========================
class ShallowCNN(nn.Module):
    def __init__(self, C=4, T=128, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(C, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),          # T -> T/2
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),          # T/2 -> T/4
            nn.AdaptiveAvgPool1d(1),  # (N, 32, 1)
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        # x: (N, C, T)
        z = self.features(x)          # (N, 32, 1)
        z = z.squeeze(-1)             # (N, 32)
        logits = self.classifier(z)   # (N, num_classes)
        return logits


# ==========================
# TRAIN / EVAL
# ==========================
def participant_split(participants, train_ratio=0.8, val_ratio=0.1):
    unique_parts = np.unique(participants)
    n = len(unique_parts)
    train_p = unique_parts[:int(train_ratio * n)]
    val_p   = unique_parts[int(train_ratio * n):int((train_ratio + val_ratio) * n)]
    test_p  = unique_parts[int((train_ratio + val_ratio) * n):]
    return train_p, val_p, test_p


def make_loaders(X, y, participants, batch_size=512):
    train_p, val_p, test_p = participant_split(participants)

    def mask(pset):
        return np.isin(participants, pset)

    train_mask = mask(train_p)
    val_mask   = mask(val_p)
    test_mask  = mask(test_p)

    train_ds = EMGDataset(X[train_mask], y[train_mask])
    val_ds   = EMGDataset(X[val_mask],   y[val_mask])
    test_ds  = EMGDataset(X[test_mask],  y[test_mask])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_p, val_p, test_p


def train_epoch(model, loader, loss_fn, optim, device):
    model.train()
    total_loss = 0.0
    total, correct = 0, 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        # simple EMG augmentation: small multiplicative noise
        noise = 0.05 * torch.randn_like(xb)
        xb_aug = xb * (1.0 + noise)

        optim.zero_grad()
        logits = model(xb_aug)
        loss = loss_fn(logits, yb)
        loss.backward()
        optim.step()

        total_loss += loss.item() * yb.size(0)
        preds = logits.argmax(dim=1)
        total += yb.size(0)
        correct += (preds == yb).sum().item()

    return total_loss / total, correct / total


def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total, correct = 0, 0
    all_y = []
    all_pred = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = loss_fn(logits, yb)

            total_loss += loss.item() * yb.size(0)
            preds = logits.argmax(dim=1)
            total += yb.size(0)
            correct += (preds == yb).sum().item()

            all_y.append(yb.cpu().numpy())
            all_pred.append(preds.cpu().numpy())

    all_y = np.concatenate(all_y)
    all_pred = np.concatenate(all_pred)

    return total_loss / total, correct / total, all_y, all_pred


def main():
    # ------------------
    # LOAD DATA
    # ------------------
    data = np.load(DATA_NPZ, allow_pickle=True)
    X = data["X"]        # (N, C, T)
    y = data["y"]        # (N,)
    participants = data["participant"]  # (N,)

    N, C, T = X.shape
    print("X shape:", X.shape)
    print("Classes:", np.unique(y))
    print("Participants:", np.unique(participants).shape[0])

    train_loader, val_loader, test_loader, train_p, val_p, test_p = make_loaders(
        X, y, participants, batch_size=512
    )
    print("Train participants:", train_p)
    print("Val participants:", val_p)
    print("Test participants:", test_p)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ShallowCNN(C=C, T=T, num_classes=3).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0.0
    best_state = None

    # ------------------
    # TRAIN LOOP
    # ------------------
    EPOCHS = 15
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optim, device)
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader, loss_fn, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model_state": model.state_dict(),
                "val_acc": val_acc,
                "epoch": epoch,
            }

    # ------------------
    # EVAL BEST ON TEST
    # ------------------
    if best_state is not None:
        model.load_state_dict(best_state["model_state"])
        print(f"\nLoaded best model from epoch {best_state['epoch']} with val_acc={best_state['val_acc']:.3f}")

    _, train_acc, y_train_true, y_train_pred = eval_epoch(model, train_loader, loss_fn, device)
    _, val_acc,   y_val_true,   y_val_pred   = eval_epoch(model, val_loader,   loss_fn, device)
    _, test_acc,  y_test_true,  y_test_pred  = eval_epoch(model, test_loader,  loss_fn, device)

    print("\n== TRAIN ==")
    print("Acc:", train_acc)
    print(classification_report(y_train_true, y_train_pred))
    print(confusion_matrix(y_train_true, y_train_pred))

    print("\n== VAL ==")
    print("Acc:", val_acc)
    print(classification_report(y_val_true, y_val_pred))
    print(confusion_matrix(y_val_true, y_val_pred))

    print("\n== TEST ==")
    print("Acc:", test_acc)
    print(classification_report(y_test_true, y_test_pred))
    print(confusion_matrix(y_test_true, y_test_pred))

    # ------------------
    # SAVE MODEL + META
    # ------------------
    torch.save(model.state_dict(), MODEL_PATH)
    print("\nSaved CNN model →", MODEL_PATH)

    meta = {
        "C": int(C),
        "T": int(T),
        "num_classes": 3,
        "intent_to_idx": {"rest": 0, "close": 1, "open": 2},
        "train_participants": list(map(str, train_p)),
        "val_participants": list(map(str, val_p)),
        "test_participants": list(map(str, test_p)),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=4)
    print("Saved meta →", META_PATH)


if __name__ == "__main__":
    main()
