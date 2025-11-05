# %%
# train_eval_pointnet.py
import os, json, math, random, numpy as np, torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from pointnet_npz_dataset import NPZPointParamDataset, TARGET_COLS
from point_to_parameter_model import PointNetRegressor  # uses your existing model
# ^ ensure this file is on PYTHONPATH or in the same folder. :contentReference[oaicite:2]{index=2}

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ==== PATHS (edit if needed) ====
TRAIN_SHARDS = "/home/nicksung/Desktop/nicksung/bwb_full_v2/data/pointnet_dataset_train"
TEST_SHARDS  = "/home/nicksung/Desktop/nicksung/bwb_full_v2/data/pointnet_dataset_test"
TRAIN_CSV    = "/home/nicksung/Desktop/nicksung/bwb_full_v2/data/train/geom_params_train.csv"
TEST_CSV     = "/home/nicksung/Desktop/nicksung/bwb_full_v2/data/test/geom_params_test.csv"

OUT_DIR      = "./runs/pointnet_regress"
BATCH_SIZE   = 32
EPOCHS       = 10000
LR           = 1e-3
LATENT       = 128
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUT_DIR, exist_ok=True)

def compute_target_norm_stats(loader):
    # gather all y over loader to compute mean/std
    ys = []
    for pts, y, _, _ in loader:
        ys.append(y.numpy())
    ys = np.concatenate(ys, axis=0)
    mu = ys.mean(axis=0)             # (9,)
    std = ys.std(axis=0) + 1e-8      # (9,)
    return mu.astype(np.float32), std.astype(np.float32)

def zscore(y, mu, std):    return (y - mu) / std
def un_zscore(y, mu, std): return y * std + mu

def split_by_geom(dataset, val_frac=0.10):
    # keep shards from the same geom together in either train or val
    geom_to_indices = defaultdict(list)
    for i in range(len(dataset)):
        _, _, geom, _ = dataset[i]
        geom_to_indices[geom].append(i)
    geoms = list(geom_to_indices.keys())
    rng = np.random.default_rng(SEED)
    rng.shuffle(geoms)
    n_val = max(1, int(len(geoms) * val_frac))
    val_geoms = set(geoms[:n_val])
    train_idx, val_idx = [], []
    for g, idxs in geom_to_indices.items():
        (val_idx if g in val_geoms else train_idx).extend(idxs)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)

def train_one_epoch(model, loader, optim, criterion, mu, std):
    model.train()
    total = 0.0; n = 0
    for pts, y, _, _ in loader:
        pts = pts.to(DEVICE)         # (B,3,N)
        y   = y.to(DEVICE)           # (B,9)
        y_n = (y - torch.from_numpy(mu).to(DEVICE)) / torch.from_numpy(std).to(DEVICE)
        optim.zero_grad()
        pred_n = model(pts)          # (B,9)
        loss = criterion(pred_n, y_n)
        loss.backward(); optim.step()
        total += loss.item() * pts.size(0)
        n += pts.size(0)
    return total / max(1, n)

@torch.no_grad()
def eval_loss(model, loader, criterion, mu, std):
    model.eval()
    total = 0.0; n = 0
    for pts, y, _, _ in loader:
        pts = pts.to(DEVICE); y = y.to(DEVICE)
        y_n = (y - torch.from_numpy(mu).to(DEVICE)) / torch.from_numpy(std).to(DEVICE)
        pred_n = model(pts)
        loss = criterion(pred_n, y_n)
        total += loss.item() * pts.size(0)
        n += pts.size(0)
    return total / max(1, n)

@torch.no_grad()
def evaluate_denorm_metrics(model, loader, mu, std):
    model.eval()
    y_true_all, y_pred_all = [], []
    for pts, y, _, _ in loader:
        pts = pts.to(DEVICE)
        pred_n = model(pts).cpu().numpy()     # normalized
        pred = un_zscore(pred_n, mu, std)     # denormalize
        y_true_all.append(y.numpy())
        y_pred_all.append(pred)
    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)

    # Per-parameter metrics
    per = {}
    for j, name in enumerate(TARGET_COLS):
        yt, yp = y_true[:, j], y_pred[:, j]
        per[name] = {
            "RMSE": math.sqrt(mean_squared_error(yt, yp)),
            "MAE": mean_absolute_error(yt, yp),
            "R2":  r2_score(yt, yp)
        }
    # Macro
    macro = {
        "RMSE": float(np.sqrt(((y_true - y_pred)**2).mean())),
        "MAE":  float(np.abs(y_true - y_pred).mean()),
        "R2":   float(r2_score(y_true, y_pred, multioutput="variance_weighted"))
    }
    return per, macro

def main():
    # Datasets
    dtrain_full = NPZPointParamDataset(TRAIN_SHARDS, TRAIN_CSV)
    dtest       = NPZPointParamDataset(TEST_SHARDS,  TEST_CSV)

    # Train/val split by geometry to avoid leakage
    dtrain, dval = split_by_geom(dtrain_full, val_frac=0.10)

    # Loaders
    train_loader = DataLoader(dtrain, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(dval,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(dtest,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Target normalization (z-score) computed on TRAIN ONLY
    mu, std = compute_target_norm_stats(train_loader)
    np.savez(os.path.join(OUT_DIR, "target_norm_stats.npz"), mu=mu, std=std, cols=np.array(TARGET_COLS, dtype=object))

    # Model/optim
    model = PointNetRegressor(latent_size=LATENT, output_size=len(TARGET_COLS)).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(1, EPOCHS+1):
        tr = train_one_epoch(model, train_loader, optim, criterion, mu, std)
        va = eval_loss(model, val_loader, criterion, mu, std)
        if va < best_val:
            best_val = va
            torch.save({"model": model.state_dict(),
                        "mu": mu, "std": std,
                        "cols": TARGET_COLS},
                       os.path.join(OUT_DIR, "best.pt"))
        if epoch % 10 == 0 or epoch == 1:
            print(f"[{epoch:04d}] train {tr:.5f} | val {va:.5f}")

    # # Load best and evaluate on TEST (denormalized metrics)
    # ckpt = torch.load(os.path.join(OUT_DIR, "best.pt"), map_location="cpu")
    # model.load_state_dict(ckpt["model"])
    # mu, std = ckpt["mu"], ckpt["std"]

    # per, macro = evaluate_denorm_metrics(model, test_loader, mu, std)

    # # Print nicely and save
    # print("\n=== Test metrics (denormalized) ===")
    # for k, v in per.items():
    #     print(f"{k:>2}: RMSE={v['RMSE']:.5f}  MAE={v['MAE']:.5f}  R²={v['R2']:.4f}")
    # print(f"--- Macro ---  RMSE={macro['RMSE']:.5f}  MAE={macro['MAE']:.5f}  R²={macro['R2']:.4f}")

    # with open(os.path.join(OUT_DIR, "test_metrics.json"), "w") as f:
    #     json.dump({"per_param": per, "macro": macro}, f, indent=2)

if __name__ == "__main__":
    main()


# %%
model = PointNetRegressor(latent_size=LATENT, output_size=len(TARGET_COLS)).to(DEVICE)
ckpt = torch.load(os.path.join(OUT_DIR, "best.pt"),
                  map_location="cpu",
                  weights_only=False)  # <- add this


# %%
dtrain_full = NPZPointParamDataset(TRAIN_SHARDS, TRAIN_CSV)
dtest       = NPZPointParamDataset(TEST_SHARDS,  TEST_CSV)

# Train/val split by geometry to avoid leakage
dtrain, dval = split_by_geom(dtrain_full, val_frac=0.10)

# Loaders
train_loader = DataLoader(dtrain, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(dval,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(dtest,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


# %%
model.load_state_dict(ckpt["model"])
mu, std = ckpt["mu"], ckpt["std"]

per, macro = evaluate_denorm_metrics(model, test_loader, mu, std)

# Print nicely and save
print("\n=== Test metrics (denormalized) ===")
for k, v in per.items():
    print(f"{k:>2}: RMSE={v['RMSE']:.5f}  MAE={v['MAE']:.5f}  R²={v['R2']:.4f}")
print(f"--- Macro ---  RMSE={macro['RMSE']:.5f}  MAE={macro['MAE']:.5f}  R²={macro['R2']:.4f}")

with open(os.path.join(OUT_DIR, "test_metrics.json"), "w") as f:
    json.dump({"per_param": per, "macro": macro}, f, indent=2)


