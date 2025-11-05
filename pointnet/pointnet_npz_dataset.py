# pointnet_npz_dataset.py
import os, glob, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset

TARGET_COLS = ["B1","B2","B3","C2","C3","C4","S1","S2","S3"]

def normalize_points_center_rms(xyz: np.ndarray) -> np.ndarray:
    # xyz: (N,3)
    xyz = xyz - xyz.mean(0, keepdims=True)
    scale = np.sqrt((xyz**2).sum(1).mean()) + 1e-8  # RMS scale
    return xyz / scale

class NPZPointParamDataset(Dataset):
    """
    Expects a directory of .npz shards where each file has:
      - xyz: (N,3) float32 points
      - geom_name: str (stored in the npz)
    And a CSV mapping geom_name -> target params (columns TARGET_COLS).
    """
    def __init__(self, shards_dir: str, params_csv: str, target_cols=TARGET_COLS, transform=None):
        self.shards_dir = shards_dir
        self.files = sorted(glob.glob(os.path.join(shards_dir, "*.npz")))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npz found in {shards_dir}")
        self.df = pd.read_csv(params_csv)
        if not set(target_cols).issubset(self.df.columns):
            missing = set(target_cols) - set(self.df.columns)
            raise ValueError(f"Missing target columns in CSV: {missing}")
        # fast lookup: geom_name -> targets row
        self.df = self.df.drop_duplicates(subset=["geom_name"])
        self.targets_map = {row["geom_name"]: row[target_cols].to_numpy(dtype=np.float32)
                            for _, row in self.df.iterrows()}
        self.target_cols = target_cols
        self.transform = transform  # optional callable on xyz

        # Pre-scan to ensure all shards have matching geom_name in CSV
        self.missing = []
        keep = []
        for f in self.files:
            with np.load(f) as data:
                geom = str(data["geom_name"])
            if geom in self.targets_map:
                keep.append(f)
            else:
                self.missing.append((os.path.basename(f), geom))
        self.files = keep
        if len(self.files) == 0:
            raise RuntimeError("After filtering, no shards matched the CSV's geom_name.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path)
        xyz = data["xyz"].astype(np.float32)  # (N,3)
        # normalize per sample (safe even if you normalized earlier)
        xyz = normalize_points_center_rms(xyz)
        if self.transform is not None:
            xyz = self.transform(xyz)
        # to (3,N) for PointNet
        pts = torch.from_numpy(xyz.T.copy())  # (3,N)
        geom = str(data["geom_name"])
        y = torch.from_numpy(self.targets_map[geom])  # (9,)
        return pts, y, geom, os.path.basename(path)

