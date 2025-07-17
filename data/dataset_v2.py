import os
import csv
import h5py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import random, numpy as np, torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


class UnifiedDesignDataset(Dataset):
    def __init__(self,
                 csv_path,
                 hdf5_points_path,
                 hdf5_normals_path,   # <--- NEW: path to normals_data.hdf5
                 norm_stats=None,
                 mode="train"):
        """
        Args:
            csv_path (str): CSV file path containing metadata.
            hdf5_points_path (str): HDF5 file path containing coords & coeffs.
            hdf5_normals_path (str): HDF5 with the "normals" group.
            norm_stats (dict, optional): Precomputed normalization stats.
            mode (str): "train" or "test". Determines key formatting for cases.
        """
        super().__init__()
        self.mode = mode

        # Read CSV
        df = pd.read_csv(csv_path, dtype={'case_name': str, 'geom_name': str})

        # Columns for shape & flight parameters
        shape_cols  = ['B1', 'B2', 'B3', 'C2', 'C3', 'C4', 'S1', 'S2', 'S3']
        flight_cols = ['alt_kft', 'Re_L', 'M_inf', 'alpha_deg']

        # Key format can differ for test vs train, so handle that:
        if self.mode == "test":
            key_format = "case_{num:03d}"
        else:
            key_format = "case_{num:04d}"

        # ---- If needed, compute normalization stats from scratch ----
        # (this part is unchanged except we do NOT scale normals.)
        if norm_stats is None:
            shape_mean = df[shape_cols].mean().values
            shape_std  = df[shape_cols].std().values
            flight_mean = df[flight_cols].mean().values
            flight_std  = df[flight_cols].std().values
            shape_std[shape_std == 0] = 1
            flight_std[flight_std == 0] = 1

            # Open the main HDF5 to compute coordinate & output stats
            with h5py.File(hdf5_points_path, "r") as h5f:
                points_grp = h5f["points"]
                cp_grp     = h5f["cp"]
                cfx_grp    = h5f["cf_x"]
                cfz_grp    = h5f["cf_z"]

                coord_min = np.array([np.inf, np.inf, np.inf])
                coord_max = np.array([-np.inf, -np.inf, -np.inf])
                all_outputs = []

                for row in df.itertuples(index=False):
                    try:
                        if "case_" in row.case_name:
                            num = int(row.case_name.split('_')[1])
                        else:
                            num = int(row.case_name)
                        ds_key = key_format.format(num=num)
                    except Exception as e:
                        print(f"Error processing {row.case_name}: {e}")
                        continue

                    if ds_key not in points_grp:
                        continue

                    pts_coords = points_grp[ds_key][()]
                    coord_min = np.minimum(coord_min, pts_coords.min(axis=0))
                    coord_max = np.maximum(coord_max, pts_coords.max(axis=0))

                    cp_vals  = cp_grp[ds_key][()]
                    cfx_vals = cfx_grp[ds_key][()]
                    cfz_vals = cfz_grp[ds_key][()]
                    outputs  = np.stack([cp_vals, cfx_vals, cfz_vals], axis=-1)
                    all_outputs.append(outputs)

                if all_outputs:
                    all_outputs = np.concatenate(all_outputs, axis=0)
                    output_mean = all_outputs.mean(axis=0)
                    output_std  = all_outputs.std(axis=0)
                    output_std[output_std == 0] = 1
                else:
                    output_mean = np.zeros(3)
                    output_std  = np.ones(3)

            # Store norm stats
            norm_stats = {
                'shape_mean': shape_mean,
                'shape_std': shape_std,
                'flight_mean': flight_mean,
                'flight_std': flight_std,
                'coord_min': coord_min,
                'coord_max': coord_max,
                'output_mean': output_mean,
                'output_std': output_std
            }

        # Save them
        self.norm_stats = norm_stats
        self.shape_mean = norm_stats['shape_mean']
        self.shape_std  = norm_stats['shape_std']
        self.flight_mean = norm_stats['flight_mean']
        self.flight_std  = norm_stats['flight_std']
        self.coord_min   = norm_stats['coord_min']
        self.coord_max   = norm_stats['coord_max']
        self.output_mean = norm_stats['output_mean']
        self.output_std  = norm_stats['output_std']

        # ---- Open both HDF5 files (points & normals) ----
        self.h5f_points = h5py.File(hdf5_points_path, "r")
        self.points_grp = self.h5f_points["points"]
        self.cp_grp     = self.h5f_points["cp"]
        self.cfx_grp    = self.h5f_points["cf_x"]
        self.cfz_grp    = self.h5f_points["cf_z"]

        # NEW: open the normals file
        self.h5f_normals = h5py.File(hdf5_normals_path, "r")
        self.normals_grp = self.h5f_normals["normals"]  # group containing normal vectors

        # ---- Build the design-level structure as before ----
        tmp_designs = {}
        for row in df.itertuples(index=False):
            geom = row.geom_name
            tmp_designs.setdefault(geom, [])
            try:
                if "case_" in row.case_name:
                    num = int(row.case_name.split('_')[1])
                else:
                    num = int(row.case_name)
                ds_key = key_format.format(num=num)
            except Exception as e:
                print(f"Error processing case_name {row.case_name}: {e}")
                continue

            # Skip if not in the HDF5
            if ds_key not in self.points_grp:
                continue

            # Normalize flight+shape
            flight_cond_arr = np.array([row.alt_kft, row.Re_L, row.M_inf, row.alpha_deg], dtype=np.float32)
            shape_params    = np.array([row.B1, row.B2, row.B3, row.C2, row.C3, row.C4, row.S1, row.S2, row.S3], dtype=np.float32)
            norm_flight     = (flight_cond_arr - self.flight_mean) / self.flight_std
            norm_shape      = (shape_params - self.shape_mean) / self.shape_std
            full_cond       = np.concatenate([norm_flight, norm_shape]).astype(np.float32)

            # ---- Load and normalize coords (existing logic) ----
            pts_coords = self.points_grp[ds_key][()]
            pts_coords = 2 * (pts_coords - self.coord_min) / (self.coord_max - self.coord_min) - 1

            # ---- Load normals (no additional scaling) ----
            if ds_key in self.normals_grp:
                normals = self.normals_grp[ds_key][()]
                # They should have the same shape[0] as pts_coords
                if normals.shape[0] != pts_coords.shape[0]:
                    print(f"Warning: mismatch in # of points vs. normals for {ds_key}!")
                    continue
            else:
                # If it doesn't exist, create a placeholder
                normals = np.zeros_like(pts_coords)

            # Concatenate them: shape => (num_points, 6)
            pts_with_normals = np.concatenate([pts_coords, normals], axis=-1)

            # ---- Load & normalize outputs ----
            cp_vals   = (self.cp_grp[ds_key][()]   - self.output_mean[0]) / self.output_std[0]
            cfx_vals  = (self.cfx_grp[ds_key][()]  - self.output_mean[1]) / self.output_std[1]
            cfz_vals  = (self.cfz_grp[ds_key][()]  - self.output_mean[2]) / self.output_std[2]
            coeffs    = np.stack([cp_vals, cfx_vals, cfz_vals], axis=-1)

            tmp_designs[geom].append({
                'case_name': ds_key,
                'flight_cond': full_cond,
                'points': pts_with_normals.astype(np.float32),  # <--- store coords + normals
                'coeffs': coeffs.astype(np.float32)
            })

        unique_geom_csv = df['geom_name'].nunique()
        unique_geom_dataset = len(tmp_designs)
        print(f"Unique geometries in CSV: {unique_geom_csv}")
        print(f"Unique geometries in dataset after HDF5 filtering: {unique_geom_dataset}")

        # Keep only valid groups
        self.design_info = [cases for cases in tmp_designs.values() if len(cases) > 0]
        self.num_designs = len(self.design_info)

    def __len__(self):
        return self.num_designs

    def __getitem__(self, idx):
        return self.design_info[idx]

    def close(self):
        # Helper if you want to manually close files
        self.h5f_points.close()
        self.h5f_normals.close()


def design_collate_fn(batch_of_designs, n_points_per_design=5000):
    """
    Randomly selects one case from each design, subsamples points,
    and returns coords, conds, and coeffs as Tensors.
    Now coords has 6 dimensions (x, y, z, nx, ny, nz).
    """
    all_coords = []
    all_targets = []
    all_conds = []
    for design_item in batch_of_designs:
        chosen_idx = np.random.randint(0, len(design_item))
        chosen_data = design_item[chosen_idx]

        coords = chosen_data['points']   # shape (N, 6) now
        coeffs = chosen_data['coeffs']   # shape (N, 3)
        cond   = chosen_data['flight_cond']  # shape (13,)

        n_total_pts = coords.shape[0]
        if n_total_pts <= n_points_per_design:
            sub_idx = np.arange(n_total_pts)
        else:
            sub_idx = np.random.choice(n_total_pts, size=n_points_per_design, replace=False)

        coords_sub = coords[sub_idx]
        coeffs_sub = coeffs[sub_idx]
        cond_sub   = np.tile(cond[None, :], (coords_sub.shape[0], 1))

        all_coords.append(coords_sub)
        all_targets.append(coeffs_sub)
        all_conds.append(cond_sub)

    coords_batch  = torch.from_numpy(np.concatenate(all_coords, axis=0))
    targets_batch = torch.from_numpy(np.concatenate(all_targets, axis=0))
    conds_batch   = torch.from_numpy(np.concatenate(all_conds, axis=0))
    return coords_batch, conds_batch, targets_batch


def split_designs(full_dataset, train_ratio=0.9):
    """
    Splits the dataset at the design (geometry) level.
    """
    n_total = len(full_dataset)
    n_train = int(train_ratio * n_total)
    n_val   = n_total - n_train
    train_ds, val_ds = random_split(full_dataset,
                                    [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    return train_ds, val_ds

