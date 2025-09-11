import h5py, pandas as pd, numpy as np, torch, random
from torch.utils.data import Dataset, random_split

# Reproducibility
random.seed(0); np.random.seed(0); torch.manual_seed(0)

class UnifiedDesignDataset(Dataset):
    """
    Reads your new CSV + one HDF5 (points/normals/cp/cf_x/cf_z).
    Groups by 'mesh' (each mesh is a design with >=1 cases).

    Returns each design as a list of case dicts:
      {
        'case_name': 'case_0000',
        'mesh': 'mesh_0000',
        'flight_cond': (13,),        # normed [3 flight + 10 shape]
        'points': (N, 6),            # coords normalized to [-1,1] + raw normals
        'coeffs': (N, 3),            # normed [cp, cfx, cfz]
      }
    """
    def __init__(self,
                 csv_path: str,
                 hdf5_path: str,
                 norm_stats: dict | None = None,
                 mode: str = "train",
                 include_alt_kft: bool = False,
                 include_beta_deg: bool = False):
        super().__init__()
        self.mode = mode

        # Columns (fixed to your spec)
        self.flight_base = ['Re_L','M_inf','alpha_deg']
        extra = []
        if include_alt_kft:  extra.append('alt_kft')
        if include_beta_deg: extra.append('beta_deg')
        self.flight_cols = self.flight_base + extra        # default: 3
        self.shape_cols  = ['B1','B2','B3','C1','C2','C3','C4','S1','S3','X3']  # 10
        self.meta_cols   = ['case_name','mesh','CD','CL','CMy']                 # kept but unused in cond

        expected = set(['case_name','mesh','alt_kft','Re_L','M_inf','alpha_deg','beta_deg',
                        'CD','CL','CMy','B1','B2','B3','C1','C2','C3','C4','S1','S3','X3'])
        df = pd.read_csv(csv_path)
        missing = sorted(list(expected - set(df.columns)))
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        # Normalize/standardize key format
        def to_case_key(val: str) -> str:
            if isinstance(val, str) and val.startswith("case_"): return val.lower()
            # allow numeric strings like "7"
            num = int(val)
            # default to 4-digit train key
            return f"case_{num:04d}" if self.mode != "test" else f"case_{num:03d}"

        df['case_key'] = df['case_name'].apply(to_case_key)
        self.csv = df

        # Open HDF5
        self.h5f = h5py.File(hdf5_path, "r")
        try:
            self.points_grp  = self.h5f["points"]
            self.normals_grp = self.h5f["normals"]
            self.cp_grp      = self.h5f["cp"]
            self.cfx_grp     = self.h5f["cf_x"]
            # cf_y is optional; not used
            self.cfz_grp     = self.h5f["cf_z"]
        except KeyError as e:
            raise KeyError(f"HDF5 missing group: {e}")

        # ----------------- compute or load normalization stats -----------------
        if norm_stats is None:
            # flight/shape stats from CSV
            flight_mean = df[self.flight_cols].mean().values.astype(np.float32)
            flight_std  = df[self.flight_cols].std().values.astype(np.float32)
            shape_mean  = df[self.shape_cols].mean().values.astype(np.float32)
            shape_std   = df[self.shape_cols].std().values.astype(np.float32)
            flight_std[flight_std == 0] = 1.0
            shape_std[shape_std == 0]   = 1.0

            # coord min/max + output mean/std from HDF5 (cp, cfx, cfz)
            coord_min = np.array([ np.inf,  np.inf,  np.inf], dtype=np.float32)
            coord_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
            out_accum = []

            for row in df.itertuples(index=False):
                k = row.case_key
                if k not in self.points_grp: continue
                P = self.points_grp[k][()]    # (N,3)
                if P.size == 0: continue
                coord_min = np.minimum(coord_min, P.min(axis=0))
                coord_max = np.maximum(coord_max, P.max(axis=0))

                cp  = self.cp_grp[k][()]
                cfx = self.cfx_grp[k][()]
                cfz = self.cfz_grp[k][()]
                out_accum.append(np.stack([cp, cfx, cfz], axis=-1))

            if out_accum:
                outs = np.concatenate(out_accum, axis=0)
                output_mean = outs.mean(axis=0).astype(np.float32)
                output_std  = outs.std(axis=0).astype(np.float32)
                output_std[output_std == 0] = 1.0
            else:
                output_mean = np.zeros(3, dtype=np.float32)
                output_std  = np.ones(3,  dtype=np.float32)

            norm_stats = dict(
                flight_mean=flight_mean, flight_std=flight_std,
                shape_mean=shape_mean,   shape_std=shape_std,
                coord_min=coord_min,     coord_max=coord_max,
                output_mean=output_mean, output_std=output_std
            )

        self.norm_stats  = norm_stats
        self.flight_mean = norm_stats['flight_mean']; self.flight_std  = norm_stats['flight_std']
        self.shape_mean  = norm_stats['shape_mean'];  self.shape_std   = norm_stats['shape_std']
        self.coord_min   = norm_stats['coord_min'];   self.coord_max   = norm_stats['coord_max']
        self.output_mean = norm_stats['output_mean']; self.output_std  = norm_stats['output_std']

        # ----------------- build designs grouped by 'mesh' -----------------
        groups = {}
        skipped = 0
        for row in df.itertuples(index=False):
            k = row.case_key
            if k not in self.points_grp: 
                skipped += 1
                continue

            # build conditioning vector (flight + shape)
            flight = np.array([getattr(row, c) for c in self.flight_cols], dtype=np.float32)
            shape  = np.array([getattr(row, c) for c in self.shape_cols],  dtype=np.float32)
            nf = (flight - self.flight_mean) / self.flight_std
            ns = (shape  - self.shape_mean)  / self.shape_std
            cond = np.concatenate([nf, ns]).astype(np.float32)  # (13 + any extras)

            # coords normalized to [-1,1], normals raw
            P  = self.points_grp[k][()]          # (N,3)
            N  = self.normals_grp[k][()] if k in self.normals_grp else np.zeros_like(P, dtype=np.float32)
            if P.shape[0] == 0 or N.shape[0] != P.shape[0]: 
                skipped += 1
                continue
            Pn = 2.0 * (P - self.coord_min) / (self.coord_max - self.coord_min + 1e-12) - 1.0
            coords6 = np.concatenate([Pn, N], axis=-1).astype(np.float32)  # (N,6)

            # outputs normalized: (cp, cfx, cfz)
            cp  = (self.cp_grp[k][()]  - self.output_mean[0]) / self.output_std[0]
            cfx = (self.cfx_grp[k][()] - self.output_mean[1]) / self.output_std[1]
            cfz = (self.cfz_grp[k][()] - self.output_mean[2]) / self.output_std[2]
            Y   = np.stack([cp, cfx, cfz], axis=-1).astype(np.float32)

            mesh_id = getattr(row, 'mesh')
            groups.setdefault(mesh_id, []).append(dict(
                case_name=k,
                mesh=mesh_id,
                flight_cond=cond,
                points=coords6,
                coeffs=Y,
                # meta if you want it:
                CD=float(getattr(row,'CD')), CL=float(getattr(row,'CL')), CMy=float(getattr(row,'CMy'))
            ))

        self.designs = [v for v in groups.values() if len(v) > 0]
        self.num_designs = len(self.designs)
        print(f"[dataset_v4] designs kept: {self.num_designs}, skipped rows: {skipped}")

        # expose dims for callers
        self.cond_dim   = self.designs and self.designs[0][0]['flight_cond'].shape[0] or (13 + len(extra))
        self.coord_dim  = 6
        self.output_dim = 3

    def __len__(self):
        return self.num_designs

    def __getitem__(self, idx):
        return self.designs[idx]

    def close(self):
        self.h5f.close()


def design_collate_fn(batch_of_designs, n_points_per_design=5000):
    """
    From each design, pick one case, subsample points, and stack.
    Returns:
      coords:  (B*n, 6)
      conds:   (B*n, cond_dim)
      targets: (B*n, 3)
    """
    all_coords, all_targets, all_conds = [], [], []
    for design in batch_of_designs:
        j = np.random.randint(0, len(design))
        sample = design[j]
        coords, coeffs, cond = sample['points'], sample['coeffs'], sample['flight_cond']

        N = coords.shape[0]
        idx = np.arange(N) if N <= n_points_per_design else np.random.choice(N, n_points_per_design, replace=False)
        c_sub  = coords[idx]
        y_sub  = coeffs[idx]
        cond_b = np.tile(cond[None, :], (c_sub.shape[0], 1))

        all_coords.append(c_sub)
        all_targets.append(y_sub)
        all_conds.append(cond_b)

    coords_batch  = torch.from_numpy(np.concatenate(all_coords, axis=0))
    targets_batch = torch.from_numpy(np.concatenate(all_targets, axis=0))
    conds_batch   = torch.from_numpy(np.concatenate(all_conds, axis=0))
    return coords_batch, conds_batch, targets_batch


def split_designs(full_dataset, train_ratio=0.9):
    n_total = len(full_dataset)
    n_train = int(train_ratio * n_total)
    n_val   = n_total - n_train
    return random_split(full_dataset,
                        [n_train, n_val],
                        generator=torch.Generator().manual_seed(42))

