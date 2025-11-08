# %%
#!/usr/bin/env python3
"""
VTK -> HDF5 surface dataset builder (NumPy 2.0-compatible)

Reads VTK files named case_0000.vtk ... case_9999.vtk from DATA_DIR,
skips known-missing cases, extracts:
  - points (N,3)
  - normals (N,3)  [uses existing 'Normals' if present, otherwise computes]
  - cp      (N,)
  - cf_x/y/z (N,)

and writes them into groups in an HDF5:
  points/<case_id>, normals/<case_id>, cp/<case_id>, cf_x/<case_id>, cf_y/<case_id>, cf_z/<case_id>

If cp/cf arrays are missing or wrong-length, fills with NaNs (to preserve the case).
Per-case writing keeps memory usage flat.

Usage:
  python build_surface_hdf5.py
"""

import sys
import numpy as np
import h5py
from pathlib import Path

# PyVista/VTK
try:
    import pyvista as pv
except Exception as e:
    print("Error: pyvista is required. Install with: pip install pyvista vtk", file=sys.stderr)
    raise

# ---------------------------- CONFIG -----------------------------------------
# DATA_DIR = Path("/home/nicksung/Desktop/nicksung/bwb_full_v2/data/vtk")
# OUT_H5   = Path("/home/nicksung/Desktop/nicksung/bwb_full_v2/data/surface_data.hdf5")

CASE_START = 0
CASE_END   = 9999  # inclusive

# Provided missing cases (string suffixes, 4 digits)
MISSING_CASES = {
    }

# HDF5 compression/chunking settings
H5_COMP = {"compression": "gzip", "compression_opts": 4, "shuffle": True}

# ------------------------- HELPERS -------------------------------------------
def case_id(i: int) -> str:
    return f"case_{i:04d}"

def vtk_path_for(i: int) -> Path:
    return DATA_DIR / f"{case_id(i)}.vtk"

def _lower_keys(dataset) -> dict:
    """Map lower-cased point_data names -> original names for case-insensitive lookup."""
    mapping = {}
    for k in dataset.point_data.keys():
        mapping[k.lower()] = k
    return mapping

def _get_array(dset, names, expect_dim=1, npts=None):
    """
    Try to fetch first matching point_data array by (case-insensitive) name list.
    Returns float32 numpy array or None if not found/shape-mismatch.
    Accepts (N,), (N,1), (N,C), or transposes (C,N) when obvious.
    """
    keymap = _lower_keys(dset)
    for nm in names:
        k = keymap.get(nm.lower())
        if k is None:
            continue
        arr = np.asarray(dset.point_data[k])  # allow copy if needed (NumPy 2.0 safe)
        if arr is None:
            continue
        arr = arr.astype(np.float32, copy=False)

        if arr.ndim == 1:
            if (npts is not None) and (arr.shape[0] != npts):
                continue
            return arr

        if arr.ndim == 2:
            # Prefer (N, C)
            if npts is not None:
                if arr.shape[0] == npts:
                    if expect_dim == 1 and arr.shape[1] == 1:
                        return arr[:, 0]
                    if expect_dim in (1, arr.shape[1]):
                        return arr
                if arr.shape[1] == npts:
                    arr_t = arr.T
                    if expect_dim == 1 and arr_t.shape[1] == 1:
                        return arr_t[:, 0]
                    if expect_dim in (1, arr_t.shape[1]):
                        return arr_t
            else:
                return arr
    return None

def _get_vector3(dset, names, npts=None):
    """
    Fetch a 3-component vector field from point_data by name, shape (N,3).
    Accepts (N,3) or (3,N); returns float32 or None.
    """
    arr = _get_array(dset, names, expect_dim=3, npts=npts)
    if arr is None:
        return None
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr.astype(np.float32, copy=False)
    return None

def _compute_normals_poly(poly: pv.PolyData) -> np.ndarray:
    """Compute point normals on a PolyData surface and return (N,3) float32."""
    nmesh = poly.compute_normals(point_normals=True, cell_normals=False, inplace=False)
    norms = nmesh.point_data.get("Normals", None)
    if norms is None:
        for k in nmesh.point_data.keys():
            if k.lower() == "normals":
                norms = nmesh.point_data[k]
                break
    if norms is None:
        raise RuntimeError("Failed to compute normals on polydata.")
    return np.asarray(norms, dtype=np.float32)  # allow copy if needed

def ensure_groups(h5):
    for gname in ["points", "normals", "cp", "cf_x", "cf_y", "cf_z", "meta"]:
        if gname not in h5:
            h5.create_group(gname)

def write_case(h5, key: str, points: np.ndarray, normals: np.ndarray,
               cp: np.ndarray, cf_x: np.ndarray, cf_y: np.ndarray, cf_z: np.ndarray):
    """Create datasets for one case under each group."""
    h5["points"].create_dataset(key, data=points, chunks=True, **H5_COMP)
    h5["normals"].create_dataset(key, data=normals, chunks=True, **H5_COMP)
    h5["cp"].create_dataset(key, data=cp, chunks=True, **H5_COMP)
    h5["cf_x"].create_dataset(key, data=cf_x, chunks=True, **H5_COMP)
    h5["cf_y"].create_dataset(key, data=cf_y, chunks=True, **H5_COMP)
    h5["cf_z"].create_dataset(key, data=cf_z, chunks=True, **H5_COMP)

# --------------------------- MAIN --------------------------------------------
def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_H5.parent.mkdir(parents=True, exist_ok=True)

    not_found = []   # files genuinely missing on disk
    failed = []      # cases that raised exceptions during processing
    written = 0

    with h5py.File(OUT_H5, "w") as h5:
        ensure_groups(h5)
        h5.attrs["source_dir"] = str(DATA_DIR)
        h5.attrs["builder"] = "vtk->hdf5 surface_data"
        h5.attrs["notes"] = "points/normals/cp/cf_x/cf_y/cf_z per case; float32; gzip compressed"

        for i in range(CASE_START, CASE_END + 1):
            cid = case_id(i)
            suffix = f"{i:04d}"
            if suffix in MISSING_CASES:
                # listed-missing; track only in meta (not 'not_found')
                continue

            vtk_path = vtk_path_for(i)
            if not vtk_path.exists():
                not_found.append(cid)
                continue

            try:
                mesh = pv.read(vtk_path)

                # Ensure PolyData surface
                if isinstance(mesh, pv.PolyData):
                    surf = mesh
                else:
                    try:
                        surf = mesh.extract_surface()
                    except Exception:
                        surf = pv.wrap(mesh)

                # Coordinates
                points = np.asarray(surf.points, dtype=np.float32)  # NumPy 2.0 safe
                npts = points.shape[0]
                if npts == 0:
                    raise RuntimeError("No points in surface.")

                # Normals: prefer existing, else compute
                normals = None
                keymap = _lower_keys(surf)
                for nm in ["Normals", "Normal", "normals", "normal"]:
                    k = keymap.get(nm.lower())
                    if k is not None:
                        arr = np.asarray(surf.point_data[k])
                        if (arr is not None) and (arr.shape[0] == npts) and (arr.shape[-1] == 3):
                            normals = np.asarray(arr, dtype=np.float32)
                            break
                if normals is None:
                    normals = _compute_normals_poly(surf)

                # ----- cp (scalar) -----
                cp = _get_array(
                    surf,
                    names=["cp", "Cp", "CP", "pressure_coefficient", "pressurecoefficient", "c_p", "C_p"],
                    expect_dim=1,
                    npts=npts,
                )
                if cp is None:
                    cp = _get_array(
                        mesh,
                        names=["cp", "Cp", "CP", "pressure_coefficient", "pressurecoefficient", "c_p", "C_p"],
                        expect_dim=1,
                        npts=npts,
                    )
                if cp is None:
                    cp = np.full((npts,), np.float32(np.nan), dtype=np.float32)

                # ----- cf (vector or components) -----
                cf_vec = _get_vector3(
                    surf,
                    names=["cf", "Cf", "CF", "wallShear", "wall_shear", "tau_w", "wallShearStress"],
                    npts=npts,
                )
                if cf_vec is None:
                    cf_vec = _get_vector3(
                        mesh,
                        names=["cf", "Cf", "CF", "wallShear", "wall_shear", "tau_w", "wallShearStress"],
                        npts=npts,
                    )

                if cf_vec is not None:
                    cf_x, cf_y, cf_z = cf_vec[:, 0], cf_vec[:, 1], cf_vec[:, 2]
                else:
                    # Try per-component names
                    cf_x = _get_array(surf,
                        names=["cf_x", "Cf_x", "CF_X", "wallShearX", "tau_w_x", "wall_shear_x"],
                        expect_dim=1, npts=npts)
                    cf_y = _get_array(surf,
                        names=["cf_y", "Cf_y", "CF_Y", "wallShearY", "tau_w_y", "wall_shear_y"],
                        expect_dim=1, npts=npts)
                    cf_z = _get_array(surf,
                        names=["cf_z", "Cf_z", "CF_Z", "wallShearZ", "tau_w_z", "wall_shear_z"],
                        expect_dim=1, npts=npts)

                    # Fallback to original mesh if any are None
                    if cf_x is None:
                        cf_x = _get_array(mesh,
                            names=["cf_x", "Cf_x", "CF_X", "wallShearX", "tau_w_x", "wall_shear_x"],
                            expect_dim=1, npts=npts)
                    if cf_y is None:
                        cf_y = _get_array(mesh,
                            names=["cf_y", "Cf_y", "CF_Y", "wallShearY", "tau_w_y", "wall_shear_y"],
                            expect_dim=1, npts=npts)
                    if cf_z is None:
                        cf_z = _get_array(mesh,
                            names=["cf_z", "Cf_z", "CF_Z", "wallShearZ", "tau_w_z", "wall_shear_z"],
                            expect_dim=1, npts=npts)

                    # Fill remaining missing with NaN
                    if cf_x is None: cf_x = np.full((npts,), np.float32(np.nan), dtype=np.float32)
                    if cf_y is None: cf_y = np.full((npts,), np.float32(np.nan), dtype=np.float32)
                    if cf_z is None: cf_z = np.full((npts,), np.float32(np.nan), dtype=np.float32)

                # Final type/shape assertions (NumPy 2.0 safe)
                points  = points.astype(np.float32, copy=False)
                normals = np.asarray(normals, dtype=np.float32)
                cp      = np.asarray(cp, dtype=np.float32)
                cf_x    = np.asarray(cf_x, dtype=np.float32)
                cf_y    = np.asarray(cf_y, dtype=np.float32)
                cf_z    = np.asarray(cf_z, dtype=np.float32)

                if points.shape != normals.shape:
                    raise RuntimeError(f"Normals shape {normals.shape} does not match points {points.shape}")
                if cp.shape != (npts,):
                    raise RuntimeError(f"cp shape {cp.shape} does not match (N,) with N={npts}")
                for name, arr in [("cf_x", cf_x), ("cf_y", cf_y), ("cf_z", cf_z)]:
                    if arr.shape != (npts,):
                        raise RuntimeError(f"{name} shape {arr.shape} does not match (N,) with N={npts}")

                # Write to HDF5
                write_case(h5, cid, points, normals, cp, cf_x, cf_y, cf_z)
                written += 1
                if written % 50 == 0:
                    print(f"Wrote {written} cases…")

            except Exception as ex:
                failed.append(f"{cid}: {ex}")

        # Save meta lists
        meta = h5["meta"]
        dt = h5py.string_dtype(encoding="utf-8")
        meta.create_dataset("listed_missing_cases", data=np.array(sorted(list(MISSING_CASES)), dtype=dt))
        meta.create_dataset("not_found_files", data=np.array(not_found, dtype=dt))
        meta.create_dataset("failed_cases", data=np.array(failed, dtype=dt))

        h5.attrs["written_cases"] = written
        h5.attrs["total_expected"] = CASE_END - CASE_START + 1 - len(MISSING_CASES)

    print("\n==== SUMMARY ====")
    print(f"HDF5 written to: {OUT_H5}")
    print(f"Cases written:   {written}")
    print(f"Listed missing:  {len(MISSING_CASES)}")
    print(f"Not found:       {len(not_found)}")
    print(f"Failed:          {len(failed)}")
    if failed:
        print("First few failures:")
        for line in failed[:5]:
            print("  -", line)

if __name__ == "__main__":
    main()



