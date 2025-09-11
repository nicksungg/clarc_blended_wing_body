# BlendedNet: FiLM-Based Aerodynamic Field Prediction

This repository contains the codebase used in the ASME IDETC 2025 paper:

**"BlendedNet: A Blended Wing Body Aircraft Dataset and Surrogate Model for Aerodynamic Predictions"**\
*Presented at ASME IDETC/CIE 2025, Anaheim, CA*

The project introduces a public high-fidelity dataset for blended wing body (BWB) aircraft, as well as a two-stage surrogate model combining PointNet and FiLM-based neural networks to predict pointwise aerodynamic coefficients. 

The dataset is publicly available at: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VJT9EP

Also, check out our paper on Arxiv: https://arxiv.org/abs/2509.07209



---

## 🚀 Highlights

- **999 BWB geometries** × \~10 flight cases → **8,830 high-fidelity simulations**
- Surface-level CFD quantities: **Cp, Cfx, Cfz** with corresponding point coordinates and normals
- **PointNet**-based encoder to recover geometric design parameters from a sampled surface
- **FiLM**-modulated neural field for predicting pointwise aerodynamic coefficients
- Detailed error metrics and R² plots included

---

## 📂 Repository Structure

```bash
.
├── models/
│   ├── film_model_v1.py                # Early FiLM model (ReLU, no residuals)
│   └── film_model_v2.py                # Final FiLM model (SIREN-style with sine + residuals)
├── dataset.py                          # Dataloading
│   train.ipynb  # Training pipeline (PointNet + FiLM)
│   test.ipynb # Evaluation
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Overview

BlendedNet is the first open dataset to provide high-resolution **pointwise aerodynamic surface coefficients** for BWB aircraft.

Each case contains:

- Geometric design parameters (9 shape parameters)
- Flight conditions (altitude, Mach, angle of attack, Reynolds length)
- CFD-derived outputs:
  - Cp (pressure coefficient)
  - Cf\_x, Cf\_z (skin friction in x/z)
  - Surface normals

Formats:

- `.csv` for metadata
- `.h5` for coordinates, normals, and coefficients
- `.vtk` for postprocessing and visualization

The dataset will be hosted on **Harvard Dataverse** (link pending).

---

## 🧠 Surrogate Model

### 1. **PointNet Regressor**

- Input: Sampled point cloud of the aircraft
- Output: 9 geometric shape parameters
- Permutation-invariant design

### 2. **FiLM Network**

- Input: 3D coordinates (+ normals), flight conditions, and shape parameters
- Output: Cp, Cf\_x, Cf\_z at each surface point
- Modulation via learned scale/shift (gamma, beta)
- Residual connections and sine activations

---

## 🔧 How to Run

### Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Train the Model

```bash
# Inside train.ipynb
```

### Evaluate

```bash
# test.ipynb

```

---

## 📈 Performance

### PointNet Parameter Prediction (R²)

| Parameter | R²     |
| --------- | ------ |
| C2/C1     | 0.9893 |
| C3/C1     | 0.9896 |
| C4/C1     | 0.9945 |
| B1/C1     | 0.9923 |
| B2/C1     | 0.9948 |
| B3/C1     | 0.9997 |
| S1        | 0.9968 |
| S2        | 0.9914 |
| S3        | 0.9973 |

### FiLM Prediction Errors (Test Set)

| Metric                                     | Cp       | Cfx      | Cfz      |
| ------------------------------------------ | -------- | -------- | -------- |
| **Conditioned on Ground Truth Parameters** |          |          |          |
| MSE                                        | 7.86e-03 | 2.80e-05 | 1.51e-05 |
| MAE                                        | 3.72e-02 | 1.35e-03 | 7.96e-04 |
| Rel L1 (%)                                 | 13.52%   | 22.09%   | 30.01%   |
| Rel L2 (%)                                 | 3.11%    | 7.74%    | 18.79%   |
| **Conditioned on Predicted Parameters**    |          |          |          |
| MSE                                        | 1.19e-02 | 1.82e-04 | 5.72e-05 |
| MAE                                        | 4.33e-02 | 1.98e-03 | 1.19e-03 |
| Rel L1 (%)                                 | 14.99%   | 24.03%   | 31.53%   |
| Rel L2 (%)                                 | 4.24%    | 16.78%   | 21.84%   |

---

## 📜 Citation

If you use this dataset or code, please cite:

```bibtex
@inproceedings{sung2025blendednet,
  title={BlendedNet: A Blended Wing Body Aircraft Dataset and Surrogate Model for Aerodynamic Predictions},
  author={Nicholas Sung and Steven Spreizer and Mohamed Elrefaie and Kaira Samuel and Matthew C. Jones and Faez Ahmed},
  booktitle={ASME IDETC/CIE},
  year={2025},
  address={Anaheim, CA},
  number={DETC2025-168977}
}
```

---

## 🛠 Acknowledgements

This material is based upon work supported under Air Force Contract No. FA8702-15-D-0001.

© 2025 Massachusetts Institute of Technology.

We also thank the MIT Lincoln Laboratory Supercomputing Center for their HPC resources.

---

## 📨 Contact

For questions, please contact:

- **Nicholas Sung**\
  Department of Mechanical Engineering, MIT\
  [nicksung@mit.edu](mailto\:nicksung@mit.edu)

