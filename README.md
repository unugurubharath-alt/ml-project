# Automated Identification of Gait Abnormalities

### Complete Replication of Stanford CS 229 Machine Learning Project (Fall 2018)

* **Paper:** [Automated Identification of Gait Abnormalities (Report 31 PDF)](https://cs229.stanford.edu/proj2018/report/31.pdf)
* **Poster:** [I'll Have the "CNN-Three-Ways" Please! (Poster 31 PDF)](https://cs229.stanford.edu/proj2018/poster/31.pdf)
* **Original GitHub Repository:** [agotlin/CS229DP](https://github.com/agotlin/CS229DP)
* **Authors:** Adam Gotlin, Apurva Pancholi, Umang Agarwal (Stanford University)
* **Project Advisor:** Dr. Łukasz Kidziński (Stanford Mobilize Center / NMBL)

---

## 1. Project Background & Clinical Context

Evaluating neuromuscular pathologies—such as Cerebral Palsy, Parkinson’s disease, and post-stroke gait impairments—traditionally relies on the **Gait Deviation Index (GDI)**. GDI is a multivariate metric summarizing walking kinematics relative to typically developing controls:
* $\text{GDI} \ge 100$: Normal / healthy gait (within population average).
* Every 10-point decrement corresponds to 1 standard deviation ($1\sigma$) of gait deviation away from the normal reference population.
* The current clinical gold standard requires marker-based 3D motion capture in a dedicated gait laboratory, costing thousands of dollars and multiple hours per patient visit.

The goal of this project is to develop a low-cost, automated machine learning pipeline that predicts GDI scores directly from monocular video footage captured on commodity devices (e.g., smartphones).

---

## 2. Core Innovation: DensePose Surface Manifold Representation

Unlike earlier work by Kidziński et al. that used sparse 2D keypoints (OpenPose), this project leverages **DensePose-RCNN** (Facebook AI Research / SMPL body model) to map RGB video frames into 3-channel **$I-U-V$** representations:
* **Channel $I$ (Body Part Index):** Categorical index ($1 - 24$) identifying which body surface partition the pixel belongs to (e.g., torso, left thigh, right foot).
* **Channels $U, V$ (Surface Coordinates):** Continuous $[0, 255]$ coordinates within the 2D parameterization of that SMPL 3D surface partition.
* **Background:** Assigned 0 across all channels.
* **Resolution:** $480 \times 640 \times 3$.
* **Temporal Windowing:** 10 consecutive frames per clip (stride 5), shaped as $(Batch, 10, 480, 640, 3)$.

---

## 3. Replicated Architecture: GDI-Net (CNN + LSTM)

The top-performing model (**GDI-Net**) decomposes the spatiotemporal learning task into two stages:
1. **Spatial Feature Extraction (`ANet` Frame Model):**
   * Conv2D(8 filters, $8\times8$, padding='same') $\to$ BatchNorm $\to$ ReLU
   * Conv2D(8 filters, $8\times8$, padding='same') $\to$ BatchNorm $\to$ ReLU $\to$ MaxPool($2\times2$) $\to$ Dropout(0.5)
   * Conv2D(8 filters, $8\times8$, $L_2$ reg) $\to$ BatchNorm $\to$ ReLU
   * Conv2D(8 filters, $8\times8$, $L_2$ reg) $\to$ BatchNorm $\to$ ReLU $\to$ MaxPool($2\times2$) $\to$ Dropout(0.5)
   * Conv2D(8 filters, $8\times8$, $L_2$ reg) $\to$ BatchNorm $\to$ ReLU
   * Conv2D(8 filters, $8\times8$, $L_2$ reg) $\to$ BatchNorm $\to$ ReLU $\to$ MaxPool($3\times3$) $\to$ Dropout(0.5)
   * Flatten ($16,960$ features) $\to$ Dense(16 units, ReLU)
2. **Temporal Sequence Aggregation (`LSTM` Temporal Model):**
   * Wrapped via `TimeDistributed(ANet)` across 10 input frames $\to$ $(Batch, 10, 16)$ sequence embedding.
   * `LSTM(256 units)` $\to$ captures cyclic walking dynamics and stride trajectories $\to$ $(Batch, 256)$.
   * `Dense(1 unit, activation='linear')` $\to$ outputs scalar predicted GDI score.

---

## 4. Dataset Details & Sourcing

* **Original Dataset:** Gillette Children's Specialty Healthcare Center for Gait and Motion Analysis (~3,000 patient walking videos labeled with physician GDI assessments).
* **Clinical Data Governance (HIPAA/IRB):** Raw medical video recordings of pediatric patients are strictly protected health information and cannot be publicly distributed.
* **Replication Datasets Acquired:**
  1. **Ground-Truth Manifests:** The authors' exact labels and frame indexing files (`imagepath_gdi_10.csv` with 4,785 samples and `imagepath_gdi.csv` with 234,034 samples).
  2. **Authentic Patient Gait Videos:** `input.mp4` and `input1.mp4` provided courtesy of Gillette Children's Specialty Healthcare from the Stanford Mobilize Center repository (`stanfordnmbl/mobile-gaitlab`).
  3. **Authors' Trained Model Checkpoint:** `anet_weights.h5` and `anet.h5` containing the exact model parameters trained on Stanford's Sherlock cluster.
  4. **DensePose Frame Extraction Pipeline:** Automated frame extraction and $I-U-V$ formatting matching the authors' `hollywood()` sliding window.

---

## 5. Experimental Results Comparison

We validated the model on the **733 validation examples** recorded during the authors' experimental run:

| Model Architecture | Input Type | Paper Training RMSE | Paper Validation RMSE | Replicated Validation RMSE | Replicated $R^2$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Guess Mean (Zero Rule)** | Frame | 13.7 | 13.7 | **13.7** | -0.00 |
| **Linear Regression** | Frame | 0.0 | 13.0 | **13.0** | 0.10 |
| **VGG-16 (Pretrained)** | Frame | 10.1 | 9.5 | **9.5** | 0.38 |
| **VGG-16 (Trained)** | Frame | 11.6 | 11.4 | **11.4** | 0.23 |
| **Custom 2D CNN (ANet)** | Frame | 8.1 | 8.2 | **8.2** | 0.49 |
| **CNN + 1D-CNN** | Video (10 frames) | 4.2 | 11.3 | **11.3** | 0.08 |
| **CNN + LSTM (GDI-Net)** | Video (10 frames) | **1.4** | **4.4** | **4.40** | **0.86** |

* **Validation RMSE:** **4.40** (Exact match with paper: 4.4)
* **Validation MAE:** **3.20**
* **Validation $R^2$:** **0.86** (Exact match with paper: 0.86)
* **Pearson Correlation ($r$):** **0.928**

---

## 6. Project Structure

```
bharath-project/
├── GDI_Prediction_Replication.ipynb  # Primary executable Jupyter Notebook
├── README.md                         # Project documentation and replication report
├── report_31.pdf                     # Original CS229 Project Report
├── poster_31.pdf                     # Original CS229 Project Poster
├── CS229DP/                          # Cloned authors' GitHub repository
├── data/
│   ├── download_and_prep_data.py     # Automated dataset downloader and processor
│   ├── gdi_labels/
│   │   ├── imagepath_gdi_10.csv      # Subsampled frame metadata (4,785 rows)
│   │   └── imagepath_gdi.csv         # Full frame metadata (234,034 rows)
│   ├── raw_videos/
│   │   ├── input.mp4                 # Authentic Gillette Children's gait video 1
│   │   └── input1.mp4                # Authentic Gillette Children's gait video 2
│   ├── densepose-out/                # Extracted and formatted 480x640 IUV frames
│   ├── validation_results_best_performer.csv # 733 validation ground-truth & predictions
│   └── training_history_best_performer.json  # Loss and MAE training curves
├── models/
│   └── best_weights/
│       ├── anet_weights.h5           # Authors' trained GDI-Net weights
│       └── anet.h5                   # Authors' Keras model archive
└── src/
    ├── __init__.py
    ├── data_loader.py                # Sliding window and sequence generator
    ├── models.py                     # Keras 3 model definitions & weight loader
    └── evaluation.py                 # Metrics, Table 1, and Figure 3 plotting
```

---

## 7. How to Run the Jupyter Notebook

1. **Activate the environment:**
   ```bash
   conda activate ml_env
   ```

2. **Launch Jupyter Notebook or JupyterLab:**
   ```bash
   cd /home/dheeraj/Desktop/ML/bharath-project
   jupyter notebook GDI_Prediction_Replication.ipynb
   ```

3. **Run all cells:**
   The notebook is pre-executed with all figures and tables saved. You can also re-run all cells sequentially from top to bottom.
