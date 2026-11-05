# 💳 Credit Card Fraud Detection: End-to-End Machine Learning System
> **Technical Architecture, Imbalance Diagnostics, Cost-Calibrated Thresholding & SHAP Explainability**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![XGBoost: 2.0+](https://img.shields.io/badge/XGBoost-2.0%2B-red.svg)](https://xgboost.readthedocs.io/)
[![SHAP: TreeExplainer](https://img.shields.io/badge/SHAP-TreeExplainer-green.svg)](https://shap.readthedocs.io/)
[![FastAPI: Async](https://img.shields.io/badge/FastAPI-Production%20Ready-teal.svg)](https://fastapi.tiangolo.com/)

---

## 🎙️ The 60-Second Opening Pitch
> *"In credit card fraud detection, our central challenge is extreme class imbalance: only **0.172%** of transactions are fraudulent (492 out of 284,807). A naive baseline predicting genuine for all transactions achieves a deceptive **99.83% accuracy** while stopping **$0** of fraud. In this project, we engineered an end-to-end, leakage-free machine learning pipeline benchmarking Logistic Regression, Random Forest, and XGBoost. By replacing misleading metrics with **PR-AUC**, diagnosing and resolving **XGBoost gradient saturation**, calibrating our operating threshold via a **business cost model ($t^* = 0.1200$)**, and generating **SHAP explainability**, our final model captures **81.7% of frauds with 79.5% precision**, supported by an interactive Web Application (HTML5 / CSS3 / Vanilla JS + FastAPI)."*

---

## 📂 Repository File Structure
```
credit-card-fraud-detection/
├── data/                        # gitignored — place creditcard.csv here
│   └── creditcard.csv           # ULB Kaggle raw dataset (284,807 rows)
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory Data Analysis & Imbalance Diagnostics
│   └── 02_modeling.ipynb        # Full Leakage-Free Modelling Pipeline
├── src/
│   ├── __init__.py
│   ├── data.py                  # Ingestion, audit, deduplication, 70/15/15 stratified split
│   ├── features.py              # StandardScaler ColumnTransformer & ImbPipelines
│   ├── train.py                 # 5-fold Stratified CV benchmark & Champion training
│   ├── threshold.py             # Banking cost model & threshold calibration (t* = 0.1200)
│   ├── evaluate.py              # Unbiased held-out test evaluation & confusion matrices
│   └── explain.py               # SHAP TreeExplainer (global beeswarm & local waterfalls)
├── reports/
│   ├── results.csv              # Generated CV model comparison benchmark table
│   └── figures/                 # All evaluation plots (PNG, 150 DPI)
│       ├── class_imbalance.png
│       ├── temporal_dynamics.png
│       ├── monetary_distribution.png
│       ├── top_drivers_correlation.png
│       ├── subspace_separation.png
│       ├── cv_benchmark_pr_f2.png
│       ├── cost_curve.png
│       ├── test_pr_roc_cm.png
│       └── shap_summary.png
├── models/
│   ├── final_model.joblib       # Serialized champion pipeline (Scaler + Tuned XGBoost)
│   └── threshold_config.json    # Calibrated operating threshold & metadata
├── app/                         # Interactive Web Application (HTML / CSS / JS)
│   ├── app.py                   # High-throughput asynchronous FastAPI backend
│   ├── index.html               # Frontend dashboard template
│   ├── style.css                # Fintech cyber-security dark styling
│   ├── script.js                # Real-time risk scoring, SHAP waterfall & simulator
│   ├── static/                  # Static assets mount
│   └── templates/               # Jinja2 template mount
├── .gitignore
├── LICENSE                      # MIT
├── README.md                    # Project documentation & presentation guide
├── requirements.txt             # Pinned dependency versions
└── run_pipeline.py              # Automated single-command reproduction script
```

---

## 🗺️ Pipeline Architecture Roadmap
```
[Raw Dataset (284,807 rows)]
         │
         ▼
[Data Integrity & Hygiene Audit] ──► 0 nulls, drop 1,081 duplicates (283,726 clean rows)
         │
         ▼
[Stratified Split (70 / 15 / 15)] ──► Train (70%), Validation (15%), Held-Out Test (15%)
         │
         ▼
[Leakage-Free Preprocessing] ──────► StandardScaler on Amount & Time strictly inside folds; V1-V28 passthrough
         │
         ▼
[5-Fold Stratified CV Benchmark] ──► Dummy vs Logistic Reg vs Random Forest vs XGBoost (SMOTE, RUS, Weights)
         │
         ▼
[Cost-Based Threshold Calibration] ─► Minimize Total Cost = Σ(FN * Amount) + (FP * $5) -> t* = 0.1200
         │
         ▼
[Held-Out Test Evaluation] ────────► PR-AUC: 0.8161 | Recall: 81.69% | Precision: 79.45% | F2: 0.8123
         │
         ▼
[Explainability & Delivery] ───────► SHAP TreeExplainer (V14, V4, V12 drivers) + HTML/CSS/JS Web App
```

---

## 📊 5-Fold Stratified Cross-Validation Benchmark
*Reproducing Section 4 of the Presentation Master Guide:*

| Model Architecture & Strategy | PR-AUC (mean ± std) | ROC-AUC (mean ± std) | F2 Score (mean ± std) |
| :--- | :--- | :--- | :--- |
| **Dummy (Majority Class)** | $0.0017 \pm 0.0000$ | $0.5000 \pm 0.0000$ | $0.0000 \pm 0.0000$ |
| **Logistic Regression (Balanced)** | $0.7316 \pm 0.0329$ | $0.9831 \pm 0.0112$ | $0.2357 \pm 0.0077$ |
| **Random Forest (No Resampling)** | $0.8419 \pm 0.0544$ | $0.9538 \pm 0.0257$ | $0.7976 \pm 0.0321$ |
| **Random Forest + SMOTE** | $0.8434 \pm 0.0548$ | $0.9779 \pm 0.0154$ | $0.8186 \pm 0.0493$ |
| **Random Forest + RUS** | $0.7361 \pm 0.0130$ | $0.9768 \pm 0.0129$ | $0.2287 \pm 0.0279$ |
| **XGBoost (Unweighted / Default)** | $0.2817 \pm 0.3404$ *(Collapse)* | $0.6733 \pm 0.1728$ | $0.3286 \pm 0.3263$ |
| **XGBoost (scale_pos_weight)** | $0.8444 \pm 0.0582$ | $0.9788 \pm 0.0139$ | $0.8227 \pm 0.0465$ |
| **XGBoost + SMOTE** | $0.8463 \pm 0.0485$ | $0.9813 \pm 0.0113$ | $0.8090 \pm 0.0454$ |
| **XGBoost (Tuned via RandomizedSearch)** | **$0.8527 \pm 0.0554$** | **$0.9829 \pm 0.0092$** | **$0.8297 \pm 0.0408$** |

---

## 🔍 Key Technical Insight: XGBoost Gradient Saturation Failure
Unweighted XGBoost collapsed on Folds 2–4 with individual PR-AUCs falling to $\approx 0.002$. **This is not a bug**:
* With $160,000$ negative vs $\approx 330$ positive gradients per training fold at learning rate $\eta = 0.3$, negative gradient updates pushed logits (margins) below $-80$:
  $$z = \sum f_m(x) < -80$$
* The logistic sigmoid saturated into numerical subnormals:
  $$\sigma(z) = \frac{1}{1 + e^{-z}} \approx \frac{1}{e^{80}} \approx 3 \times 10^{-35}$$
* This destroyed probability rank-ordering. Adding:
  $$\text{scale\_pos\_weight} = \frac{N_{\text{negative}}}{N_{\text{positive}}} \approx 576.35$$
  balanced negative and positive gradient accumulation, instantly restoring and stabilizing PR-AUC to **$0.8444$**.

---

## 💰 Cost-Based Decision Threshold Optimization
Standard models apply an arbitrary cutoff at $t = 0.50$. In banking, error costs are **asymmetric**:
$$\text{Total Cost}(t) = \sum_{i \in \text{FN}} (\text{FN\_Multiplier} \times \text{Amount}_i) + (\text{FP\_Cost} \times N_{\text{FP}})$$
Where $\text{FN\_Multiplier} = 1.0$ (direct stolen capital) and $\text{FP\_Cost} = \$5.00$ (analyst triage & verification SMS).

Minimizing this cost function on the **$15\%$ Validation Set** locates the optimal threshold: **$t^* = 0.1200$**.

### Unbiased Evaluation on Held-Out Test Set ($42,559$ transactions, $71$ frauds):
| Evaluation Metric | Tuned Threshold ($t^* = 0.1200$) | Default Threshold ($t = 0.50$) | Operational Impact |
| :--- | :--- | :--- | :--- |
| **Frauds Caught (Recall)** | **$58 / 71$ ($81.69\%$)** | $55 / 71$ ($77.46\%$) | **$+3$ additional frauds stopped** |
| **Missed Frauds (FN)** | **$13$** | $16$ | **$18.7\%$ reduction in breaches** |
| **False Alarms (FP)** | **$15$** (Precision = $79.45\%$) | $7$ (Precision = $88.71\%$) | $+8$ reviews required ($\$40$ cost) |
| **F2 Score (Recall-weighted)** | **$0.8123$** | $0.7948$ | Higher operational protection |
| **PR-AUC** | **$0.8161$** | $0.8161$ | Exceeds random guess ($0.0017$) |

---

## 🧠 Explainability with SHAP (TreeExplainer)
1. **Global Drivers:** Features **$V_{14}, V_4, V_{12}, V_{10}, V_{17}$** dominate fraud predictions. Strongly **negative values of $V_{14}$ and $V_{12}$** drastically increase the model's fraud probability score.
2. **Local Waterfall Auditing:** Individual transaction breakdowns explain how features compound to cross the $0.1200$ threshold for True Positives, explain False Positives, and audit False Negatives.
3. **Domain Caveat:** $V_1$–$V_{28}$ are PCA components, proving statistical signal stability; associating real-world tags (e.g. merchant category) requires the pre-PCA bank mapping.

---

## 🌐 Interactive Web Application (HTML / CSS / JS + FastAPI)
Instead of Streamlit, the system includes a high-performance **FastAPI backend with a vanilla HTML5 / CSS3 / JavaScript frontend**:
* **Live Risk Screener:** Real-time single transaction scoring with dynamic threshold slider, risk gauge, and client-side animated SHAP waterfall contribution bars.
* **Quick Presets:** Instant testing with curated "Known Fraud", "Borderline Edge Case", and "Legitimate" transactions.
* **Cost Simulator:** Real-time business cost and tradeoff modeling slider.
* **Batch CSV Triage:** High-throughput batch streaming simulator highlighting flagged frauds and capital at risk.

### Running the Web Application:
```bash
# From inside credit-card-fraud-detection/
python app/app.py
```
Open **`http://127.0.0.1:8000`** in your browser.

---

## ⚡ Quickstart: Running the Entire Pipeline
```bash
# 1. Clone repository & install dependencies
pip install -r requirements.txt

# 2. Place creditcard.csv in data/ (already set up)

# 3. Execute full pipeline (data audit, CV benchmark, thresholding, SHAP, figures)
python run_pipeline.py

# 4. Launch web application
python app/app.py
```


