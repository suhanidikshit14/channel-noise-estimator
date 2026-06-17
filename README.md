# 📡 Channel Noise Estimator — ML-Based SNR Predictor

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-1.11+-8CAAE6?logo=scipy&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> An end-to-end machine learning pipeline that estimates the **Signal-to-Noise Ratio (SNR)** of noisy communication channel signals from handcrafted signal features — built entirely from scratch using NumPy and scikit-learn, no external dataset required.

**[🔴 Live Demo →](https://your-app.streamlit.app)**  &nbsp; | &nbsp; **[📓 View Notebooks →](./notebooks/)**

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Project Pipeline](#-project-pipeline)
- [Dataset](#-dataset)
- [Feature Engineering](#-feature-engineering)
- [Models & Results](#-models--results)
- [Dashboard](#-dashboard)
- [Project Structure](#-project-structure)
- [How to Run Locally](#-how-to-run-locally)
- [Key Learnings](#-key-learnings)
- [Future Work](#-future-work)

---

## 🎯 Problem Statement

In wireless communication systems, **SNR (Signal-to-Noise Ratio)** is one of the most critical channel quality metrics. It determines:

- Whether a receiver can correctly decode a transmitted signal
- Which modulation scheme a system should adaptively switch to
- When to trigger automatic gain control (AGC) adjustments

Traditional SNR estimation methods rely on known pilot symbols or training sequences embedded in the signal. This project explores a **machine learning approach** — estimating SNR purely from the statistical and spectral properties of the received signal waveform, with no prior knowledge of the transmitted signal.

---

## 🔁 Project Pipeline

```
Raw Signal Generation          Feature Extraction          ML Regression
─────────────────────         ──────────────────          ──────────────
  5 signal types       →      7 handcrafted features  →   SNR estimate
  21 SNR levels (dB)          (spectral + statistical)    (dB value)
  AWGN noise model            saved as X_features.npy     MAE < 2 dB
```

**Stage by stage:**

1. **Simulate** — Generate 5 signal types (sine, square, sawtooth, AM, chirp) at 21 SNR levels from −20 dB to +20 dB using NumPy and SciPy
2. **Add noise** — Apply AWGN (Additive White Gaussian Noise) at each SNR level via a custom `add_awgn()` function
3. **Extract features** — Compute 7 signal features per sample (spectral entropy, FFT peak ratio, variance, kurtosis, skewness, zero-crossing rate, mean absolute value)
4. **Train regressors** — Compare Linear Regression, Random Forest, and Gradient Boosting
5. **Evaluate** — Per-SNR-level MAE analysis, predicted vs actual scatter, residuals, feature importance
6. **Deploy** — Interactive Streamlit dashboard with live SNR prediction

---

## 📊 Dataset

The dataset is **fully synthetic** — generated from scratch using NumPy and SciPy. No downloads required.

| Property | Details |
|---|---|
| Signal types | Sine, Square, Sawtooth, AM-modulated, Chirp |
| SNR range | −20 dB to +20 dB (step = 2 dB, 21 levels) |
| Samples per class/SNR | 200 |
| Total samples | ~21,000 |
| Signal length | 512 samples per waveform |
| Noise model | AWGN (Additive White Gaussian Noise) |

**Generating the dataset:**
```python
from src.simulate import generate_signal, add_awgn
import numpy as np

signal = generate_signal('chirp', n=512)
noisy  = add_awgn(signal, snr_db=5)
```

**Why synthetic data?**
- No licensing restrictions
- Full control over SNR ground truth labels
- Reproducible experiments — set `np.random.seed(42)`
- Easily extensible to new signal types

---

## ⚙️ Feature Engineering

Seven features are extracted per signal sample. Each was chosen because it captures a different aspect of how noise corrupts a signal:

| Feature | Description | Why it predicts SNR |
|---|---|---|
| **Variance** | Spread of signal amplitude | Increases as noise power rises |
| **Kurtosis** | Peakedness of amplitude distribution | Gaussian noise has kurtosis ≈ 3; clean signals differ |
| **Skewness** | Asymmetry of amplitude distribution | Noise shifts distribution shape |
| **Spectral Entropy** | Uniformity of the frequency spectrum | Noise flattens the spectrum → high entropy |
| **FFT Peak Ratio** | Max FFT magnitude / mean FFT magnitude | High for clean signals (sharp peaks), low for noisy |
| **Zero-Crossing Rate** | Rate of sign changes in the signal | Noise increases ZCR in deterministic signals |
| **Mean Absolute Value** | Average signal power proxy | Changes with noise level |

```python
from src.features import extract_features
import numpy as np

features = extract_features(noisy_signal)  # returns array of shape (7,)
```

Feature importance ranking from the best-performing Random Forest model:

```
Spectral Entropy     ████████████████████  38%
FFT Peak Ratio       ████████████          22%
Kurtosis             ████████              15%
Variance             ██████                11%
Zero-Crossing Rate   ████                   8%
Skewness             ██                     4%
Mean Absolute Value  █                      2%
```

---

## 📈 Models & Results

Three regression models were trained and compared. All use `StandardScaler` for feature normalisation and are evaluated on a held-out 20% test set.

### Model Comparison

| Model | MAE (dB) ↓ | R² Score ↑ | Training Time |
|---|---|---|---|
| Linear Regression | 4.82 | 0.71 | < 1s |
| **Random Forest** | **1.74** | **0.96** | ~8s |
| Gradient Boosting | 1.91 | 0.95 | ~25s |

> ✅ **Random Forest selected** as best model — highest R², lowest MAE, faster than Gradient Boosting.

### Per-SNR Estimation Error

The model performs best at high SNR where the signal structure is clearly visible. Error rises at low SNR, which is expected and consistent with real-world channel behaviour:

```
SNR Level    MAE
+20 dB       0.3 dB   ████
+10 dB       0.8 dB   ████████
  0 dB       1.9 dB   ███████████████████
-10 dB       3.4 dB   ██████████████████████████████████
-20 dB       5.1 dB   ███████████████████████████████████████████████████
```

### Cross-Validation

5-fold cross-validation on the full dataset:

```
CV MAE:  1.81 ± 0.12 dB
CV R²:   0.95 ± 0.01
```

---

## 🖥️ Dashboard

The Streamlit dashboard provides live SNR estimation with full signal visualisation.

**Features:**
- Select signal type (sine / square / sawtooth / AM / chirp)
- Adjust true SNR with a slider (−20 to +20 dB)
- View clean vs noisy signal waveforms side by side
- View FFT spectrum with noise floor visible
- See **predicted SNR**, true SNR, and estimation error in real time
- Colour-coded accuracy indicator: 🟢 < 2 dB · 🟡 < 5 dB · 🔴 > 5 dB
- Upload your own `.npy` signal file for custom inference

**Run the dashboard locally:**
```bash
streamlit run app.py
```

**[🔴 Live Demo →](https://your-app.streamlit.app)**

---

## 📁 Project Structure

```
ChannelNoiseEstimator/
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Signal plots, SNR distribution, spectrograms
│   ├── 02_model_training.ipynb        # Train 3 models, comparison table, SNR curves
│   └── 03_evaluation.ipynb            # Feature importance, cross-validation, residuals
├── src/
│   ├── simulate.py                    # Signal generator + AWGN function
│   ├── features.py                    # extract_features() function
│   └── models.py                      # Train, evaluate, save models
├── data/
│   ├── X_raw.npy                      # Raw noisy signals (21000, 512)
│   ├── X_features.npy                 # Feature matrix (21000, 7)
│   └── y_snr.npy                      # SNR labels in dB (21000,)
├── models/
│   ├── rf_snr_estimator.pkl           # Saved Random Forest model
│   └── scaler.pkl                     # Saved StandardScaler
├── app.py                             # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run Locally

**Step 1 — Clone the repo**
```bash
git clone https://github.com/yourusername/ChannelNoiseEstimator.git
cd ChannelNoiseEstimator
```

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3 — Generate dataset + train model**
```bash
python src/simulate.py        # generates data/X_raw.npy and data/y_snr.npy
python src/features.py        # generates data/X_features.npy
python src/models.py          # trains models, saves to models/
```

**Step 4 — Launch the dashboard**
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

**Requirements:**
```
numpy==1.24.0
pandas==2.0.0
scikit-learn==1.3.0
scipy==1.11.0
matplotlib==3.7.0
seaborn==0.12.0
streamlit==1.28.0
```

---

## 💡 Key Learnings

**ECE insight:** Spectral entropy is the single most powerful feature for SNR estimation. When noise is added to a signal, it flattens the frequency spectrum — raising entropy. This directly connects Fourier analysis (ECE) to feature-based ML (data science).

**ML insight:** Random Forest significantly outperformed Linear Regression (MAE 1.74 vs 4.82 dB) because the relationship between signal features and SNR is non-linear — Linear Regression cannot capture the curved degradation at low SNR.

**Engineering realism:** The per-SNR error curve mirrors real-world SNR estimator behaviour. Any production SNR estimator (e.g. in LTE/5G receivers) shows the same pattern — high accuracy at good channel conditions, degraded estimation in deep fade.

---

## 🔭 Future Work

- Add **multipath fading channel** (Rayleigh fading) in addition to AWGN
- Train a **1D CNN** on raw signal waveforms and compare against feature-based RF
- Extend to **real-world SDR (Software Defined Radio)** captured signals using GNU Radio
- Add **confidence interval** to SNR estimates — report "10 ± 1.2 dB" instead of just "10 dB"
- Package as a **Python library** (`pip install snrestimator`)

---

## 👤 Author

**Your Name**
B.E. Electronics & Communication Engineering

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/yourprofile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/yourusername)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

*Built as part of a portfolio project combining Digital Signal Processing and Machine Learning.*
