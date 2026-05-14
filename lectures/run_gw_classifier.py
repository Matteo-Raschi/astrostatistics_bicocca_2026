import os
import urllib.request
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import time

URL = "https://github.com/dgerosa/pdetclassifier/releases/download/v0.2/sample_2e7_design_precessing_higherordermodes_3detectors.h5"
FILENAME = "sample_2e7_design_precessing_higherordermodes_3detectors.h5"

# 1. Download data if it doesn't exist
if not os.path.exists(FILENAME):
    print(f"Downloading {FILENAME} (This may take a while, it's >1GB)...")
    urllib.request.urlretrieve(URL, FILENAME)
    print("Download complete.")
else:
    print(f"File {FILENAME} already exists. Skipping download.")

# 2. Load data
print("Loading data...")
with h5py.File(FILENAME, "r") as f:
    # Look at the keys to understand the structure
    # According to the notebook, attributes include:
    # mtot, q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, ra, dec, iota, psi, z, snr
    
    # We will downsample for training to speed things up (e.g., take first 200,000 samples)
    N_SAMPLES = 200000
    
    # Features
    features = ['mtot', 'q', 'chi1x', 'chi1y', 'chi1z', 'chi2x', 'chi2y', 'chi2z', 'ra', 'dec', 'iota', 'psi', 'z']
    
    X = np.column_stack([f[feat][:N_SAMPLES] for feat in features])
    
    # Target based on SNR
    snr = f['snr'][:N_SAMPLES]
    y = (snr > 12).astype(int)
    
print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features.")
print(f"Number of detectable sources (y=1): {np.sum(y)}")
print(f"Number of non-detectable sources (y=0): {len(y) - np.sum(y)}")

# 3. Preprocessing
print("Splitting and scaling data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train classifiers
print("Training Naive Bayes...")
gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)
y_prob_gnb = gnb.predict_proba(X_test_scaled)[:, 1]

print("Training Random Forest Classifier...")
t0 = time.time()
rf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
rf.fit(X_train_scaled, y_train)
y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]
print(f"RF training took {time.time() - t0:.2f} seconds.")

# 5. Evaluate and Plot ROC curves
print("Computing ROC curves...")
fpr_gnb, tpr_gnb, _ = metrics.roc_curve(y_test, y_prob_gnb)
auc_gnb = metrics.auc(fpr_gnb, tpr_gnb)

fpr_rf, tpr_rf, _ = metrics.roc_curve(y_test, y_prob_rf)
auc_rf = metrics.auc(fpr_rf, tpr_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_gnb, tpr_gnb, label=f'Gaussian Naive Bayes (AUC = {auc_gnb:.3f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Contamination)')
plt.ylabel('True Positive Rate (Completeness)')
plt.title('ROC Curve for Gravitational Wave Detectability')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()

output_fig = "roc_curve_gw_detection.png"
plt.savefig(output_fig, dpi=300)
print(f"Saved ROC curve to {output_fig}")

# Feature importances for Random Forest
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.savefig("feature_importances_gw.png", dpi=300)
print("Saved feature importances to feature_importances_gw.png")

print("Analysis complete!")
