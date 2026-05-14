import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

file = pd.read_csv("C:/Users/user/OneDrive/Documenti/Astro/astrostatistics_bicocca_2026/solutions/galaxyquasar.csv")

# compute colors
file['u-g'] = file['u'] - file['g']
file['g-r'] = file['g'] - file['r']
file['r-i'] = file['r'] - file['i']
file['i-z'] = file['i'] - file['z']

colors = ['u-g', 'g-r', 'r-i', 'i-z']

print("Correlation matrix for colors:")
print(file[colors].corr())

# Correlation matrix for original bands
print("\nCorrelation matrix for bands:")
print(file[['u', 'g', 'r', 'i', 'z']].corr())

# Target
y = (file['class'] == 'QSO').astype(int)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(file[colors], y, test_size=0.3, random_state=42)

print("\nModel Performance (GaussianNB):")
# subsets and all together
import itertools

best_auc = 0
best_subset = None

for i in range(1, 5):
    for subset in itertools.combinations(colors, i):
        clf = GaussianNB()
        clf.fit(X_train[list(subset)], y_train)
        y_pred = clf.predict_proba(X_test[list(subset)])[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        print(f"Subset {subset}: AUC = {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_subset = subset

print(f"\nBest Subset: {best_subset} with AUC = {best_auc:.4f}")

# Feature importance using Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
importances = rf.feature_importances_
print("\nFeature importances (Random Forest):")
for color, imp in zip(colors, importances):
    print(f"{color}: {imp:.4f}")
