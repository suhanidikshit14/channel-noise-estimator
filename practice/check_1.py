import numpy as np

feature_names = [
    "Variance",
    "Kurtosis",
    "Skewness",
    "Zero Crossing Rate",
    "Spectral Entropy",
    "FFT Peak Ratio",
    "Mean Absolute Value"
]

X = np.load(r"C:\Users\suhan\OneDrive\Desktop\channel noise estimator\data\X_features.npy")

print("Feature Statistics\n")

for i, name in enumerate(feature_names):

    feature = X[:, i]

    print(f"\n{name}")
    print("-" * 40)

    print("Min   :", np.min(feature))
    print("Max   :", np.max(feature))
    print("Mean  :", np.mean(feature))
    print("Std   :", np.std(feature))