import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ---------------------------
# Load Dataset
# ---------------------------

X = np.load(r"C:\Users\suhan\OneDrive\Desktop\channel noise estimator\data\X_features.npy")
y = np.load(r"C:\Users\suhan\OneDrive\Desktop\channel noise estimator\data\y_snr.npy")

print("Original Shape:", X.shape)

# ---------------------------
# Remove Kurtosis
# Column 1 = Kurtosis
# ---------------------------

X_no_kurtosis = np.delete(
    X,
    1,
    axis=1
)

print(
    "Shape without Kurtosis:",
    X_no_kurtosis.shape
)

# ---------------------------
# Train-Test Split
# ---------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_no_kurtosis,
    y,
    test_size=0.20,
    random_state=42
)

# ---------------------------
# Feature Scaling
# ---------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

# ---------------------------
# Train Random Forest
# ---------------------------

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(
    X_train_scaled,
    y_train
)

# ---------------------------
# Predictions
# ---------------------------

pred = rf.predict(
    X_test_scaled
)

# ---------------------------
# Metrics
# ---------------------------

mae = mean_absolute_error(
    y_test,
    pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        pred
    )
)

r2 = r2_score(
    y_test,
    pred
)

print("\nResults WITHOUT Kurtosis")
print("-" * 40)

print(
    f"MAE  : {mae:.4f}"
)

print(
    f"RMSE : {rmse:.4f}"
)

print(
    f"R²   : {r2:.4f}"
)