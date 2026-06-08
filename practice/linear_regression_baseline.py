import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)
X_feat=np.load("data/X_features.npy")
y=np.load("data/y_snr.npy")
X_train, X_test, y_train, y_test= train_test_split(X_feat,y,test_size=0.2,random_state=42)
print("training samples: ",len(X_train))
print("Testing samples: ",len(X_test))
#creating the scaler object
scaler=StandardScaler()
scaler.fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)
#creating model object
model=LinearRegression()
model.fit(X_train_scaled,y_train)
y_pred=model.predict(X_test_scaled)
#Evaluate model
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)
print("\n linear regression results: ")
print("MAE:", round(mae, 5))
print("RMSE:",round(rmse, 5))
print("R²:",round(r2, 5))
#regression plot(actual vs predicted model)
plt.figure(figsize=(8,8))
plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)#each point means(actual snr,predicted snr)
plt.plot(
    [-20,20],
    [-20,20],
    'r--',
    linewidth=2,
    label="Perfect Prediction"#ideal prediction is where actual data = predicted data
)
plt.xlabel("Actual SNR (dB)")
plt.ylabel("Predicted SNR (dB)")
plt.title(
    "Actual vs Predicted SNR"
)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/scatter plot of baseline linear regression model")
plt.show()

