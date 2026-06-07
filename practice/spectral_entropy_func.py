import numpy as np
import matplotlib.pyplot as plt
def compute_spectral_entropy(signal):
    fft_vals=np.fft.rfft(signal)
    power=np.abs(fft_vals)**2
    power=power/np.sum(power)
    entropy=-np.sum(power*np.log2(power+1e-12))
    return entropy
X_raw=np.load("data/X_raw.npy")
y=np.load("data/y_snr.npy")
entropy=compute_spectral_entropy(X_raw[0])
print("Spectral Entropy: ",entropy)

#verify Entropy increases as SNR decreases
snr_values=[20,10,0,-10,-20]
for snr in snr_values:
    idx=np.where(y==snr)[0][0]
    signal=X_raw[idx]
    entropy=compute_spectral_entropy(signal)
    print( f"SNR = {snr:>3} dB",
        f"Entropy = {entropy:.4f}")
X_feat= np.load(
    "data/X_features.npy"
)
print(
    "Loaded Shape:",
    X_feat.shape
)
# Correlation Analysis
feature_names = [
    "Variance",
    "Kurtosis",
    "Skewness",
    "ZCR",
    "Spectral Entropy",
    "FFT Peak Ratio",
    "Mean Absolute Value"
]
correlations = []
print("\nFeature Correlations with SNR\n")
for i in range(
    X_feat.shape[1]
):
    corr = np.corrcoef(
        X_feat[:, i],
        y
    )[0, 1]
    correlations.append(corr)
    print(
        f"{feature_names[i]:20s}: {corr:.4f}"
    )

# Correlation Bar Plot
plt.figure(
    figsize=(10, 5)
)
plt.bar(
    feature_names,
    correlations
)
plt.title(
    "Feature Correlation with SNR"
)
plt.xlabel(
    "Features"
)

plt.ylabel(
    "Correlation Coefficient"
)
plt.xticks(
    rotation=45
)
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/bar chart of comparison correlation of different features and snr")
plt.show()