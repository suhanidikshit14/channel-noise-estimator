import numpy as np
from scipy.stats import skew,kurtosis
import matplotlib.pyplot as plt
#load dataset
X_raw=np.load("data/X_raw.npy")
y=np.load("data/y_snr.npy")
#define neccesary functions
def zero_crossing_rate(signal):
    return np.mean(
        np.diff(np.sign(signal)) != 0
    )
def spectral_entropy_fun(signal):
    fft_values=np.fft.rfft(signal)
    power=np.abs(fft_values)**2
    power=power/np.sum(power)
    entropy=-np.sum(power*np.log2(power+1e-12))
    return entropy
def fft_peak_ratio(signal):
    fft_mag=np.abs(np.fft.rfft(signal))
    peak=np.max(fft_mag)
    mean_spectrum=np.mean(fft_mag)
    return peak/(mean_spectrum+1e-12)
#defining extract feature function
def extract_features(signal):

    variance = np.var(signal)
    kurt = kurtosis(
        signal,
        fisher=False
    )
    skewness = skew(signal)
    zcr = zero_crossing_rate(signal)
    spectral_entropy = spectral_entropy_fun(signal)

    peak_ratio = fft_peak_ratio(signal)
    mean_abs = np.mean(
        np.abs(signal)
    )
    return [
        variance,
        kurt,
        skewness,
        zcr,
        spectral_entropy,
        peak_ratio,
        mean_abs
    ]
#building feature matrix
X_feat=[]
for i,signal in enumerate(X_raw):
    features=extract_features(signal)
    X_feat.append(features)
    if(i+1)%1000==0:
      print(f"processed{i+1}/{len(X_raw)} signals")#tell after every 1000 signals are processed
X_feat=np.array(X_feat)
np.save("data/X_features.npy",X_feat)
print("complete")
print("X_feat shape: ",X_feat)
#to plot comparisno bar chart of correlation of snr and feature between differnet features
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
