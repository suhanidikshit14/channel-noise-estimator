import numpy as np
import matplotlib.pyplot as plt
X_raw=np.load("data/X_raw.npy")
y=np.load("data/y_snr.npy")
#comparison of high snr and low snr
high_idx=np.where(y==20)[0][0]
low_idx=np.where(y==-20)[0][0]
high_signal=X_raw[high_idx]
low_signal=X_raw[low_idx]
high_fft=np.abs(np.fft.rfft(high_signal))
low_fft=np.abs(np.fft.rfft(low_signal))
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(high_fft)
plt.title("FFT Magnitude(SNR=20dB)")
plt.xlabel("frequency bins")
plt.ylabel("Magnitude")
plt.subplot(1,2,2)
plt.plot(low_fft)
plt.title("FFT Magnitude(SNR=-20dB)")
plt.xlabel("frequency bins")
plt.ylabel("Magnitude")
plt.savefig("low and high snr fft comparison.png")
plt.show()