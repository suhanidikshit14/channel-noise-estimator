import numpy as np
import matplotlib.pyplot as plt
X_raw=np.load("data\X_raw.npy")
print("shape of X_raw: ",X_raw.shape)
y=np.load("data\y_snr.npy")
print("shape of y: ",y.shape)
 # to compute the fft magnitude of one sample
sample=X_raw[0]
fft_vals=np.fft.fft(sample)
fft_mag=np.abs(fft_vals)
plt.figure(figsize=(10,5))
plt.plot(fft_mag)
plt.title("fft magnitude spectrum")
plt.xlabel("frequency bin")
plt.ylabel("magnitude")
plt.grid(True)
plt.savefig("plots/plot of fft of sample signal.png")
plt.show()
