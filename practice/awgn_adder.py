import numpy as np
import matplotlib.pyplot as plt
def add_awgn(signal,snr_db):
    #add awgn for given snr_bd
    sig_pow=np.mean(signal**2)
    noise_pow=sig_pow/(10**(snr_db/10))
    noise_std=np.sqrt(noise_pow)
    noise=np.random.normal(0,noise_std,len(signal))
    noisy_signal=signal+noise
    return noisy_signal

t=np.linspace(0,1,1000)
signal=np.sin(2*np.pi*5*t)#frequency 5hz
snr_values= [20,10,0,-10]
plt.figure(figsize=(14,8))
for i, snr in enumerate(snr_values):
    noisy_signal = add_awgn(signal, snr)

    plt.subplot(2, 2, i + 1)
    plt.plot(t, noisy_signal)
    plt.title(f"SNR = {snr} dB")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)

plt.tight_layout()
plt.savefig("plot of snr comparisons")
plt.show()