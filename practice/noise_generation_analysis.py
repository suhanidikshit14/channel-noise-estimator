import numpy as np
signal=np.sin(np.linspace(0,2*(np.pi),100))
noise_low=np.random.normal(0,0.1,100)
noise_high=np.random.normal(0,1,100)
noisy_signal_low=signal+noise_low #low noise signal
noisy_signal_high=signal+noise_high #high noise signal
print("mean of signal without noise: ",np.mean(signal))
print("mean of signal with low noise: ",np.mean(noisy_signal_low))
print("mean of signal with high noise: ",np.mean(noisy_signal_high))
print("variance of signal with low noise: ",np.var(noisy_signal_low))
print("variance of signal with high noise: ",np.var(noisy_signal_high))
print("standard deviation of signal with low noise: ",np.std(noisy_signal_low))
print("Standard deviation of signal with high noise: ",np.std(noisy_signal_high))
