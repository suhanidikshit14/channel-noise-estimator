import numpy as np
import os
from scipy import signal as sig
def generate_signal(sig_type, length=512):

    t = np.linspace(0, 1, length)

    freq = np.random.uniform(2, 200)

    if sig_type == "sine":

        signal_data = np.sin(
            2 * np.pi * freq * t
        )

    elif sig_type == "square":

        signal_data = sig.square(
            2 * np.pi * freq * t
        )

    elif sig_type == "sawtooth":

        signal_data = sig.sawtooth(
            2 * np.pi * freq * t
        )

    elif sig_type == "am":

        carrier_freq = np.random.uniform(1000,2000)

        message_freq = np.random.uniform(2,200)

        message = np.sin(
            2 * np.pi * message_freq * t
        )

        carrier = np.cos(
            2 * np.pi * carrier_freq * t
        )

        signal_data = (
            1 + 0.5 * message
        ) * carrier

    elif sig_type == "chirp":

        start_freq = np.random.uniform(2, 10)

        end_freq = np.random.uniform(150, 200)

        signal_data = sig.chirp(
            t,
            f0=start_freq,
            f1=end_freq,
            t1=1,
            method="linear"
        )

    else:

        raise ValueError(
            f"Unknown signal type: {sig_type}"
        )

    return signal_data
def add_awgn(signal,snr_db):
    #add awgn for given snr_bd
    sig_pow=np.mean(signal**2)
    noise_pow=sig_pow/(10**(snr_db/10))
    noise_std=np.sqrt(noise_pow)
    noise=np.random.normal(0,noise_std,len(signal))
    noisy_signal=signal+noise
    return noisy_signal
signal_types = [
    "sine",
    "square",
    "sawtooth",
    "am",
    "chirp"
]
snr_levels = np.arange(-20, 22, 2)

X_raw = []
y = []

for sig_type in signal_types:

    print(f"Generating {sig_type} signals...")

    for snr in snr_levels:

        for i in range(200):

            clean_signal = generate_signal(
                sig_type
            )

            noisy_signal = add_awgn(
                clean_signal,
                snr
            )

            X_raw.append(noisy_signal)

            y.append(snr)
X_raw = np.array(X_raw)

y = np.array(y)


# -----------------------------------
# Save Dataset
# -----------------------------------

os.makedirs(
    "data",
    exist_ok=True
)

np.save(
    "data/X_raw.npy",
    X_raw
)

np.save(
    "data/y_snr.npy",
    y
)


# -----------------------------------
# Verification
# -----------------------------------

print("\nDataset Generation Complete!")

print(
    "Total Samples:",
    len(X_raw)
)

print(
    "X_raw Shape:",
    X_raw.shape
)

print(
    "y Shape:",
    y.shape
)
    
    
