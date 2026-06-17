import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import signal as sig
from scipy.stats import kurtosis, skew

st.set_page_config(page_title="ML Channel Noise Estimator", layout="wide")

@st.cache_resource
def load_artifacts():
    with open("models/rf_snr_estimator.pkl", "rb") as f:
        model = pickle.load(f)

    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler

model, scaler = load_artifacts()

def generate_signal(sig_type, length=512):
    t = np.linspace(0, 1, length)
    freq = np.random.uniform(2, 200)

    if sig_type == "sine":
        signal_data = np.sin(2 * np.pi * freq * t)

    elif sig_type == "square":
        signal_data = sig.square(2 * np.pi * freq * t)

    elif sig_type == "sawtooth":
        signal_data = sig.sawtooth(2 * np.pi * freq * t)

    elif sig_type == "am":
        carrier_freq = np.random.uniform(1000, 2000)
        message_freq = np.random.uniform(2, 200)

        message = np.sin(2 * np.pi * message_freq * t)
        carrier = np.cos(2 * np.pi * carrier_freq * t)

        signal_data = (1 + 0.5 * message) * carrier

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
        raise ValueError(f"Unknown signal type: {sig_type}")

    return signal_data

def add_awgn(signal, snr_db):
    sig_pow = np.mean(signal ** 2)
    noise_pow = sig_pow / (10 ** (snr_db / 10))
    noise_std = np.sqrt(noise_pow)

    noise = np.random.normal(0, noise_std, len(signal))

    return signal + noise

def zero_crossing_rate(signal):
    return np.mean(np.diff(np.sign(signal)) != 0)

def spectral_entropy_fun(signal):
    fft_values = np.fft.rfft(signal)
    power = np.abs(fft_values) ** 2
    power = power / np.sum(power)
    entropy = -np.sum(power * np.log2(power + 1e-12))
    return entropy

def fft_peak_ratio(signal):
    fft_mag = np.abs(np.fft.rfft(signal))
    peak = np.max(fft_mag)
    mean_spectrum = np.mean(fft_mag)
    return peak / (mean_spectrum + 1e-12)

def extract_features(signal):
    return np.array([
        np.var(signal),
        kurtosis(signal, fisher=False),
        skew(signal),
        zero_crossing_rate(signal),
        spectral_entropy_fun(signal),
        fft_peak_ratio(signal),
        np.mean(np.abs(signal))
    ])

st.title("📡 ML Channel Noise Estimator")
st.write("Random Forest based SNR prediction")

signal_type = st.sidebar.selectbox(
    "Signal Type",
    ["sine", "square", "sawtooth", "am", "chirp"]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Signal (.npy)",
    type=["npy"]
)

if uploaded_file is None:
    true_snr = st.sidebar.slider(
        "True SNR (dB)",
        -20,
        20,
        0
    )
else:
    true_snr = None

generate_button = st.sidebar.button("Generate")
batch_button = st.sidebar.button("Run 100 Random Estimates")

if generate_button:

    if uploaded_file is not None:

        try:
            noisy_signal = np.load(uploaded_file)
            noisy_signal = np.squeeze(noisy_signal)

            if len(noisy_signal) != 512:
                st.error("Signal must contain exactly 512 samples.")
                st.stop()

            clean_signal = None

        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.stop()

    else:

        clean_signal = generate_signal(signal_type)
        noisy_signal = add_awgn(clean_signal, true_snr)

    features = extract_features(noisy_signal).reshape(1, -1)

    features_scaled = scaler.transform(features)

    predicted_snr = model.predict(features_scaled)[0]

    if uploaded_file is None:

        error = abs(predicted_snr - true_snr)

        col1, col2, col3 = st.columns(3)

        col1.metric("True SNR", f"{true_snr:.2f} dB")
        col2.metric("Predicted SNR", f"{predicted_snr:.2f} dB")
        col3.metric("Error", f"{error:.2f} dB")

        if error < 2:
            st.success("🟢 Excellent Estimate")
        elif error < 5:
            st.warning("🟠 Moderate Error")
        else:
            st.error("🔴 Large Error")

    else:

        st.metric(
            "Predicted SNR",
            f"{predicted_snr:.2f} dB"
        )

    st.subheader("Time Domain Signals")

    if clean_signal is not None:

        fig, ax = plt.subplots(1, 2, figsize=(14, 4))

        ax[0].plot(clean_signal)
        ax[0].set_title("Clean Signal")
        ax[0].grid(True)

        ax[1].plot(noisy_signal)
        ax[1].set_title(f"Noisy Signal ({true_snr} dB)")
        ax[1].grid(True)

    else:

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(noisy_signal)
        ax.set_title("Uploaded Signal")
        ax.grid(True)

    st.pyplot(fig)

    st.subheader("FFT Spectrum")

    fft_mag = np.abs(np.fft.fft(noisy_signal))
    freqs = np.fft.fftfreq(len(noisy_signal), d=1/512)

    positive = freqs >= 0

    fig2, ax2 = plt.subplots(figsize=(10, 4))

    ax2.plot(freqs[positive], fft_mag[positive])

    ax2.set_title("FFT Magnitude Spectrum")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    ax2.grid(True)

    st.pyplot(fig2)

if batch_button:

    st.header("Batch Evaluation")

    errors = []

    signal_types = [
        "sine",
        "square",
        "sawtooth",
        "am",
        "chirp"
    ]

    for _ in range(100):

        sig_type = np.random.choice(signal_types)
        snr = np.random.randint(-20, 21)

        clean = generate_signal(sig_type)
        noisy = add_awgn(clean, snr)

        features = extract_features(noisy).reshape(1, -1)

        pred = model.predict(
            scaler.transform(features)
        )[0]

        errors.append(abs(pred - snr))

    col1, col2, col3 = st.columns(3)

    col1.metric("Mean Error", f"{np.mean(errors):.2f} dB")
    col2.metric("Median Error", f"{np.median(errors):.2f} dB")
    col3.metric("Max Error", f"{np.max(errors):.2f} dB")

    within_2db = np.mean(np.array(errors) < 2) * 100
    within_5db = np.mean(np.array(errors) < 5) * 100

    col4, col5 = st.columns(2)

    col4.metric("Within 2 dB", f"{within_2db:.1f}%")
    col5.metric("Within 5 dB", f"{within_5db:.1f}%")

    fig3, ax3 = plt.subplots(figsize=(8, 4))

    ax3.hist(errors, bins=20)

    ax3.set_title("Error Distribution")
    ax3.set_xlabel("Absolute Error (dB)")
    ax3.set_ylabel("Count")
    ax3.grid(True)

    st.pyplot(fig3)

