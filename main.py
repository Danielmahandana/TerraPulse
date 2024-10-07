import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import json

# STA/LTA ratio calculation
def sta_lta_fixed(signal, short_window, long_window):
    """Compute the STA/LTA ratio with short and long windows."""
    sta = np.convolve(signal ** 2, np.ones(short_window), mode='same')
    lta = np.convolve(signal ** 2, np.ones(long_window), mode='same')
    with np.errstate(divide='ignore', invalid='ignore'):
        sta_lta_ratio = np.where(lta != 0, sta / lta, 0)
    return sta_lta_ratio

# Butterworth filter design
def butter_filter(filter_type, cutoff, fs, order=5, cutoff2=None):
    nyq = 0.5 * fs
    if filter_type == "Highpass":
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
    elif filter_type == "Lowpass":
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
    elif filter_type == "Bandpass":
        normal_cutoff = [cutoff / nyq, cutoff2 / nyq]
        b, a = butter(order, normal_cutoff, btype='band', analog=False)
    elif filter_type == "Notch":
        bandwidth = 1.0  # 1 Hz bandwidth
        low = (cutoff - bandwidth / 2) / nyq
        high = (cutoff + bandwidth / 2) / nyq
        normal_cutoff = [low, high]
        b, a = butter(order, normal_cutoff, btype='bandstop', analog=False)
    return b, a

# Apply filter
def apply_filter(data, filter_type, cutoff, fs, order=5, cutoff2=None):
    b, a = butter_filter(filter_type, cutoff, fs, order, cutoff2)
    y = filtfilt(b, a, data)
    return y

# Feature extraction
def extract_features(signal):
    fft_values = np.fft.fft(signal).real[:len(signal) // 2]
    features = {
        'mean': np.mean(signal),
        'std_dev': np.std(signal),
        'max_amplitude': np.max(signal),
    }

    for i, fft_value in enumerate(fft_values):
        features[f'fft_{i}'] = fft_value

    return pd.DataFrame([features])

# CNN Model
def build_cnn(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification (event/no event)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    st.title("🌋 Seismic Event Detection & Classification")
    st.sidebar.title("Settings")

    uploaded_files = st.sidebar.file_uploader("Upload Seismic Data Files", type=["csv"], accept_multiple_files=True)

    if uploaded_files:
        st.sidebar.success("Files uploaded successfully!")
        datasets = [pd.read_csv(f) for f in uploaded_files]
        file_options = [f.name for f in uploaded_files]

        selected_files = st.sidebar.multiselect("Select Files for Analysis", file_options, default=file_options)
        if not selected_files:
            st.warning("Please select at least one file for analysis.")
            return

        for selected_file in selected_files:
            data = next(df for i, df in enumerate(datasets) if file_options[i] == selected_file)

            st.header(f"**File: {selected_file}**")
            st.subheader("Available Columns:")
            st.write(data.columns.tolist())

            time_columns = [col for col in data.columns if 'time' in col.lower()]
            velocity_columns = [col for col in data.columns if 'velocity' in col.lower()]

            if not time_columns:
                st.error("No time column found.")
                continue

            if not velocity_columns:
                st.error("No velocity column found.")
                continue

            time_col = st.selectbox(f"Select Time Column for {selected_file}", time_columns, index=0)
            velocity_col = st.selectbox(f"Select Velocity Column for {selected_file}", velocity_columns, index=0)

            time = data[time_col]
            signal = data[velocity_col]

            if st.sidebar.checkbox("Normalize/Standardize Signal"):
                scaler = StandardScaler()
                signal = scaler.fit_transform(signal.values.reshape(-1, 1)).flatten()
                st.info("Signal has been standardized.")

            filter_type = st.sidebar.selectbox("Select Filter Type", ("Highpass", "Lowpass", "Bandpass", "Notch"))
            cutoff_frequency = st.sidebar.slider("Cutoff Frequency (Hz)", min_value=0.01, max_value=50.0, value=0.1)
            order = st.sidebar.slider("Filter Order", min_value=1, max_value=10, value=5)

            if filter_type in ["Bandpass", "Notch"]:
                cutoff_frequency_2 = st.sidebar.slider("Second Cutoff Frequency (Hz)", min_value=0.01, max_value=50.0, value=0.5)

            sampling_rate = 1.0 / (time.iloc[1] - time.iloc[0])
            if filter_type in ["Bandpass", "Notch"]:
                filtered_signal = apply_filter(signal, filter_type, cutoff_frequency, sampling_rate, order, cutoff_frequency_2)
            else:
                filtered_signal = apply_filter(signal, filter_type, cutoff_frequency, sampling_rate, order)

            with st.expander("Seismic Signal (Raw vs. Filtered)"):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(time, signal, label="Raw Signal")
                ax.plot(time, filtered_signal, label="Filtered Signal", color="red")
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Velocity (m/s)")
                ax.legend()
                st.pyplot(fig)

            short_window = int(0.5 * sampling_rate)
            long_window = int(10 * sampling_rate)
            sta_lta_ratio = sta_lta_fixed(filtered_signal, short_window, long_window)

            threshold = st.sidebar.slider("STA/LTA Detection Threshold", 1.0, 10.0, 3.0)
            seismic_events = time[sta_lta_ratio > threshold]

            with st.expander("STA/LTA Event Detection"):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(time, sta_lta_ratio, label="STA/LTA Ratio", color="orange")
                ax.axhline(y=threshold, color="red", linestyle="--", label="Threshold")
                ax.scatter(seismic_events, [threshold] * len(seismic_events), color="green", label="Detected Events")
                ax.legend()
                st.pyplot(fig)

            st.write("Detected Seismic Event Times (seconds):")
            st.write(seismic_events)

            st.write("Feature Extraction:")
            features = extract_features(filtered_signal)
            st.write(features)

            X = np.array(filtered_signal).reshape(-1, 1)
            y = (sta_lta_ratio > threshold).astype(int)

            X_train, X_test = X[:int(0.8*len(X))], X[int(0.8*len(X)):]
            y_train, y_test = y[:int(0.8*len(y))], y[int(0.8*len(y)):]

            model = build_cnn((X_train.shape[1], 1))

            history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

            y_pred = (model.predict(X_test) > 0.5).astype(int)
            st.write(classification_report(y_test, y_pred))

            st.write("Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

if __name__ == "__main__":
    main()
