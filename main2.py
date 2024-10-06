import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler

# Data loading with caching
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

def sta_lta_fixed(signal, short_window, long_window):
    """Compute the STA/LTA ratio."""
    sta = np.convolve(signal**2, np.ones(short_window), mode='valid')
    lta = np.convolve(signal**2, np.ones(long_window), mode='valid')
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        sta_lta_ratio = np.where(lta != 0, sta / lta, 0)
    
    return sta_lta_ratio

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
        # For notch, we design a bandstop filter around the cutoff frequency
        bandwidth = 1.0  # 1 Hz bandwidth
        low = (cutoff - bandwidth / 2) / nyq
        high = (cutoff + bandwidth / 2) / nyq
        normal_cutoff = [low, high]
        b, a = butter(order, normal_cutoff, btype='bandstop', analog=False)
    return b, a

def apply_filter(data, filter_type, cutoff, fs, order=5, cutoff2=None):
    b, a = butter_filter(filter_type, cutoff, fs, order, cutoff2)
    y = filtfilt(b, a, data)
    return y

def main():
    # Title and Sidebar Setup
    st.title("🌋 Seismic Event Detection & Classification")
    st.sidebar.title("Settings")

    # File Upload
    uploaded_files = st.sidebar.file_uploader(
        "Upload Seismic Data Files", type=["csv"], accept_multiple_files=True
    )

    if uploaded_files:
        st.sidebar.success("Files uploaded successfully!")
        datasets = [load_data(f) for f in uploaded_files]
        file_options = [f.name for f in uploaded_files]

        # Multi-file selection for comparison
        selected_files = st.sidebar.multiselect(
            "Select Files for Comparison", file_options, default=file_options
        )

        if not selected_files:
            st.warning("Please select at least one file for analysis.")
            return

        # Model Selection
        model_option = st.sidebar.selectbox("Select Model", ("RandomForest", "Gradient Boosting", "SVM"))

        # Model Hyperparameters
        st.sidebar.subheader("Model Hyperparameters")
        if model_option == "RandomForest":
            n_estimators = st.sidebar.slider("Number of Trees", 50, 300, 100)
            max_depth = st.sidebar.slider("Max Depth", 3, 30, 10)
        elif model_option == "Gradient Boosting":
            n_estimators = st.sidebar.slider("Number of Boosting Stages", 50, 300, 100)
            learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1)
        elif model_option == "SVM":
            C = st.sidebar.slider("C (Regularization)", 0.1, 10.0, 1.0)
            kernel = st.sidebar.selectbox("Kernel", ("linear", "rbf"))

        # Loop over selected files for comparison
        for selected_file in selected_files:
            # Retrieve the corresponding dataframe
            data = next(df for i, df in enumerate(datasets) if file_options[i] == selected_file)

            st.header(f"**File: {selected_file}**")
            st.subheader("Available Columns:")
            st.write(data.columns.tolist())

            # Automatically detect time and velocity columns
            time_columns = [col for col in data.columns if 'time' in col.lower()]
            velocity_columns = [col for col in data.columns if 'velocity' in col.lower()]

            if not time_columns:
                st.error("No time column found. Please ensure your CSV has a time column.")
                continue

            if not velocity_columns:
                st.error("No velocity column found. Please ensure your CSV has a velocity column.")
                continue

            # Display detected columns
            st.subheader("Detected Columns:")
            st.write(f"Time Columns: {time_columns}")
            st.write(f"Velocity Columns: {velocity_columns}")

            # Allow user to select the correct columns if multiple are detected
            time_col = st.selectbox(f"Select Time Column for {selected_file}", time_columns, index=0)
            velocity_col = st.selectbox(f"Select Velocity Column for {selected_file}", velocity_columns, index=0)

            # Handle absolute and relative time
            if 'abs' in time_col.lower():
                try:
                    data[time_col] = pd.to_datetime(data[time_col], format='%Y-%m-%dT%H:%M:%S.%f')
                    data['rel_time(sec)'] = (data[time_col] - data[time_col].iloc[0]).dt.total_seconds()
                    time = data['rel_time(sec)']
                    st.success("Converted absolute time to relative time.")
                except Exception as e:
                    st.error(f"Error converting absolute time: {e}")
                    continue
            elif 'rel' in time_col.lower():
                time = data[time_col]
            else:
                st.error("Time column must contain 'abs' or 'rel' to indicate absolute or relative time.")
                continue

            signal = data[velocity_col]

            # Data Preview
            with st.expander("🔍 Data Preview"):
                st.write("First 5 rows of the seismic data:")
                st.write(data.head())

            # Normalization Option
            normalize_option = st.sidebar.checkbox("Normalize/Standardize Signal")
            if normalize_option:
                scaler = StandardScaler()
                signal = scaler.fit_transform(signal.values.reshape(-1, 1)).flatten()
                st.info("Signal has been standardized.")

            # Filter Settings
            filter_type = st.sidebar.selectbox("Select Filter Type", ("Highpass", "Lowpass", "Bandpass", "Notch"), index=0)
            cutoff_frequency = st.sidebar.slider("Filter Cutoff Frequency (Hz)", min_value=0.01, max_value=50.0, value=0.1)
            order = st.sidebar.slider("Filter Order", min_value=1, max_value=10, value=5)

            # Additional filter parameters for bandpass and notch
            cutoff_frequency_2 = None
            if filter_type in ["Bandpass", "Notch"]:
                cutoff_frequency_2 = st.sidebar.slider("Second Cutoff Frequency (Hz)", min_value=0.01, max_value=50.0, value=0.5)

            # Sampling Rate
            try:
                sampling_rate = 1.0 / (time.iloc[1] - time.iloc[0])
            except Exception as e:
                st.error(f"Error calculating sampling rate: {e}")
                continue

            # Apply filter
            if filter_type in ["Bandpass", "Notch"]:
                filtered_signal = apply_filter(signal, filter_type, cutoff_frequency, sampling_rate, order, cutoff_frequency_2)
            else:
                filtered_signal = apply_filter(signal, filter_type, cutoff_frequency, sampling_rate, order)

            # Plot Raw and Filtered Signals
            with st.expander("🔍 Seismic Signal (Raw vs. Filtered)"):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(time, signal, label='Raw Signal')
                ax.plot(time, filtered_signal, label='Filtered Signal', color='red')
                ax.set_xlabel('Relative Time (seconds)')
                ax.set_ylabel('Velocity (m/s)')
                ax.set_title('Seismic Signal (Raw vs. Filtered)')
                ax.legend()
                st.pyplot(fig)

            # STA/LTA Ratio Detection
            short_window = int(0.5 * sampling_rate)
            long_window = int(10 * sampling_rate)
            sta_lta_ratio = sta_lta_fixed(filtered_signal, short_window, long_window)

            # STA/LTA Threshold Slider
            threshold = st.sidebar.slider("STA/LTA Detection Threshold", min_value=1.0, max_value=10.0, value=3.0)

            # Find Seismic Events
            seismic_events = time[:len(sta_lta_ratio)][sta_lta_ratio > threshold]

            # STA/LTA Method Visualization
            with st.expander("📈 Seismic Event Detection (STA/LTA Method)"):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(time[:len(sta_lta_ratio)], sta_lta_ratio, label='STA/LTA Ratio')
                ax.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
                ax.scatter(seismic_events, np.repeat(threshold, len(seismic_events)), color='orange', label='Detected Events')
                ax.set_xlabel('Time (seconds)')
                ax.set_ylabel('STA/LTA Ratio')
                ax.set_title('STA/LTA Event Detection')
                ax.legend()
                st.pyplot(fig)

            # FFT Analysis
            n = len(filtered_signal)
            fft_result = fft(filtered_signal)
            freqs = fftfreq(n, 1/sampling_rate)

            # FFT Plot
            with st.expander("🔍 FFT Analysis"):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(freqs[:n//2], np.abs(fft_result[:n//2]), color='blue')
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Amplitude')
                ax.set_title('FFT of Filtered Signal')
                st.pyplot(fig)

            # Model Training
            st.subheader("Model Training")
            train_data = st.text_input("Input training data for classification (CSV format)")

            if train_data:
                # Handle classification training here (placeholder)
                st.success("Training completed! (placeholder)")

if __name__ == "__main__":
    main()
