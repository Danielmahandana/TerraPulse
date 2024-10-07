import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import json

# Load data with caching
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

# STA/LTA algorithm
def sta_lta_fixed(signal, short_window, long_window):
    sta = np.convolve(signal ** 2, np.ones(short_window), mode='same')
    lta = np.convolve(signal ** 2, np.ones(long_window), mode='same')
    with np.errstate(divide='ignore', invalid='ignore'):
        sta_lta_ratio = np.where(lta != 0, sta / lta, 0)
    return sta_lta_ratio

# Butterworth filter
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
        bandwidth = 1.0
        low = (cutoff - bandwidth / 2) / nyq
        high = (cutoff + bandwidth / 2) / nyq
        normal_cutoff = [low, high]
        b, a = butter(order, normal_cutoff, btype='bandstop', analog=False)
    return b, a

def apply_filter(data, filter_type, cutoff, fs, order=5, cutoff2=None):
    b, a = butter_filter(filter_type, cutoff, fs, order, cutoff2)
    y = filtfilt(b, a, data)
    return y

# Feature extraction
def extract_features(signal):
    features = {
        'mean': np.mean(signal),
        'std_dev': np.std(signal),
        'max_amplitude': np.max(signal),
        'fft': np.fft.fft(signal).real[:len(signal)//2]  # Only the real part
    }
    return pd.DataFrame(features, index=[0])

def main():
    st.title("🌋 Seismic Event Detection & Classification (NASA Ready)")
    st.sidebar.title("Settings")

    uploaded_files = st.sidebar.file_uploader("Upload Seismic Data Files", type=["csv"], accept_multiple_files=True)

    if uploaded_files:
        st.sidebar.success("Files uploaded successfully!")
        datasets = [load_data(f) for f in uploaded_files]
        file_options = [f.name for f in uploaded_files]

        selected_files = st.sidebar.multiselect("Select Files for Analysis", file_options, default=file_options)
        if not selected_files:
            st.warning("Please select at least one file for analysis.")
            return

        # Select model
        model_option = st.sidebar.selectbox("Select Model", ("RandomForest", "Gradient Boosting", "SVM", "Ensemble"))

        # Model hyperparameters
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

        for selected_file in selected_files:
            data = next(df for i, df in enumerate(datasets) if file_options[i] == selected_file)

            st.header(f"**File: {selected_file}**")
            st.subheader("Available Columns:")
            st.write(data.columns.tolist())

            # Auto-detect time and velocity columns
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

            # Filter setup
            filter_type = st.sidebar.selectbox("Select Filter Type", ("Highpass", "Lowpass", "Bandpass", "Notch"))
            cutoff_frequency = st.sidebar.slider("Cutoff Frequency (Hz)", min_value=0.01, max_value=50.0, value=0.1)
            order = st.sidebar.slider("Filter Order", min_value=1, max_value=10, value=5)
            cutoff_frequency_2 = None
            if filter_type in ["Bandpass", "Notch"]:
                cutoff_frequency_2 = st.sidebar.slider("Second Cutoff Frequency (Hz)", min_value=0.01, max_value=50.0, value=0.5)

            # Apply filter
            sampling_rate = 1.0 / (time.iloc[1] - time.iloc[0])
            filtered_signal = apply_filter(signal, filter_type, cutoff_frequency, sampling_rate, order, cutoff_frequency_2)

            # Plot raw and filtered signals
            with st.expander("Seismic Signal (Raw vs. Filtered)"):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(time, signal, label="Raw Signal")
                ax.plot(time, filtered_signal, label="Filtered Signal", color="red")
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Velocity (m/s)")
                ax.legend()
                st.pyplot(fig)

            # STA/LTA Event Detection
            short_window = int(0.5 * sampling_rate)
            long_window = int(10 * sampling_rate)
            sta_lta_ratio = sta_lta_fixed(filtered_signal, short_window, long_window)
            threshold = st.sidebar.slider("STA/LTA Detection Threshold", 1.0, 10.0, 3.0)
            seismic_events = time[sta_lta_ratio > threshold]

            # Plot STA/LTA detection
            with st.expander("STA/LTA Event Detection"):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(time, sta_lta_ratio, label="STA/LTA Ratio", color="orange")
                ax.axhline(y=threshold, color="red", linestyle="--", label="Threshold")
                ax.scatter(seismic_events, [threshold] * len(seismic_events), color="green", label="Detected Events")
                ax.legend()
                st.pyplot(fig)

            st.write("Detected Seismic Event Times (seconds):")
            st.write(seismic_events)

            # Extract features
            st.write("Feature Extraction:")
            features = extract_features(filtered_signal)
            st.write(features)

            # Prepare data for ML model
            X = features.values
            y = np.random.randint(0, 2, size=X.shape[0])  # Dummy labels for demonstration

            # Model selection and training
            if model_option == "RandomForest":
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
            elif model_option == "Gradient Boosting":
                model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
            elif model_option == "SVM":
                model = SVC(C=C, kernel=kernel)
            elif model_option == "Ensemble":
                model = VotingClassifier(estimators=[
                    ('rf', RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)),
                    ('gb', GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)),
                    ('svm', SVC(C=C, kernel=kernel))
                ])

            # Cross-validation and model evaluation
            kf = KFold(n_splits=5)
            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                st.write(f"Confusion Matrix for {selected_file}:")
                st.write(confusion_matrix(y_test, predictions))
                st.write(f"Classification Report for {selected_file}:")
                st.write(classification_report(y_test, predictions))

            # GeoJSON Export
            geojson_export = st.checkbox("Export to GeoJSON")
            if geojson_export:
                geojson_data = {
                    "type": "FeatureCollection",
                    "features": []
                }
                for event_time in seismic_events:
                    geojson_data["features"].append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [event_time, 0]
                        },
                        "properties": {
                            "time": event_time
                        }
                    })
                geojson_str = json.dumps(geojson_data, indent=4)
                st.download_button("Download GeoJSON", geojson_str, file_name=f"{selected_file}_events.geojson")

if __name__ == "__main__":
    main()
