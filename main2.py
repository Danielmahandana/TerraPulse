import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.signal import butter, filtfilt, spectrogram
from scipy.fft import fft, fftfreq
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import io

# Streamlit Caching for expensive computations
@st.cache
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

st.title("🌋 Seismic Event Detection & Classification")
st.sidebar.title("Settings")

# File upload and dynamic file handling
uploaded_files = st.sidebar.file_uploader("Upload Seismic Data Files", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    st.sidebar.success("Files uploaded successfully!")
    datasets = [load_data(f) for f in uploaded_files]

    # Option to select file for comparison
    file_options = [f.name for f in uploaded_files]
    selected_files = st.sidebar.multiselect("Select Files for Comparison", file_options)

    # Option to select a model
    model_option = st.sidebar.selectbox("Select Model", ("RandomForest", "Gradient Boosting", "SVM"))

    # Model hyperparameters
    st.sidebar.write("Hyperparameters")
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
        data = next(df for i, df in enumerate(datasets) if file_options[i] == selected_file)
        
        # Data preview and preprocessing
        st.write(f"**File: {selected_file}**")
        st.write(data.head())

        # Extract time and signal data
        time = data['time_rel(sec)']
        signal = data['velocity(m/s)']

        # Data normalization/standardization
        normalize_option = st.sidebar.checkbox("Normalize/Standardize Signal")
        if normalize_option:
            signal = (signal - signal.mean()) / signal.std()

        # Highpass filter settings
        cutoff_frequency = st.sidebar.slider("Highpass Filter Cutoff Frequency", min_value=0.01, max_value=1.0, value=0.1)
        order = st.sidebar.slider("Filter Order", min_value=1, max_value=10, value=5)
        sampling_rate = 1.0 / (time[1] - time[0])

        def butter_highpass(cutoff, fs, order=5):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='high', analog=False)
            return b, a

        def highpass_filter(data, cutoff, fs, order=5):
            b, a = butter_highpass(cutoff, fs, order=order)
            y = filtfilt(b, a, data)
            return y

        filtered_signal = highpass_filter(signal, cutoff_frequency, sampling_rate, order)

        # STA/LTA detection method
        def sta_lta_fixed(signal, short_window, long_window):
            sta = np.cumsum(signal ** 2)
            sta = (sta[short_window:] - sta[:-short_window]) / short_window
            lta = (sta[long_window:] - sta[:-long_window]) / long_window
            sta = sta[:len(lta)]
            return sta / (lta + 1e-9)

        short_window = int(0.5 * sampling_rate)
        long_window = int(10 * sampling_rate)
        sta_lta_ratio = sta_lta_fixed(filtered_signal, short_window, long_window)

        # Detect seismic events
        threshold = st.sidebar.slider("STA/LTA Detection Threshold", min_value=1.0, max_value=10.0, value=3.0)
        seismic_events = time[:len(sta_lta_ratio)][sta_lta_ratio > threshold]

        # FFT analysis
        N = len(filtered_signal)
        T = 1 / sampling_rate
        yf = fft(filtered_signal)
        xf = fftfreq(N, T)[:N // 2]
        peak_frequency = xf[np.argmax(np.abs(yf[:N // 2]))]

        # Event energy calculation
        event_energy = np.sum(filtered_signal ** 2)

        # Visualization of the signal
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time, filtered_signal, label='Filtered Signal', color='red')
        ax.set_title(f'Seismic Signal (Filtered) - {selected_file}')
        st.pyplot(fig)

        # Feature extraction for classification
        event_properties = []
        for event_time in seismic_events:
            idx = np.where(np.isclose(time, event_time, atol=1e-3))[0]
            if len(idx) > 0:
                amplitude = np.max(filtered_signal[idx[0] - short_window:idx[0] + short_window])
                duration = len(sta_lta_ratio[idx[0] - short_window:idx[0] + short_window]) / sampling_rate
                event_properties.append((event_time, amplitude, duration, peak_frequency, event_energy))

        event_properties_df = pd.DataFrame(event_properties, columns=['Time (s)', 'Amplitude', 'Duration (s)', 'Peak Frequency (Hz)', 'Event Energy'])

        # Model selection and fitting
        if len(event_properties_df) > 0:
            X = event_properties_df[['Amplitude', 'Duration (s)', 'Peak Frequency (Hz)', 'Event Energy']]
            y = ["seismic_event"] * len(X)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Model selection
            if model_option == "RandomForest":
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            elif model_option == "Gradient Boosting":
                model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
            elif model_option == "SVM":
                model = SVC(C=C, kernel=kernel, random_state=42)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.write("### Classification Report")
            st.text(classification_report(y_test, y_pred))

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            st.pyplot(fig_cm)

        # Option to download event data as CSV
        st.download_button(
            label="Download Event Data as CSV",
            data=event_properties_df.to_csv(index=False),
            file_name=f'seismic_event_properties_{selected_file}.csv',
            mime='text/csv',
        )
else:
    st.warning("Please upload at least one CSV file with seismic data.")
