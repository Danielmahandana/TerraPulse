import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.signal import butter, filtfilt, spectrogram
from scipy.fft import fft, fftfreq
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

st.title("🌋 Seismic Event Detection & Classification")

# Sidebar for inputs
st.sidebar.title("Settings")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload Seismic Data File", type=["csv"])

if uploaded_file is not None:
    st.sidebar.success("File uploaded successfully!")

    # Read the uploaded CSV data
    try:
        seismic_data = pd.read_csv(uploaded_file)
        st.sidebar.success("Data loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {str(e)}")
        st.stop()

    # Data preview
    with st.expander("🔍 Data Preview"):
        st.write("First 5 rows of the seismic data:")
        st.write(seismic_data.head())

    # Ensure required columns are present
    if 'rel_time(sec)' not in seismic_data.columns or 'velocity(c/s)' not in seismic_data.columns:
        st.error("The uploaded file must contain 'rel_time(sec)' and 'velocity(c/s)' columns.")
        st.stop()

    # Extract time and signal data
    time = seismic_data['rel_time(sec)']
    signal = seismic_data['velocity(c/s)']

    # Allow user to adjust highpass filter settings
    cutoff_frequency = st.sidebar.slider("Highpass Filter Cutoff Frequency (Hz)", min_value=0.01, max_value=1.0, value=0.1)
    order = st.sidebar.slider("Filter Order", min_value=1, max_value=10, value=5)

    # Highpass filter function
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def highpass_filter(data, cutoff, fs, order=5):
        b, a = butter_highpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    # Calculate sampling rate from time data
    sampling_rate = 1.0 / (time[1] - time[0])

    # Apply highpass filter to the signal
    filtered_signal = highpass_filter(signal, cutoff_frequency, sampling_rate, order)

    # Plot raw and filtered signals
    with st.expander("🔍 Seismic Signal (Raw vs. Filtered)"):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time, signal, label='Raw Signal')
        ax.plot(time, filtered_signal, label='Filtered Signal', color='red')
        ax.set_xlabel('Relative Time (seconds)')
        ax.set_ylabel('Velocity (c/s)')
        ax.set_title('Seismic Signal (Raw vs. Filtered)')
        ax.legend()
        st.pyplot(fig)

    # STA/LTA ratio detection method
    def sta_lta_fixed(signal, short_window, long_window):
        sta = np.cumsum(np.abs(signal)**2)
        sta = (sta[short_window:] - sta[:-short_window]) / short_window
        lta = (sta[long_window:] - sta[:-long_window]) / long_window
        sta = sta[:len(lta)]  # Ensure same length
        return sta / (lta + 1e-9)

    # STA/LTA Detection with windows based on the sampling rate
    short_window = int(0.5 * sampling_rate)
    long_window = int(10 * sampling_rate)
    sta_lta_ratio = sta_lta_fixed(filtered_signal, short_window, long_window)

    # STA/LTA Threshold Slider
    threshold = st.sidebar.slider("STA/LTA Detection Threshold", min_value=1.0, max_value=10.0, value=3.0)

    # Detect seismic events where STA/LTA ratio exceeds the threshold
    seismic_events = time[:len(sta_lta_ratio)][sta_lta_ratio > threshold]

    # Plot STA/LTA ratio and detected events
    with st.expander("📈 Seismic Event Detection (STA/LTA Method)"):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time[:len(sta_lta_ratio)], sta_lta_ratio, label='STA/LTA Ratio')
        ax.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
        ax.scatter(seismic_events, np.ones_like(seismic_events) * threshold, color='red', marker='x', label='Detected Events')
        ax.set_xlabel('Relative Time (seconds)')
        ax.set_ylabel('STA/LTA Ratio')
        ax.set_title('Seismic Event Detection')
        ax.legend()
        st.pyplot(fig)

    st.write("### Detected seismic event times (seconds):")
    st.write(seismic_events)

    # FFT Analysis of the filtered signal
    with st.expander("🔍 FFT of Filtered Seismic Signal"):
        N = len(filtered_signal)
        T = 1 / sampling_rate
        yf = fft(filtered_signal)
        xf = fftfreq(N, T)[:N // 2]
        fig = px.line(x=xf, y=2.0 / N * np.abs(yf[0:N // 2]), title="FFT of Filtered Seismic Signal")
        fig.update_layout(xaxis_title="Frequency (Hz)", yaxis_title="Amplitude")
        st.plotly_chart(fig)

    # Spectrogram of the filtered signal
    with st.expander("🔍 Spectrogram of Filtered Seismic Signal"):
        frequencies, times, Sxx = spectrogram(filtered_signal, fs=sampling_rate)
        fig = px.imshow(10 * np.log10(Sxx), x=times, y=frequencies, origin='lower', aspect='auto', color_continuous_scale='Viridis')
        fig.update_layout(title="Spectrogram of Filtered Seismic Signal", xaxis_title="Time (seconds)", yaxis_title="Frequency (Hz)")
        fig.update_yaxes(range=[0, 10])
        st.plotly_chart(fig)

    # Event properties extraction for classification
    st.write("### Event Properties for Classification")
    event_properties = []
    for event_time in seismic_events:
        index = np.where(np.isclose(time, event_time, atol=1e-3))[0]
        if len(index) > 0:
            idx = index[0]
            amplitude = np.max(np.abs(filtered_signal[idx - short_window:idx + short_window]))
            duration = len(sta_lta_ratio[idx - short_window:idx + short_window]) / sampling_rate
            event_properties.append((event_time, amplitude, duration))

    # Display event properties in a DataFrame
    event_properties_df = pd.DataFrame(event_properties, columns=['Time (s)', 'Amplitude', 'Duration (s)'])
    st.write(event_properties_df)

    # Download button for event properties
    st.download_button(
        label="Download Event Data as CSV",
        data=event_properties_df.to_csv(index=False),
        file_name='seismic_event_properties.csv',
        mime='text/csv',
    )

    # Classification of seismic events
    if len(event_properties) > 0:
        features = np.array(event_properties_df[['Amplitude', 'Duration (s)']])
        labels = ["seismic_event"] * len(features)

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

        # RandomForestClassifier for classification
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        # Display classification report
        st.write("### Classification Report")
        st.text(classification_report(y_test, y_pred))

else:
    st.warning("Please upload a CSV file with seismic data.")
