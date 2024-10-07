import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler

# Data loading with caching
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

def sta_lta_fixed(signal, short_window, long_window):
    """Compute the STA/LTA ratio."""
    sta = np.convolve(signal**2, np.ones(short_window), mode='same') / short_window
    lta = np.convolve(signal**2, np.ones(long_window), mode='same') / long_window
    
    with np.errstate(divide='ignore', invalid='ignore'):
        sta_lta_ratio = np.where(lta != 0, sta / lta, 0)
    
    return sta_lta_ratio

def butter_filter(filter_type, cutoff, fs, order=5, cutoff2=None):
    nyq = 0.5 * fs
    if filter_type == "Highpass":
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    elif filter_type == "Lowpass":
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    elif filter_type == "Bandpass":
        normal_cutoff = [cutoff / nyq, cutoff2 / nyq]
        b, a = signal.butter(order, normal_cutoff, btype='band', analog=False)
    elif filter_type == "Notch":
        bandwidth = 1.0
        low = (cutoff - bandwidth / 2) / nyq
        high = (cutoff + bandwidth / 2) / nyq
        normal_cutoff = [low, high]
        b, a = signal.butter(order, normal_cutoff, btype='bandstop', analog=False)
    return b, a

def apply_filter(data, filter_type, cutoff, fs, order=5, cutoff2=None):
    b, a = butter_filter(filter_type, cutoff, fs, order, cutoff2)
    y = signal.filtfilt(b, a, data)
    return y

def characterize_event(signal, event_start, event_end, sampling_rate):
    event_signal = signal[event_start:event_end]
    
    duration = (event_end - event_start) / sampling_rate
    peak_amplitude = np.max(np.abs(event_signal))
    freqs, psd = signal.welch(event_signal, fs=sampling_rate, nperseg=min(256, len(event_signal)))
    dominant_freq = freqs[np.argmax(psd)]
    
    return {
        "duration": duration,
        "peak_amplitude": peak_amplitude,
        "dominant_frequency": dominant_freq
    }

def plot_seismic_events(time, signal, filtered_signal, sta_lta_ratio, threshold, seismic_events, plot_options):
    fig = go.Figure()
    
    if plot_options['raw_signal']:
        fig.add_trace(go.Scatter(x=time, y=signal, name='Raw Signal', line=dict(color='blue')))
    
    if plot_options['filtered_signal']:
        fig.add_trace(go.Scatter(x=time, y=filtered_signal, name='Filtered Signal', line=dict(color='red')))
    
    if plot_options['sta_lta_ratio']:
        fig.add_trace(go.Scatter(x=time, y=sta_lta_ratio, name='STA/LTA Ratio', line=dict(color='orange'), yaxis='y2'))
    
    if plot_options['detected_events']:
        fig.add_trace(go.Scatter(x=seismic_events, y=[threshold]*len(seismic_events), mode='markers', 
                                 name='Detected Events', marker=dict(color='green', size=10)))

    fig.update_layout(
        title='Seismic Signal Analysis and Event Detection',
        xaxis_title='Time (seconds)',
        yaxis_title='Velocity (m/s)',
        yaxis2=dict(title='STA/LTA Ratio', overlaying='y', side='right'),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    return fig

def main():
    st.title("🌋 Seismic Event Detection & Analysis")
    st.sidebar.title("Settings")

    uploaded_files = st.sidebar.file_uploader(
        "Upload Seismic Data Files", type=["csv"], accept_multiple_files=True
    )

    if uploaded_files:
        st.sidebar.success("Files uploaded successfully!")
        datasets = [load_data(f) for f in uploaded_files]
        file_options = [f.name for f in uploaded_files]

        selected_files = st.sidebar.multiselect(
            "Select Files for Analysis", file_options, default=file_options[0]
        )

        if not selected_files:
            st.warning("Please select at least one file for analysis.")
            return

        # Customizable Visualization
        st.sidebar.subheader("Plot Options")
        plot_options = {
            'raw_signal': st.sidebar.checkbox("Show Raw Signal", value=True),
            'filtered_signal': st.sidebar.checkbox("Show Filtered Signal", value=True),
            'sta_lta_ratio': st.sidebar.checkbox("Show STA/LTA Ratio", value=True),
            'detected_events': st.sidebar.checkbox("Show Detected Events", value=True)
        }

        filter_type = st.sidebar.selectbox("Select Filter Type", ("Highpass", "Lowpass", "Bandpass", "Notch"), index=0)
        cutoff_frequency = st.sidebar.slider("Filter Cutoff Frequency (Hz)", min_value=0.01, max_value=50.0, value=0.1)
        order = st.sidebar.slider("Filter Order", min_value=1, max_value=10, value=5)

        cutoff_frequency_2 = None
        if filter_type in ["Bandpass", "Notch"]:
            cutoff_frequency_2 = st.sidebar.slider("Second Cutoff Frequency (Hz)", min_value=0.01, max_value=50.0, value=0.5)

        normalize_option = st.sidebar.checkbox("Normalize Signal")

        for selected_file in selected_files:
            data = next(df for i, df in enumerate(datasets) if file_options[i] == selected_file)

            st.header(f"**File: {selected_file}**")
            
            time_columns = [col for col in data.columns if 'time' in col.lower()]
            velocity_columns = [col for col in data.columns if 'velocity' in col.lower()]

            if not time_columns or not velocity_columns:
                st.error("Required columns not found. Please ensure your CSV has time and velocity columns.")
                continue

            time_col = st.selectbox(f"Select Time Column for {selected_file}", time_columns, index=0)
            velocity_col = st.selectbox(f"Select Velocity Column for {selected_file}", velocity_columns, index=0)

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

            with st.expander("🔍 Data Preview"):
                st.write(data.head())

            if normalize_option:
                scaler = StandardScaler()
                signal = scaler.fit_transform(signal.values.reshape(-1, 1)).flatten()
                st.info("Signal has been normalized.")

            try:
                sampling_rate = 1.0 / (time.iloc[1] - time.iloc[0])
            except Exception as e:
                st.error(f"Error calculating sampling rate: {e}")
                continue

            filtered_signal = apply_filter(signal, filter_type, cutoff_frequency, sampling_rate, order, cutoff_frequency_2)

            short_window = int(0.5 * sampling_rate)
            long_window = int(10 * sampling_rate)
            sta_lta_ratio = sta_lta_fixed(filtered_signal, short_window, long_window)

            threshold = st.sidebar.slider("STA/LTA Detection Threshold", min_value=1.0, max_value=10.0, value=3.0)

            seismic_events = time[sta_lta_ratio > threshold]

            fig = plot_seismic_events(time, signal, filtered_signal, sta_lta_ratio, threshold, seismic_events, plot_options)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Detected Seismic Events")
            st.write(f"Number of events detected: {len(seismic_events)}")
            st.write("Event timestamps:")
            st.write(seismic_events)

            # Event Characterization
            event_characteristics = []
            for event_time in seismic_events:
                event_start = np.argmin(np.abs(time - event_time))
                event_end = min(event_start + int(10 * sampling_rate), len(signal))  # 10-second window or end of signal
                characteristics = characterize_event(filtered_signal, event_start, event_end, sampling_rate)
                event_characteristics.append(characteristics)

            # Display event characteristics
            st.subheader("Event Characteristics")
            event_df = pd.DataFrame(event_characteristics)
            st.write(event_df)

if __name__ == "__main__":
    main()
