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

# Updated caching decorator
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

def main():
    st.title("🌋 Seismic Event Detection & Classification")
    st.sidebar.title("Settings")

    # File upload and dynamic file handling
    uploaded_files = st.sidebar.file_uploader(
        "Upload Seismic Data Files", type=["csv"], accept_multiple_files=True
    )

    if uploaded_files:
        st.sidebar.success("Files uploaded successfully!")
        datasets = [load_data(f) for f in uploaded_files]
        file_options = [f.name for f in uploaded_files]

        # Allow user to select multiple files for comparison
        selected_files = st.sidebar.multiselect(
            "Select Files for Comparison", file_options, default=file_options
        )

        if not selected_files:
            st.warning("Please select at least one file for analysis.")
            return

        # Model selection
        model_option = st.sidebar.selectbox(
            "Select Model", ("RandomForest", "Gradient Boosting", "SVM")
        )

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

        # Loop over selected files for comparison
        for selected_file in selected_files:
            # Retrieve the corresponding dataframe
            data = next(
                df for i, df in enumerate(datasets) if file_options[i] == selected_file
            )

            st.header(f"**File: {selected_file}**")

            # Display available columns
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
            time_col = st.selectbox(
                f"Select Time Column for {selected_file}", time_columns, index=0
            )
            velocity_col = st.selectbox(
                f"Select Velocity Column for {selected_file}", velocity_columns, index=0
            )

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

            # Data preview
            with st.expander("🔍 Data Preview"):
                st.write("First 5 rows of the seismic data:")
                st.write(data.head())

            # Data normalization/standardization
            normalize_option = st.sidebar.checkbox("Normalize/Standardize Signal")
            if normalize_option:
                scaler = StandardScaler()
                signal = scaler.fit_transform(signal.values.reshape(-1, 1)).flatten()
                st.info("Signal has been standardized.")

            # Filter settings
            filter_type = st.sidebar.selectbox(
                "Select Filter Type", ("Highpass", "Lowpass", "Bandpass", "Notch"), index=0
            )
            cutoff_frequency = st.sidebar.slider(
                "Filter Cutoff Frequency (Hz)", min_value=0.01, max_value=50.0, value=0.1
            )
            order = st.sidebar.slider("Filter Order", min_value=1, max_value=10, value=5)

            # Additional filter parameters for bandpass and notch
            if filter_type in ["Bandpass", "Notch"]:
                cutoff_frequency_2 = st.sidebar.slider(
                    "Second Cutoff Frequency (Hz)", min_value=0.01, max_value=50.0, value=0.5
                )
            else:
                cutoff_frequency_2 = None

            # Highpass filter function
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

            # Sampling rate
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

            # Plot raw and filtered signals
            with st.expander("🔍 Seismic Signal (Raw vs. Filtered)"):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(time, signal, label='Raw Signal')
                ax.plot(time, filtered_signal, label='Filtered Signal', color='red')
                ax.set_xlabel('Relative Time (seconds)')
                ax.set_ylabel('Velocity (m/s)')
                ax.set_title('Seismic Signal (Raw vs. Filtered)')
                ax.legend()
                st.pyplot(fig)

            # STA/LTA ratio detection method
            def sta_lta_fixed(signal, short_window, long_window):
                sta = np.cumsum(signal ** 2)
                sta = (sta[short_window:] - sta[:-short_window]) / short_window
                lta = (sta[long_window:] - sta[:-long_window]) / long_window
                sta = sta[:len(lta)]
                return sta / (lta + 1e-9)

            # STA/LTA Detection
            short_window = int(0.5 * sampling_rate)
            long_window = int(10 * sampling_rate)
            sta_lta_ratio = sta_lta_fixed(filtered_signal, short_window, long_window)

            # STA/LTA Threshold Slider
            threshold = st.sidebar.slider("STA/LTA Detection Threshold", min_value=1.0, max_value=10.0, value=3.0)

            # Find seismic events
            seismic_events = time[:len(sta_lta_ratio)][sta_lta_ratio > threshold]

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

            st.write("### Detected Seismic Event Times (seconds):")
            st.write(seismic_events)

            # FFT Analysis
            with st.expander("🔍 FFT of Filtered Seismic Signal"):
                N = len(filtered_signal)
                T = 1 / sampling_rate
                yf = fft(filtered_signal)
                xf = fftfreq(N, T)[:N // 2]
                peak_frequency = xf[np.argmax(np.abs(yf[:N // 2]))]
                fig_fft = px.line(x=xf, y=2.0 / N * np.abs(yf[0:N // 2]), title="FFT of Filtered Seismic Signal")
                fig_fft.update_layout(xaxis_title="Frequency (Hz)", yaxis_title="Amplitude")
                st.plotly_chart(fig_fft)
                st.write(f"**Peak Frequency:** {peak_frequency:.2f} Hz")

            # Spectrogram
            with st.expander("🔍 Spectrogram of Filtered Seismic Signal"):
                frequencies, times_spec, Sxx = spectrogram(filtered_signal, fs=sampling_rate)
                fig_spectrogram = px.imshow(
                    10 * np.log10(Sxx),
                    x=times_spec,
                    y=frequencies,
                    origin='lower',
                    aspect='auto',
                    color_continuous_scale='Viridis',
                    title="Spectrogram of Filtered Seismic Signal"
                )
                fig_spectrogram.update_layout(xaxis_title="Time (seconds)", yaxis_title="Frequency (Hz)")
                fig_spectrogram.update_yaxes(range=[0, 10])
                st.plotly_chart(fig_spectrogram)

            # Event properties table
            st.write("### Event Properties for Classification")
            event_properties = []
            for event_time in seismic_events:
                idx = np.where(np.isclose(time, event_time, atol=1e-3))[0]
                if len(idx) > 0:
                    center = idx[0]
                    start = max(center - short_window, 0)
                    end = min(center + short_window, len(filtered_signal))
                    amplitude = np.max(filtered_signal[start:end])
                    duration = (end - start) / sampling_rate
                    # Additional features
                    event_signal = filtered_signal[start:end]
                    yf_event = fft(event_signal)
                    xf_event = fftfreq(len(event_signal), T)[:len(event_signal) // 2]
                    peak_freq_event = xf_event[np.argmax(np.abs(yf_event[:len(yf_event) // 2]))]
                    event_energy = np.sum(event_signal ** 2)
                    sta_lta_value = sta_lta_ratio[center] if center < len(sta_lta_ratio) else np.nan
                    event_properties.append((
                        event_time,
                        amplitude,
                        duration,
                        peak_freq_event,
                        event_energy,
                        sta_lta_value
                    ))

            event_properties_df = pd.DataFrame(event_properties, columns=[
                'Time (s)', 'Amplitude', 'Duration (s)', 'Peak Frequency (Hz)', 'Event Energy', 'STA/LTA Ratio'
            ])

            st.write(event_properties_df)

            # Allow users to download event properties as CSV
            st.download_button(
                label="Download Event Data as CSV",
                data=event_properties_df.to_csv(index=False).encode('utf-8'),
                file_name=f'seismic_event_properties_{selected_file}.csv',
                mime='text/csv',
            )

            # Classification
            if not event_properties_df.empty:
                st.subheader("Machine Learning Classification")
                # Feature selection
                features = ['Amplitude', 'Duration (s)', 'Peak Frequency (Hz)', 'Event Energy', 'STA/LTA Ratio']
                X = event_properties_df[features].fillna(0)  # Handle any NaNs
                y = ["seismic_event"] * len(X)  # Dummy labels; replace with actual labels if available

                # Feature Scaling
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Model selection
                if model_option == "RandomForest":
                    model = RandomForestClassifier(
                        n_estimators=n_estimators, max_depth=max_depth, random_state=42
                    )
                elif model_option == "Gradient Boosting":
                    model = GradientBoostingClassifier(
                        n_estimators=n_estimators, learning_rate=learning_rate, random_state=42
                    )
                elif model_option == "SVM":
                    model = SVC(C=C, kernel=kernel, probability=True, random_state=42)

                # Cross-Validation Setup
                kfold = KFold(n_splits=5, shuffle=True, random_state=42)
                classification_reports = []
                confusion_matrices = []

                for train_index, test_index in kfold.split(X_scaled):
                    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    classification_reports.append(classification_report(y_test, y_pred, output_dict=True))
                    confusion_matrices.append(confusion_matrix(y_test, y_pred))

                # Aggregate Classification Report
                avg_report = {}
                for key in classification_reports[0].keys():
                    if key != 'accuracy':
                        avg_report[key] = {
                            metric: np.mean([report[key][metric] for report in classification_reports])
                            for metric in classification_reports[0][key].keys()
                        }
                    else:
                        avg_report[key] = np.mean([report[key] for report in classification_reports])

                st.write("### Averaged Classification Report (5-Fold Cross-Validation)")
                st.json(avg_report)

                # Aggregate Confusion Matrix
                avg_cm = np.mean(confusion_matrices, axis=0).astype(int)
                fig_cm = plt.figure(figsize=(5, 4))
                sns.heatmap(avg_cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Average Confusion Matrix')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                st.pyplot(fig_cm)

                # Feature Importance (for tree-based models)
                if model_option in ["RandomForest", "Gradient Boosting"]:
                    importances = model.feature_importances_
                    feature_importance_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': importances
                    }).sort_values(by='Importance', ascending=False)

                    st.write("### Feature Importance")
                    fig_fi = px.bar(
                        feature_importance_df,
                        x='Feature',
                        y='Importance',
                        title='Feature Importance',
                        labels={'Importance': 'Importance Score'}
                    )
                    st.plotly_chart(fig_fi)

            else:
                st.warning("No seismic events detected for classification.")

    else:
        st.warning("Please upload at least one CSV file with seismic data.")

if __name__ == "__main__":
    main()
