import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fftpack import fft
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import gc

# Function to load data in batches
def load_data_in_batches(file, batch_size=10000):
    chunks = []
    try:
        # Load data in chunks
        for chunk in pd.read_csv(file, chunksize=batch_size):
            chunks.append(chunk)
            if len(chunks) > 0:
                st.write(f"Loaded {len(chunks) * batch_size} rows...")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None
    
    # Concatenate the chunks into a single DataFrame
    data = pd.concat(chunks, axis=0)
    return data

# Function to preprocess and normalize data
def preprocess_data(data):
    if 'time_rel(sec)' not in data.columns or 'velocity(m/s)' not in data.columns:
        st.error("Required columns missing: 'time_rel(sec)', 'velocity(m/s)'")
        return None, None
    
    time = data['time_rel(sec)']
    signal = data['velocity(m/s)']
    
    # Normalize the signal
    signal = (signal - signal.mean()) / signal.std()
    return time, signal

# Plotting function (avoid replotting unnecessarily)
def plot_time_series(time, signal, title):
    fig, ax = plt.subplots()
    ax.plot(time, signal)
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title(title)
    st.pyplot(fig)

# Apply FFT and visualize it
def apply_fft(signal):
    fft_result = fft(signal)
    fft_magnitude = np.abs(fft_result)
    return fft_magnitude

def plot_fft(fft_magnitude, sample_rate):
    freq = np.fft.fftfreq(len(fft_magnitude), d=1/sample_rate)
    fig, ax = plt.subplots()
    ax.plot(freq[:len(fft_magnitude)//2], fft_magnitude[:len(fft_magnitude)//2])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.set_title('FFT Spectrum')
    st.pyplot(fig)

# Confusion matrix display
def display_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    st.pyplot(plt)

# Main app logic
st.title("TerraPulse - Seismic Event Detection & Comparison")

# Sidebar for navigation
section = st.sidebar.selectbox("Select a section", ['Data Upload', 'Preprocessing & Visualization', 'Event Detection & Classification', 'Model Training & Tuning'])

# Data upload section
if section == 'Data Upload':
    st.header("Upload your seismic datasets")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file:
        batch_size = st.number_input("Batch size (rows to load at a time)", min_value=1000, max_value=100000, value=10000)
        data = load_data_in_batches(uploaded_file, batch_size=batch_size)
        if data is not None:
            st.write(f"Loaded data preview ({batch_size} rows per batch):")
            st.write(data.head())
            st.session_state['data'] = data

# Preprocessing & Visualization section
if section == 'Preprocessing & Visualization':
    st.header("Preprocess & Visualize Data")

    if 'data' not in st.session_state:
        st.warning("Upload data first.")
    else:
        data = st.session_state['data']
        time, signal = preprocess_data(data)
        
        if time is not None and signal is not None:
            st.subheader("Time-Series Data")
            plot_time_series(time, signal, 'Seismic Signal')

            # Apply FFT and visualize frequency domain
            if st.button("Show FFT Analysis"):
                fft_magnitude = apply_fft(signal)
                sample_rate = 1 / (time.iloc[1] - time.iloc[0])
                plot_fft(fft_magnitude, sample_rate)

# Event Detection & Classification section
if section == 'Event Detection & Classification':
    st.header("Event Detection & Classification")

    if 'data' not in st.session_state:
        st.warning("Upload and preprocess data first.")
    else:
        data = st.session_state['data']
        time, signal = preprocess_data(data)
        
        if time is not None and signal is not None:
            # Detection logic
            threshold = st.slider("Detection threshold", min_value=0.1, max_value=5.0, value=1.0)
            detected_events = (signal > threshold).astype(int)
            st.write(f"Number of events detected: {detected_events.sum()}")

            st.subheader("Detected Events")
            st.line_chart(detected_events)

# Model Training & Tuning section
if section == 'Model Training & Tuning':
    st.header("Train & Tune a Machine Learning Model")

    if 'data' not in st.session_state:
        st.warning("Upload and preprocess data first.")
    else:
        data = st.session_state['data']
        time, signal = preprocess_data(data)

        if time is not None and signal is not None:
            feature_cols = st.multiselect("Select features", ['Amplitude', 'FFT Peak', 'Signal Energy'])
            if len(feature_cols) == 0:
                st.warning("Please select at least one feature.")
            else:
                X = data[feature_cols]
                y = st.session_state['data']['target']  # Placeholder for real target variable

                if st.button("Train Model"):
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
                    rf_model.fit(X_train, y_train)
                    
                    y_pred = rf_model.predict(X_test)
                    st.subheader("Classification Report")
                    st.text(classification_report(y_test, y_pred))

                    st.subheader("Confusion Matrix")
                    display_confusion_matrix(y_test, y_pred)

                    # Show feature importance
                    st.subheader("Feature Importance")
                    feature_importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
                    st.bar_chart(feature_importances)

# Perform garbage collection to free memory
gc.collect()
