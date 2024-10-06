import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fftpack import fft
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import time

# Cache for performance optimization
@st.cache_data
def load_data(file):
    try:
        data = pd.read_csv(file)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def preprocess_data(data, apply_normalization=True):
    if 'time_rel(sec)' not in data.columns or 'velocity(m/s)' not in data.columns:
        st.error("Required columns missing: 'time_rel(sec)', 'velocity(m/s)'")
        return None, None
    
    time = data['time_rel(sec)']
    signal = data['velocity(m/s)']
    
    if apply_normalization:
        signal = (signal - signal.mean()) / signal.std()
    
    return time, signal

def plot_time_series(time, signal, title):
    fig, ax = plt.subplots()
    ax.plot(time, signal, label='Signal')
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)

def apply_fft(signal):
    fft_result = fft(signal)
    fft_magnitude = np.abs(fft_result)
    return fft_magnitude

def plot_fft(fft_magnitude, sample_rate):
    freq = np.fft.fftfreq(len(fft_magnitude), d=1/sample_rate)
    plt.plot(freq[:len(fft_magnitude)//2], fft_magnitude[:len(fft_magnitude)//2])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('FFT Spectrum')
    st.pyplot(plt)

def display_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    st.pyplot(plt)

# Main app
st.title("TerraPulse - Seismic Event Detection & Comparison")
st.sidebar.title("Navigation")
section = st.sidebar.selectbox("Select a section", ['Data Upload', 'Preprocessing & Visualization', 'Event Detection & Classification', 'Model Training & Tuning'])

# Data upload and display
if section == 'Data Upload':
    st.header("Upload your seismic datasets")
    uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True, type=['csv'])

    if uploaded_files:
        datasets = {}
        for uploaded_file in uploaded_files:
            data = load_data(uploaded_file)
            if data is not None:
                st.write(f"Preview of {uploaded_file.name}")
                st.write(data.head())
                datasets[uploaded_file.name] = data
        if len(datasets) > 1:
            st.write("Multiple datasets loaded. You can now compare them.")

# Data preprocessing and visualization
if section == 'Preprocessing & Visualization':
    st.header("Preprocess & Visualize Data")

    if 'datasets' not in st.session_state:
        st.warning("Upload data first in the Data Upload section.")
    else:
        selected_dataset = st.selectbox("Choose a dataset for preprocessing", list(st.session_state['datasets'].keys()))
        data = st.session_state['datasets'][selected_dataset]
        time, signal = preprocess_data(data)
        
        if time is not None and signal is not None:
            st.subheader("Time-Series Data")
            plot_time_series(time, signal, 'Seismic Signal')

            # Apply FFT and visualize frequency domain
            fft_button = st.button("Show FFT Analysis")
            if fft_button:
                fft_magnitude = apply_fft(signal)
                plot_fft(fft_magnitude, sample_rate=1 / (time.iloc[1] - time.iloc[0]))

# Event Detection & Classification
if section == 'Event Detection & Classification':
    st.header("Event Detection & Classification")

    if 'datasets' not in st.session_state:
        st.warning("Upload and preprocess data first.")
    else:
        selected_dataset = st.selectbox("Choose a dataset for classification", list(st.session_state['datasets'].keys()))
        data = st.session_state['datasets'][selected_dataset]
        time, signal = preprocess_data(data)
        
        if time is not None and signal is not None:
            # Simple event detection logic: trigger when signal > threshold
            threshold = st.slider("Detection threshold", min_value=0.1, max_value=5.0, value=1.0)
            detected_events = (signal > threshold).astype(int)
            st.write(f"Number of events detected: {detected_events.sum()}")

            st.subheader("Detected Events")
            st.line_chart(detected_events)

# Machine Learning Model Training and Tuning
if section == 'Model Training & Tuning':
    st.header("Train & Tune a Machine Learning Model")

    if 'datasets' not in st.session_state:
        st.warning("Upload and preprocess data first.")
    else:
        selected_dataset = st.selectbox("Choose a dataset for model training", list(st.session_state['datasets'].keys()))
        data = st.session_state['datasets'][selected_dataset]
        time, signal = preprocess_data(data)

        if time is not None and signal is not None:
            st.write("Select Features and Parameters")
            feature_cols = st.multiselect("Select features", ['Amplitude', 'FFT Peak', 'Signal Energy'])
            target_col = st.selectbox("Select target column (if available)", data.columns)
            
            if st.button("Train Model"):
                X = data[feature_cols]
                y = data[target_col]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
                rf_model.fit(X_train, y_train)

                y_pred = rf_model.predict(X_test)
                st.subheader("Classification Report")
                st.text(classification_report(y_test, y_pred))

                st.subheader("Confusion Matrix")
                display_confusion_matrix(y_test, y_pred)

                st.subheader("Feature Importance")
                feature_importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
                st.bar_chart(feature_importances)

# Footer
st.sidebar.write("---")
st.sidebar.write("© 2024 TerraPulse Seismic Analysis")

