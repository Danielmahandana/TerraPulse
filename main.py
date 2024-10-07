import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Function to apply different filters to the seismic signal
def apply_filter(data, filter_type, cutoff_freq_1, cutoff_freq_2=None, sampling_rate=1.0):
    nyquist = 0.5 * sampling_rate
    if filter_type == "Highpass":
        b, a = butter(4, cutoff_freq_1 / nyquist, btype="highpass")
    elif filter_type == "Lowpass":
        b, a = butter(4, cutoff_freq_1 / nyquist, btype="lowpass")
    elif filter_type == "Bandpass" and cutoff_freq_2:
        b, a = butter(4, [cutoff_freq_1 / nyquist, cutoff_freq_2 / nyquist], btype="bandpass")
    elif filter_type == "Notch" and cutoff_freq_2:
        b, a = butter(4, [cutoff_freq_1 / nyquist, cutoff_freq_2 / nyquist], btype="bandstop")
    else:
        st.error("Invalid filter type or missing second cutoff frequency for Bandpass/Notch.")
        return data
    return filtfilt(b, a, data)

# CNN Model Builder
def build_cnn(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification, adjust as needed
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Main function to run the Streamlit app
def main():
    st.title("Seismic Signal Analysis - TerraPulse")

    st.write("""
    ## Step-by-Step Guide:
    1. Upload a seismic dataset (CSV format).
    2. Select the time column and signal column for analysis.
    3. Apply filters to the signal (optional).
    4. View the filtered signal.
    5. Train and evaluate a CNN on the signal data.
    """)

    # File uploader
    uploaded_file = st.file_uploader("Upload your seismic data CSV file", type="csv")

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


        st.write("## Original Signal")
        st.line_chart(signal)

        # Sidebar options for filtering
        st.sidebar.title("Filter Settings")
        filter_type = st.sidebar.selectbox("Select Filter Type", ["Highpass", "Lowpass", "Bandpass", "Notch"])
        cutoff_frequency_1 = st.sidebar.slider("First Cutoff Frequency (Hz)", 0.1, 10.0, 1.0)

        if filter_type in ["Bandpass", "Notch"]:
            cutoff_frequency_2 = st.sidebar.slider("Second Cutoff Frequency (Hz)", 0.1, 10.0, 3.0)
        else:
            cutoff_frequency_2 = None

        filtered_signal = apply_filter(signal, filter_type, cutoff_frequency_1, cutoff_frequency_2, sampling_rate)

        st.write("## Filtered Signal")
        fig, ax = plt.subplots()
        ax.plot(time, filtered_signal)
        ax.set_title("Filtered Seismic Signal")
        st.pyplot(fig)

        # CNN classification
        st.write("## CNN Signal Classification")

        # Prepare data for CNN
        X = np.array(filtered_signal).reshape(-1, 1)
        y = np.random.randint(0, 2, size=(X.shape[0],))  # Dummy labels, replace with actual labels if available

        # Train/test split
        X_train, X_test = X[:int(0.8*len(X))], X[int(0.8*len(X)):]
        y_train, y_test = y[:int(0.8*len(y))], y[int(0.8*len(y)):]

        # Reshape data for CNN input
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)

        # Build and train the model
        model = build_cnn((X_train.shape[1], 1))

        history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

        # Plot accuracy
        st.write("## Training Accuracy")
        fig_acc, ax_acc = plt.subplots()
        ax_acc.plot(history.history['accuracy'], label='Training Accuracy')
        ax_acc.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax_acc.set_title("CNN Training Accuracy")
        ax_acc.legend()
        st.pyplot(fig_acc)

if __name__ == "__main__":
    main()
