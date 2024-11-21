import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

# Simulation parameters
fs = 500  # Sampling frequency in Hz
duration = 2  # Duration of display window in seconds
total_time = 10  # Total simulation time in seconds
t = np.linspace(0, total_time, int(fs * total_time), endpoint=False)

# Function to generate synthetic ECG signal
def generate_ecg(t, heart_rate=60, noise_std=0.05):
    rr_interval = 60 / heart_rate  # seconds per beat
    qrs_wave = np.exp(-np.square((t % rr_interval) - rr_interval / 2) / (2 * 0.03**2)) * 1.2
    p_wave = 0.1 * np.sin(2 * np.pi * t / rr_interval)
    t_wave = 0.15 * np.sin(4 * np.pi * t / rr_interval)
    noise = np.random.normal(0, noise_std, t.shape)  # baseline noise
    ecg_signal = qrs_wave + p_wave + t_wave + noise
    return ecg_signal

# Generating synthetic ECG signal
ecg_signal = generate_ecg(t)


labels = np.random.choice([0, 1], len(ecg_signal))  # 0 for Normal, 1 for Abnormal


segment_length = 100
X = np.array([ecg_signal[i:i+segment_length] for i in range(0, len(ecg_signal) - segment_length)])
y = labels[:len(X)]

# Building a simple TensorFlow model for classification
model = Sequential([
    Dense(64, activation='relu', input_shape=(segment_length,)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')  # Output for binary classification
])

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training  the final abnormal and normal heat wave classification model
model.fit(X, y, epochs=10, batch_size=32, verbose=1)

# Seting up up the final plot
fig, ax = plt.subplots()
ax.set_xlim(0, duration)
ax.set_ylim(-0.5, 1.5)
line, = ax.plot([], [], lw=2)
label_text = ax.text(1.5, 1.2, '', fontsize=15, color='red')

# Animating the graph function using the 1st declared code block
def update(frame):
    start_idx = frame
    end_idx = start_idx + int(duration * fs)
    xdata = np.linspace(0, duration, end_idx - start_idx)
    ydata = ecg_signal[start_idx:end_idx]
    
    # Predict using the model on the latest segment
    segment = ecg_signal[start_idx:end_idx][:segment_length]
    segment = segment.reshape(1, -1)  # Reshape for model
    prediction = model.predict(segment)
    predicted_label = np.argmax(prediction)

    # Update plot data
    line.set_data(xdata, ydata)
    
    # Display label prediction
    label_text.set_text('Abnormal' if predicted_label == 1 else 'Normal')
    
    return line, label_text

# Run the animation
ani = animation.FuncAnimation(fig, update, frames=np.arange(0, len(t) - int(duration * fs)), blit=True, interval=1000 / fs)

plt.title("ECG Simulation with Real-Time Classification")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()
