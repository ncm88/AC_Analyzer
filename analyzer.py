import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cmath

#Just a basic tool I made to simplify the data processing for AC waveforms, modify as needed

path = "../data/lab1.2a5.10v.csv"
data = pd.read_csv(path, delimiter='\t')
time = data['Time (ms)'] * 1e-3
voltage1 = data['Ch1 V']
voltage2 = data['Ch2 V']
voltage3 = data['Ch3 V']
current1 = data['Ch1 A']
current2 = data['Ch2 A']
current3 = data['Ch3 A']

def plot_waveform(time, data, y_label):
    plt.figure(figsize=(14, 6))
    plt.plot(time, data)
    plt.xlabel("Time (s)")
    plt.ylabel(y_label)
    plt.title(f"Time-Domain Plot of {y_label}")
    plt.grid(True)
    plt.show()

def power_analysis(time, voltage, current):
    # Compute instantaneous power
    instantaneous_power = voltage * current

    # Compute real and reactive power
    real_power = np.mean(instantaneous_power)
    reactive_power = np.sqrt(np.mean(voltage**2) * np.mean(current**2) - real_power**2)

    # Plot results on separate subplots
    fig, ax = plt.subplots(3, 1, figsize=(14, 15))

    # Instantaneous power
    ax[0].plot(time, instantaneous_power)
    ax[0].set_title("Instantaneous Power")
    ax[0].set_ylabel("Power (W)")
    ax[0].grid(True)

    # Real power
    ax[1].axhline(y=real_power, color='r', linestyle='-')
    ax[1].set_title("Real Power")
    ax[1].set_ylabel("Power (W)")
    ax[1].grid(True)

    # Reactive power
    ax[2].axhline(y=reactive_power, color='b', linestyle='-')
    ax[2].set_title("Reactive Power")
    ax[2].set_ylabel("Power (VAR)")
    ax[2].set_xlabel("Time (s)")
    ax[2].grid(True)
    plt.tight_layout()
    plt.show()
    return real_power, reactive_power


def harmonic_analysis(time, data, y_label):
    num_samples = len(data)
    sampling_frequency = 1 / (time[1] - time[0])
    frequencies = np.fft.fftfreq(num_samples, 1/sampling_frequency)
    spectrum = np.fft.fft(data)
    magnitude_spectrum = np.abs(spectrum) / num_samples

    positive_freq_indices = np.where(frequencies > 0)
    magnitude_spectrum_normalized = magnitude_spectrum[positive_freq_indices] / magnitude_spectrum[positive_freq_indices][0]

    plt.figure(figsize=(14, 6))
    plt.stem(frequencies[positive_freq_indices], magnitude_spectrum_normalized, use_line_collection=True)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(f"Amplitude (Normalized to {y_label} Fundamental)")
    plt.title(f"Harmonic Content of {y_label}")
    plt.grid(True)
    plt.xlim(0, 2000)
    plt.show()


def compute_phasor(data):
    magnitude = np.mean(data)
    angle = cmath.phase(np.mean(data))
    return magnitude * np.exp(1j * angle)


def phasor_diagram(phasors):
    plt.figure(figsize=(10, 10))
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    for idx, (label, phasor) in enumerate(phasors.items()):
        plt.quiver(0, 0, phasor.real, phasor.imag, angles='xy', scale_units='xy', scale=1, color=colors[idx], label=label)

    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.title("Phasor Diagram")
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.show()
