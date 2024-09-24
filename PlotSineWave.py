import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive

# Plot a sine wave
def plot_sine_wave(frequency=.5):
    x = np.linspace(0, 4 * np.pi, 1000)
    y = np.sin(frequency * x)
    plt.figure(figsize=(10, 4))
    plt.plot(x, y)
    plt.title(f'Sine Wave with Frequency = {frequency} Hz')
    plt.xlabel('X')
    plt.ylabel('sin(X)')
    plt.grid(True)
    plt.ylim(-1.1, 1.1)
    plt.show()

# Use ipywidgets to make an interactive widget
interactive_plot = interactive(plot_sine_wave, frequency=(0.1, 15.0, 0.1))
interactive_plot