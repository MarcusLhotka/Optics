"""
Fraunhofer Diffraction for Square Aperture

Marcus Lhotka
EE 5621 Physical Optics
Homework Assignment 5
"""

import numpy as np
import numpy.fft as ft
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from matplotlib.widgets import Button
import matplotlib

# Create a 1024x1024 array filled with zeros
realSpace = np.zeros((1024, 1024))
# Calculate the width of the central lines
linewidth = 1024 // 9  # This ensures a ratio of 8:1 zeros per one
# Set the size of the central aperture
realSpace[512-linewidth//2:512+linewidth//2, 512-linewidth//2:512+linewidth//2] = 1

#1024/9=113.77 pixels per section. Lens extends for 2 sections so total pixels are 227.55.
#pixels per mm conversion
pxTo_mm = 1 / 227
extent_mm = [0, 1024 * pxTo_mm, 1024 * pxTo_mm, 0]
extentInvs_mm = [0, 1024 // pxTo_mm, 1024 // pxTo_mm, 0]
freqSpace = ft.fftshift(ft.fft2(realSpace))
absFreqSpace = np.abs(freqSpace)
absFreqSpace = absFreqSpace / np.max(absFreqSpace)
# Compute frequency axes
fx = np.fft.fftshift(np.fft.fftfreq(1024, d=pxTo_mm))
fy = np.fft.fftshift(np.fft.fftfreq(1024, d=pxTo_mm))
extent_freq = [fx.min(), fx.max(), fy.min(), fy.max()]


# Create a new figure
plt.figure(figsize=(10, 10))

# Plot the array
plt.imshow(realSpace, cmap='binary', extent=extent_mm)
# Invert the y-axis to compensate for Array direction
plt.gca().invert_yaxis()

# Add a colorbar
plt.colorbar()

# Add title and labels
plt.title('1mm lens')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')


# Create a new figure
plt.figure(figsize=(10, 10))

# Plot the array
plt.imshow(absFreqSpace, cmap='binary', extent=extent_mm)
# Invert the y-axis to compensate for Array direction
plt.gca().invert_yaxis()

# Add a colorbar
plt.colorbar()

# Add title and labels
plt.title('1mm lens')
plt.xlabel('Fx-axis')
plt.ylabel('Fy-axis')

# Show the plot
plt.show()

