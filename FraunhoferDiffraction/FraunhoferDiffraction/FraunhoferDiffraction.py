"""
Fraunhofer Diffraction for Square Aperture
asda
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

wLength = 1e-3  # 1 micrometer in millimeters
d = 1e6  # 1 km in millimeters
lambdaD = wLength*d

# Create a 1024x1024 array filled with zeros
realSpace = np.zeros((1080, 1080))
# Calculate the width of the central lines
linewidth = 1080 // 9  # This ensures a ratio of 8:1 zeros per one
# Set the size of the central aperture
realSpace[540-linewidth//2:540+linewidth//2, 540-linewidth//2:540+linewidth//2] = 1

freqSpace = ft.fftshift(ft.fft2(realSpace))
absFreqSpace = np.abs(freqSpace)**2
absFreqSpace = absFreqSpace / np.max(absFreqSpace)


#1080/9=120 pixels per section. Lens extends for 2 sections so total pixels are 227.55.
#pixels per mm conversion
pxTo_mm = 1 / 120
extent_mm = [-540*pxTo_mm, 540 * pxTo_mm, 540 * pxTo_mm, -540*pxTo_mm]
# Calculate spatial frequencies
fx = ft.fftshift(ft.fftfreq(1080, d=1))
fy = fx.copy()
#extentInvs_mm= [fx[0], fx[-1], fy[0], fy[-1]]
#Convert into the new spacial domain
extentInvs_mm= [fx[0]*lambdaD, fx[-1]*lambdaD, fy[0]*lambdaD, fy[-1]*lambdaD]


xPrimeLim=50
yPrimeLim=50

# Create a new figure
plt.figure(figsize=(10, 10))
# Plot the lens
plt.imshow(realSpace, cmap='binary', extent=extent_mm)
# Invert the y-axis to compensate for Array direction
plt.gca().invert_yaxis()
# Add a colorbar
plt.colorbar()
# Add title and labels
plt.title('1mm lens')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')


# Plot 1D slice
plt.figure(figsize=(10, 6))
plt.plot(fx, absFreqSpace[:, 540])
plt.title('1D Slice of Fraunhofer Diffraction Pattern')
plt.xlabel('X Prime (mm)')
plt.ylabel('Intensity')


# Plot 1D slice
plt.figure(figsize=(10, 6))
plt.plot(fx, absFreqSpace[540, :])
plt.title('1D Slice of Fraunhofer Diffraction Pattern')
plt.xlabel('Y Prime (mm)')
plt.ylabel('Intensity')


# Create a new figure
plt.figure(figsize=(10, 10))

# Plot the diffraction
plt.imshow(absFreqSpace, cmap='binary', extent=extentInvs_mm)
# Invert the y-axis to compensate for Array direction
plt.gca().invert_yaxis()
plt.xlim(-xPrimeLim,xPrimeLim)
plt.ylim(-yPrimeLim,yPrimeLim)
# Add a colorbar
plt.colorbar()

# Add title and labels
plt.title('1mm lens')
plt.xlabel('Xd-axis')
plt.ylabel('Yd-axis')

# Show the plot
plt.show()

