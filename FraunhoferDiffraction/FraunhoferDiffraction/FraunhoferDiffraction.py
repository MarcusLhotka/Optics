"""
Fraunhofer Diffraction for Square Aperture

Marcus Lhotka
EE 5621 Physical Optics
Homework Assignment 5
"""

import numpy as np
import numpy.fft as ft
import matplotlib.pyplot as plt

# Parameters
wavelength = 1e-6  # 1 micrometer (m)
aperture_width = 1e-3  # 1 mm (m)
propagation_distance = 1000  # 1 km (m)

# Array
N = 1024  # Array size
padding_ratio = 8  # 8:1

# Real Space
array_size = aperture_width * padding_ratio
dx = array_size / N
realSpace = np.zeros((N, N))
linewidth = (N // padding_ratio) # Width of the aperture

realSpace[N//2 - linewidth//2:N//2 + linewidth//2,
          N//2 - linewidth//2:N//2 + linewidth//2] = 1

# Frequency Space
freqSpace = ft.fftshift(ft.fft2(realSpace))
absFreqSpace = ((np.abs(freqSpace))**(.25))
absFreqSpace = absFreqSpace / np.max(absFreqSpace)

# Calculate Spatial Frequencies
dfx = 1 / array_size  # Frequency increment
fx = np.linspace(-N/2 * dfx, (N/2 - 1) * dfx, N)
fy = fx.copy()

# Fraunhofer Substitution: x' = lambda * z * fx
x_prime = wavelength * propagation_distance * fx
y_prime = wavelength * propagation_distance * fy

# Fraunhofer Space for graphing
extent = [x_prime[0], x_prime[-1], y_prime[0], y_prime[-1]]

# Create x and y axes for graphing real space
x = np.linspace(-array_size/2, array_size/2, N)
y = x.copy()
#Extent
extent_mm = [x[0]*1e3, x[-1]*1e3, y[0]*1e3, y[-1]*1e3]

# Create a new figure to show the lens
plt.figure(figsize=(8, 8))
# Plot the lens (aperture)
plt.imshow(realSpace, cmap='gray', extent=extent_mm,)  # Use 'extent' to scale axes
plt.gca().invert_yaxis() #Invert y axis because imshow defaults to y axis increasing downwards
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.title('Square Aperture')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.colorbar()

# 2D Plot
plt.figure(figsize=(8, 8))
plt.imshow(absFreqSpace, extent=extent, cmap='gray')
plt.xlabel("x' (m)")
plt.ylabel("y' (m)")
#plt.xlim(-20,20)
#plt.ylim(-20,20)
plt.title("Fraunhofer Diffraction Pattern")
plt.colorbar(label="Normalized 4th Root of Intensity")

# 1D X Slice
plt.figure(figsize=(8, 6))
plt.plot(x_prime, absFreqSpace[N//2, :])  # Slice through the center
plt.xlabel("y' (m)")
plt.ylabel("Normalized 4th Root of Intensity")
plt.title("1D Slice (x'=0) Fraunhofer Diffraction Pattern")
plt.grid(True)

# 1D Y Slice
plt.figure(figsize=(8, 6))
plt.plot(y_prime, absFreqSpace[:, N//2])  # Slice through the center
plt.xlabel("x' (m)")
plt.ylabel("Normalized 4th Root of Intensity")
plt.title("1D Slice of (y'=0) Fraunhofer Diffraction Pattern")
plt.grid(True)
plt.show()