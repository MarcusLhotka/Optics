"""
Fraunhofer Diffraction for Square Aperture

Marcus Lhotka
EE 5621 Physical Optics
Homework Assignment 5
"""

"""
TODO:
Calculate Nyquist Rate to find ideal sampling rate
"""
import numpy as np
import numpy.fft as ft
import matplotlib.pyplot as plt

# Parameters
wavelength = 1e-6  # 1 micrometer (m)
aperture_width = 1e-3  # 1 mm (m)
propagation_distance = 700  # 1 km (m)
k = ((2 * np.pi) / wavelength)  # Wavenumber

# Array
arraysize = 2048  # Array size
padding_ratio = 41  # 8:1

# Real Space
spacialArraySize = aperture_width * padding_ratio
dx = spacialArraySize / arraysize
realSpace = np.zeros((arraysize, arraysize))
linewidth = (arraysize // padding_ratio) # Width of the aperture

realSpace[arraysize//2 - linewidth//2:arraysize//2 + linewidth//2,
          arraysize//2 - linewidth//2:arraysize//2 + linewidth//2] = 1

# Frequency Space
freqSpace = ft.fftshift(ft.fft2(realSpace))

#Making sure that taking the inverse transform yields the original aperture
testReal=np.abs(ft.ifft2(freqSpace))

#Scale and Normalize output
absFreqSpace = ((np.abs(freqSpace))**(.25))
absFreqSpace = absFreqSpace / np.max(absFreqSpace)

# Calculate Spatial Frequencies
dfx = 1 / spacialArraySize  # Frequency increment
fx = np.linspace(-arraysize/2 * dfx, (arraysize/2 - 1) * dfx, arraysize)
fy = fx.copy()
FX, FY = np.meshgrid(fx, fy)

# Fresnel transfer function
H_fresnel = np.exp(1j * k * propagation_distance) * np.exp(-1j * np.pi * wavelength * propagation_distance * (FX**2 + FY**2))
fresnelDiff= freqSpace*H_fresnel
xy_prime_fresnel=np.abs(ft.ifft2(fresnelDiff))
abs_xy_prime_fresnel=(np.abs(xy_prime_fresnel)**.25)
abs_xy_prime_fresnel=abs_xy_prime_fresnel/np.max(abs_xy_prime_fresnel)

# Fraunhofer Substitution: x' = lambda * z * fx
x_prime = wavelength * propagation_distance * fx
y_prime = wavelength * propagation_distance * fy

# Fraunhofer Space for graphing
extent = [x_prime[0], x_prime[-1], y_prime[0], y_prime[-1]]

# Create x and y axes for graphing real space
x = np.linspace(-spacialArraySize/2, spacialArraySize/2, arraysize)
y = x.copy()
#Extent
extent_mm = [x[0]*1e3, x[-1]*1e3, y[0]*1e3, y[-1]*1e3]


# 2D Fraunhofer Plot
plt.figure(figsize=(8, 8))
plt.imshow(absFreqSpace, extent=extent, cmap='gray')
plt.xlabel("x' (m)")
plt.ylabel("y' (m)")
#plt.xlim(-20,20)
#plt.ylim(-20,20)
plt.title("Fraunhofer Diffraction Pattern")
plt.colorbar(label="Normalized 4th Root of Intensity")

# 2D Fresnel Plot
plt.figure(figsize=(8, 8))
plt.imshow(abs_xy_prime_fresnel, extent=extent_mm, cmap='gray')
plt.xlabel("x' (mm)")
plt.ylabel("y' (mm)")
plt.title("Fresnel Diffraction Pattern")
plt.colorbar(label="Normalized 4th Root of Intensity")

plt.show()
