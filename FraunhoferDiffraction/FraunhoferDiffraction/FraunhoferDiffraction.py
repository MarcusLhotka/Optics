import numpy as np
import numpy.fft as ft
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Parameters
wavelength = 1e-6  # 1 micrometer (m)
aperture_width = 1e-3  # 1 mm (m)
propagation_distance = 1/2  # 1 km (m)
k = ((2 * np.pi) / wavelength)  # Wavenumber

def calculate_fresnel(arraysize, padding_ratio):
    # Array
    spacialArraySize = aperture_width * padding_ratio
    dx = spacialArraySize / arraysize
    realSpace = np.zeros((arraysize, arraysize))
    linewidth = (arraysize // padding_ratio) # Width of the aperture

    realSpace[arraysize//2 - linewidth//2:arraysize//2 + linewidth//2,
              arraysize//2 - linewidth//2:arraysize//2 + linewidth//2] = 1

    # Frequency Space
    freqSpace = ft.fftshift(ft.fft2(realSpace))

    # Calculate Spatial Frequencies
    dfx = 1 / spacialArraySize  # Frequency increment
    fx = np.linspace(-arraysize/2 * dfx, (arraysize/2 - 1) * dfx, arraysize)
    fy = fx.copy()
    FX, FY = np.meshgrid(fx, fy)

    # Fresnel transfer function
    H_fresnel = np.exp(1j * k * propagation_distance) * np.exp(-1j * np.pi * wavelength * propagation_distance * (FX**2 + FY**2))
    fresnelDiff = freqSpace * H_fresnel
    xy_prime_fresnel = np.abs(ft.ifft2(fresnelDiff))
    abs_xy_prime_fresnel = (np.abs(xy_prime_fresnel)**.25)
    abs_xy_prime_fresnel = abs_xy_prime_fresnel / np.max(abs_xy_prime_fresnel)

    # Extract phase information from H_fresnel
    phase_H_fresnel = np.unwrap(np.angle(H_fresnel))
    phase_slice = phase_H_fresnel[arraysize // 2, :]  # Central row

    return fx, phase_slice, abs_xy_prime_fresnel

def update_plot(val):
    arraysize = int(slider_arraysize.val)
    padding_ratio = int(slider_padding.val)
    
    fx, phase_slice, _ = calculate_fresnel(arraysize, padding_ratio)
    
    # Update the plot
    line.set_xdata(fx)
    line.set_ydata(phase_slice)
    ax.set_xlim(fx.min(), fx.max())
    ax.set_ylim(phase_slice.min(), phase_slice.max())
    fig.canvas.draw_idle()

def plot_fresnel_diffraction(event):
    arraysize = int(slider_arraysize.val)
    padding_ratio = int(slider_padding.val)
    
    _, _, abs_xy_prime_fresnel = calculate_fresnel(arraysize, padding_ratio)
    
    # Create x and y axes for graphing real space
    spacialArraySize = aperture_width * padding_ratio
    x = np.linspace(-spacialArraySize/2, spacialArraySize/2, arraysize)
    y = x.copy()
    extent_mm = [x[0]*1e3, x[-1]*1e3, y[0]*1e3, y[-1]*1e3]

    # 2D Fresnel Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(abs_xy_prime_fresnel, extent=extent_mm, cmap='gray')
    plt.xlabel("x' (mm)")
    plt.ylabel("y' (mm)")
    plt.title("Fresnel Diffraction Pattern")
    plt.colorbar(label="Normalized 4th Root of Intensity")
    plt.show()

# Create the initial plot
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.3)

# Initial values
initial_arraysize = 2048*2
initial_padding_ratio = 32

# Calculate initial plot
fx, phase_slice, _ = calculate_fresnel(initial_arraysize, initial_padding_ratio)

# Plot the initial data
line, = ax.plot(fx, phase_slice)
ax.set_xlabel("fx (/m)")
ax.set_ylabel("Phase (radians)")
ax.set_title("1D Slice of Phase of Fresnel Transfer Function")
ax.grid(True)

# Create sliders
ax_arraysize = plt.axes([0.1, 0.15, 0.65, 0.03])
ax_padding = plt.axes([0.1, 0.1, 0.65, 0.03])

slider_arraysize = Slider(ax_arraysize, 'Array Size', 1024, 8192, valinit=initial_arraysize, valstep=1024)
slider_padding = Slider(ax_padding, 'Padding Ratio', 2, 64, valinit=initial_padding_ratio, valstep=1)

# Connect sliders to the update function
slider_arraysize.on_changed(update_plot)
slider_padding.on_changed(update_plot)

# Add a button to plot the Fresnel Diffraction Pattern
ax_button = plt.axes([0.8, 0.025, 0.15, 0.04])
button = Button(ax_button, 'Plot Fresnel')
button.on_clicked(plot_fresnel_diffraction)

plt.show()
