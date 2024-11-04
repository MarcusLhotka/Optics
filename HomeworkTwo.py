"""
Plotting dispersion against frequency for the 
lightline and surface plasmon polaritons
"""

#import libraries
from asyncio.windows_events import NULL
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
import matplotlib
import fresnel_functions as ff
from matplotlib.widgets import Button
from scipy import stats
from scipy.signal import savgol_filter


# Constants
c = 3e8  # Speed of light in m/s
hbar = 6.582119569e-16  # Reduced Planck's constant in eV*s
e = 1.602176634e-19  # Elementary charge in C

# Define media
gold = ff.Medium("Gold", t=47e-9, epsilon_inf=5.2, omega_p=9.0, gamma=0.068)
air = ff.Medium("Air", n=1)



# Generate frequency range
omega = np.linspace(0.1, 10, 1000) * gold.omega_p  # in terms of plasma frequency

# Calculate k_spp and k_light
k_spp_real = np.real([ff.k_spp(o, gold, air) for o in omega])
k_spp_imag = np.imag([ff.k_spp(o, gold, air) for o in omega])
k_light = omega / c

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(k_spp_real/gold.omega_p*c, omega/gold.omega_p, label='SPP (Real)')
#plt.plot(k_spp_imag/gold.omega_p*c, omega/gold.omega_p, label='SPP (Imaginary)')
plt.plot(k_light/gold.omega_p*c, omega/gold.omega_p, '--', label='Light line')

plt.xlabel('k$_{x}$c / $\omega_p$')
plt.ylabel('$\omega$ / $\omega_p$')
plt.title('SPP Dispersion Relation')
plt.legend()
plt.grid(True)
plt.ylim(0, 1.2)
plt.xlim(0, 2)

plt.show()