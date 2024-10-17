#import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
import matplotlib
import fresnel_functions as ff
from matplotlib.widgets import Button

rc_fonts = {
    "text.usetex": True,
    "font.size": 14,
    "mathtext.default": "regular",
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "figure.titlesize": 15,
    "text.latex.preamble": r"\usepackage{amsmath,amssymb,bm,physics,lmodern}",
    "font.family": "serif",
    "font.serif": "computer modern roman",
}
matplotlib.rcParams.update(rc_fonts)

#Values
glass = ff.Medium("Glass", n=1.711)
gold = ff.Medium("Gold", t=47, epsilon_inf=5.2, omega_p=9, gamma=0.068)
Lamda = 785e-9 #wavelength
c = 299792458  # meters per second
omega = ff.omega(Lamda)
gold.n = ff.calculate_n_metal(omega, gold)
n_w_range = np.linspace(1.3, 1.4, 1000)  # Adjust range as needed
theta_range = []
"""
For all potential values of n water we must find the thetaSPR
"""
# Calculate theta for different n_w values
theta_range, r = ff.find_zero_reflectance_n_w_and_angle(gold.t, glass.n, gold.n, omega)
for theta in theta_range:
    print(str(theta))

# Calculate the slope (RI sensitivity)
slope, _ = np.polyfit(n_w_range, theta_range, 1)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(n_w_range, theta_range)
plt.xlabel('Refractive Index of Water (n_w)')
plt.ylabel('Resonance Angle (degrees)')
plt.title('SPR Angle vs Refractive Index of Water')
plt.grid(True)

plt.text(0.05, 0.95, f'RI Sensitivity = {slope:.2f} deg/RIU', 
         transform=plt.gca().transAxes, 
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
         verticalalignment='top')

plt.show()