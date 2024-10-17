import numpy as np
import matplotlib.pyplot as plt
import fresnel_functions as ff

# Define constants and media
glass = ff.Medium("Glass", n=1.711)
gold = ff.Medium("Gold", t=47e-9, epsilon_inf=5.2, omega_p=9, gamma=0.068)
protein = ff.Medium("Protein", n=1.45, t=0)
water = ff.Medium("Water", n=1.328)
Lamda = 814e-9  # wavelength
c = 299792458  # speed of light in m/s
omega = 2 * np.pi * (c / Lamda)

# Calculate gold.n using the Drude model
gold.n = ff.calculate_n_metal(omega, gold)

def calculate_resonance_angle(film_thickness):
    def reflectance(theta):
        theta = theta[0]
        protein.t = film_thickness
        m12, theta_t = ff.buildInterfaceMatrix_p(glass.n, gold.n, theta)
        p2 = ff.buildPropagationMatrix(gold, theta_t, omega)
        m23, theta_p = ff.buildInterfaceMatrix_p(gold.n, protein.n, theta_t)
        p3 = ff.buildPropagationMatrix(protein, theta_p, omega)
        m34, _ = ff.buildInterfaceMatrix_p(protein.n, water.n, theta_p)
        mptot = m34 @ p3 @ m23 @ p2 @ m12
        return np.abs(mptot[1, 0] / mptot[1, 1])**2

    try:
        result = ff.minimize(reflectance, np.radians(45), 
                             bounds=[(np.radians(30), np.radians(80))],
                             method='L-BFGS-B')
        return np.degrees(result.x[0])
    except:
        return np.nan

# Calculate resonance angles for different film thicknesses
thicknesses = np.linspace(0, 250e-9, 251)  # 0 to 250 nm, 251 points
resonance_angles = [calculate_resonance_angle(t) for t in thicknesses]

# Remove any NaN values
valid_indices = ~np.isnan(resonance_angles)
thicknesses = thicknesses[valid_indices]
resonance_angles = np.array(resonance_angles)[valid_indices]

# Calculate sensitivity
sensitivity = np.diff(resonance_angles) / np.diff(thicknesses)

# Plot resonance angle vs. film thickness
plt.figure(figsize=(10, 6))
plt.plot(thicknesses * 1e9, resonance_angles)
plt.xlabel('Film Thickness (nm)')
plt.ylabel('Resonance Angle (degrees)')
plt.title('Resonance Angle vs. Film Thickness')
plt.grid(True)
plt.show()

# Plot sensitivity vs. film thickness
plt.figure(figsize=(10, 6))
plt.plot(thicknesses[1:] * 1e9, sensitivity * 1e9)  # Convert to deg/nm
plt.xlabel('Film Thickness (nm)')
plt.ylabel('Sensitivity (deg/nm)')
plt.title('Sensitivity vs. Film Thickness')
plt.grid(True)
plt.show()

# Calculate average sensitivity
avg_sensitivity = np.mean(sensitivity) * 1e9  # Convert to deg/nm
print(f"Average sensitivity: {avg_sensitivity:.4f} deg/nm")