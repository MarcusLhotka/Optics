"""
Fresnel Equations applied to a Prism
"""

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

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

"""
First, we determine refractive index of medium
"""
n_g = 1.711 #Glass
n_m = complex(0.185, 5.11) #metla 0.185
#print (str(n_m))
n_w = 1.328 # water
Lamda = 814e-9 #wavelength
mthicc = 45e-9 #metal THICCness
c = 299792458  # meters per second
omega = 2*np.pi*(c/Lamda)

def reflectCoeff_p(n1, n2, theta_i):
    sin_theta_t = (n1 * np.sin(theta_i)) / n2
    cosThetaI = np.cos(theta_i)
    cosThetaT = np.sqrt(1 - sin_theta_t**2)
    if sin_theta_t > 1:
        cosThetaT = 1j * np.sqrt(sin_theta_t**2 - 1)
    r = ((n2*cosThetaI-n1*cosThetaT)/(n1*cosThetaT+n2*cosThetaI))
    return r

def transCoeff_p(n1, n2, theta_i):
    sin_theta_t = (n1 * np.sin(theta_i)) / n2
    cosThetaI = np.cos(theta_i)
    cosThetaT = np.sqrt(1 - sin_theta_t**2)
    if sin_theta_t > 1:
        cosThetaT = 1j * np.sqrt(sin_theta_t**2 - 1)
    t = ((2*n1*cosThetaI)/(n1*cosThetaT+n2*cosThetaI))
    return t

def reflectCoeff_s(n1, n2, theta_i):
    sin_theta_t = (n1 * np.sin(theta_i)) / n2
    cosThetaI = np.cos(theta_i)
    cosThetaT = np.sqrt(1 - sin_theta_t**2)
    if sin_theta_t > 1:
        cosThetaT = 1j * np.sqrt(sin_theta_t**2 - 1)
    r = ((n1*cosThetaI-n2*cosThetaT)/(n1*cosThetaI+n2*cosThetaT))
    return r

def transCoeff_s(n1, n2, theta_i):
    sin_theta_t = (n1 * np.sin(theta_i)) / n2
    cosThetaI = np.cos(theta_i)
    cosThetaT = np.sqrt(1 - sin_theta_t**2)
    if sin_theta_t > 1:
        cosThetaT = 1j * np.sqrt(sin_theta_t**2 - 1)
    t = ((2*n1*cosThetaI)/(n1*cosThetaI+n2*cosThetaT))
    return t

def buildInterfaceMatrix_p (n1, n2, theta_i):
    sin_theta_t = (n1 * np.sin(theta_i)) / n2
    theta_t = np.arcsin(sin_theta_t)
    cosThetaI = np.cos(theta_i)
    cosThetaT = np.cos(theta_t)
    r12 = reflectCoeff_p(n1, n2, theta_i)
    r21 = reflectCoeff_p(n2, n1, theta_t)
    t12 = transCoeff_p(n1, n2, theta_i)
    t21 = transCoeff_p(n2, n1, theta_t)
    m= 1/t21*np.array([
        [(t12*t21-r12*r21), r21],
        [(-r12), 1]], dtype=complex)
    return m, theta_t

def buildInterfaceMatrix_s (n1, n2, theta_i):
    sin_theta_t = (n1 * np.sin(theta_i)) / n2
    theta_t = np.arcsin(sin_theta_t)
    cosThetaI = np.cos(theta_i)
    cosThetaT = np.cos(theta_t)
    r12 = reflectCoeff_s(n1, n2, theta_i)
    r21 = reflectCoeff_s(n2, n1, theta_t)
    t12 = transCoeff_s(n1, n2, theta_i)
    t21 = transCoeff_s(n2, n1, theta_t)
    m= ((1/t21)*np.array([
        [(t12*t21-r12*r21), r21],
        [(-r12), 1]], dtype=complex))
    return m, theta_t

def buildPropagationMatrix(n1, d1, theta_i, w):
    """
    Function that builds the propagation matrix for losses while the field is moving through a medium
    n1 = material index
    d1 = thiccness of the copy=
    theta_i = angle while entering the medium
    w = angular frequency of light
    """
    phi = (n1*(w/c)*d1*np.cos(theta_i))
    p = np.array([
        [(np.e**(complex(0,phi))), 0],
        [0, (np.e**(complex(0,-phi)))]], dtype=complex)
    return p

# Creates array of theta_i axis values
theta_range = np.radians(np.linspace(0, 90, 1000))

#create interface matricies
# m12 -> p2 -> m23
interface_results = [buildInterfaceMatrix_s(n_g, n_m, theta) for theta in theta_range]
m12 = np.array([result[0] for result in interface_results])
theta2 = np.array([result[1] for result in interface_results])
p2 = np.array([buildPropagationMatrix(n_m, mthicc, theta, omega) for theta in theta2])
m23 = np.array([buildInterfaceMatrix_s(n_m, n_w, theta)[0] for theta in theta2])
mptot = np.array([m23[i] @ p2[i] @ m12[i] for i in range(len(m12))])
print(mptot.shape)
rptot = np.array([np.abs(m[1, 0] / m[1, 1])**2 for m in mptot])
print(rptot.shape)
print(np.isnan(rptot).any(), np.isinf(rptot).any())

interface_results_p = [buildInterfaceMatrix_p(n_g, n_m, theta) for theta in theta_range]
m12_p = np.array([result[0] for result in interface_results_p])
theta2_p = np.array([result[1] for result in interface_results_p])
p2_p = np.array([buildPropagationMatrix(n_m, mthicc, theta, omega) for theta in theta2_p])
m23_p = np.array([buildInterfaceMatrix_p(n_m, n_w, theta)[0] for theta in theta2_p])
mptot_p = np.array([m23_p[i] @ p2_p[i] @ m12_p[i] for i in range(len(m12))])
rptot_p = np.array([np.abs(m[1, 0] / m[1, 1])**2 for m in mptot_p])
print(np.isnan(rptot_p).any(), np.isinf(rptot_p).any())

plt.figure(figsize=(8, 6))
plt.plot(np.degrees(theta_range), rptot, label='Reflectance')
plt.plot(np.degrees(theta_range), rptot_p, label='Reflectance')
plt.xlabel('Incident Angle (degrees)')
plt.ylabel('Reflectance')
plt.ylim(0, 2)
plt.title('Reflectance vs Incident Angle')
plt.grid(True)
plt.legend()
plt.show()