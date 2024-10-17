"""
fresnel_functions.py helper file
I would like to thank the AI assistant from Perplexity for providing guidance on the analysis of SPR data.
Perplexity AI. (2024). Assistance with SPR analysis.
"""

#import libraries
from asyncio.windows_events import NULL
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
import matplotlib
from scipy.optimize import OptimizeResult, minimize
from scipy.optimize import differential_evolution

#Universal Constants
c = 299792458  # meters per second

def reflectCoeff_p(n1, n2, theta_i):
    """
    Function to calculate the p-polarization reflectance coefficient
    This functions takes in 2 refractive indicies and an incident angle
    it returns a reflectance coefficient
    """
    sin_theta_t = (n1 * np.sin(theta_i)) / n2
    sin_theta_i = np.sin(theta_i)
    cosThetaI = np.cos(theta_i)
    # Handles the case for total internal reflectance
    if sin_theta_i > 1:
        cosThetaI = 1j * np.sqrt(sin_theta_i**2 - 1)
    # Allows for complex angles
    cosThetaT = np.sqrt(1 - sin_theta_t**2)
    if sin_theta_t > 1:
        cosThetaT = 1j * np.sqrt(sin_theta_t**2 - 1)
    r = ((n2*cosThetaI-n1*cosThetaT)/(n1*cosThetaT+n2*cosThetaI))
    return r

def transCoeff_p(n1, n2, theta_i):
    """
    Function to calculate the p-polarization transmittance coefficient
    This functions takes in 2 refractive indicies and an incident angle
    it returns a transmittance coefficient
    """
    sin_theta_t = (n1 * np.sin(theta_i)) / n2
    sin_theta_i = np.sin(theta_i)
    cosThetaI = np.cos(theta_i)
    # Handles the case for total internal reflectance
    if sin_theta_i > 1:
        cosThetaI = 1j * np.sqrt(sin_theta_i**2 - 1)
    # Allows for complex angles
    cosThetaT = np.sqrt(1 - sin_theta_t**2)
    if sin_theta_t > 1:
        cosThetaT = 1j * np.sqrt(sin_theta_t**2 - 1)
    t = ((2*n1*cosThetaI)/(n1*cosThetaT+n2*cosThetaI))
    return t

def reflectCoeff_s(n1, n2, theta_i):
    """
    Function to calculate the s-polarization reflectance coefficient
    This functions takes in 2 refractive indicies and an incident angle
    it returns a reflectance coefficient
    """
    sin_theta_t = (n1 * np.sin(theta_i)) / n2
    sin_theta_i = np.sin(theta_i)
    cosThetaI = np.cos(theta_i)
    # Handles the case for total internal reflectance
    if sin_theta_i > 1:
        cosThetaI = 1j * np.sqrt(sin_theta_i**2 - 1)
    # Allows for complex angles
    cosThetaT = np.sqrt(1 - sin_theta_t**2)
    if sin_theta_t > 1:
        cosThetaT = 1j * np.sqrt(sin_theta_t**2 - 1)
    r = ((n1*cosThetaI-n2*cosThetaT)/(n1*cosThetaI+n2*cosThetaT))
    return r

def transCoeff_s(n1, n2, theta_i):
    """
    Function to calculate the s-polarization transmittance coefficient
    This functions takes in 2 refractive indicies and an incident angle
    it returns a transmittance coefficient
    """
    sin_theta_t = (n1 * np.sin(theta_i)) / n2
    sin_theta_i = np.sin(theta_i)
    cosThetaI = np.cos(theta_i)
    # Handles the case for total internal reflectance
    if sin_theta_i > 1:
        cosThetaI = 1j * np.sqrt(sin_theta_i**2 - 1)
    # Allows for complex angles
    cosThetaT = np.sqrt(1 - sin_theta_t**2)
    if sin_theta_t > 1:
        cosThetaT = 1j * np.sqrt(sin_theta_t**2 - 1)
    t = ((2*n1*cosThetaI)/(n1*cosThetaI+n2*cosThetaT))
    return t


def buildInterfaceMatrix_p (n1, n2, theta_i):
    """
    Function to create a matrix that describes the interaction of
    light at the interface of two mediums for p-polarization
    This functions takes in 2 refractive indicies and an incident angle
    it returns a matrix and a transmittance angle
    """
    if n1 is None or n2 is None or theta_i is None:
        raise ValueError("n1, n2, and theta_i must not be None")
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
    """
    Function to create a matrix that describes the interaction of
    light at the interface of two mediums for s-polarization
    This functions takes in 2 refractive indicies and an incident angle
    it returns a matrix and a transmittance angle
    """
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


def buildPropagationMatrix(medium, theta_i, w):
    """
    Function that builds the propagation matrix for losses while the field is moving through a medium
    medium: Medium object containing refractive index (n) and thickness (t)
    theta_i: angle of incidence
    w: angular frequency of light
    """
    # Calculate complex cosine
    cos_theta = np.sqrt(1 - np.sin(theta_i)**2)
    sin_theta_i = np.sin(theta_i)
    if sin_theta_i > 1:
        cos_theta = 1j * np.sqrt(sin_theta_i**2 - 1)
    
    # Calculate the phase
    k0 = w / c  # wave number in vacuum
    kz = k0 * medium.n * cos_theta
    phi = kz * medium.t
    
    # Build the propagation matrix
    exp_phi = np.exp(1j * phi)
    p = np.array([
        [exp_phi, 0],
        [0, 1/exp_phi]
    ], dtype=complex)
    
    return p
"""
TODO: create a general function that can take in any number of mediums and return a full reflectance matrix
def findRForAllTheta(mediums, omega, theta_range):
    currentTheta = theta_range
    p=[]
    m=[]
    bigM=[]
    isp=False
    ism=False
    for i in range(len(mediums)):
        if mediums[i].t != NULL and mediums[i].t > 0:
            p = np.array([buildPropagationMatrix(mediums[i].n, mediums[i].t, theta, omega) for theta in currentTheta])
            isp=True
        else: isp=False
        if i < len(mediums)-1:
            matrices = [buildInterfaceMatrix_s(mediums[i].n, mediums[i+1].n, theta) for theta in currentTheta]
            m = np.array([matrix[0] for matrix in matrices])
            currentTheta = np.array([matrix[1] for matrix in matrices])
            ism=True
        else: ism=False
        if isp and ism: bigM.append(m@p) 
        elif isp and not ism: bigM.append(p)
        else: bigM.append(m)
    mtot=np.eye(2)
    for matrices in reversed(bigM):
        mtot = np.array([matrices[i] @ mtot for i in range(len(matrices))])
    rtot = np.array([np.abs(m[1, 0] / m[1, 1])**2 for m in mtot])
    return rtot
"""

def omega(Lamda):
    # Calculates omega for a given lambda
    omega = 2*np.pi*(c/Lamda)
    return omega

def drude_model(omega, metalMed):
    """
    Calculate the complex dielectric function using the Drude model.
    
    :param omega: angular frequency of light
    :param epsilon_inf: high-frequency dielectric constant
    :param omega_p: plasma frequency
    :param gamma: damping constant
    :return: complex dielectric function
    """
    if not metalMed.is_metal():
        raise ValueError("This medium is not a metal")
    return metalMed.epsilon_inf - (metalMed.omega_p**2 / (omega**2 + 1j*metalMed.gamma*omega))

def reflectance(mthicc, theta, n_g, n_w, n_m, omega):
    """
    Calculates the reflectance for a given set of up to 3 mediums
    """
    mthicc = mthicc[0] if isinstance(mthicc, (list, np.ndarray)) else mthicc
    mthicc = mthicc * 1e-9 if mthicc > 1e-6 else mthicc
    medium=Medium("Medium", t=mthicc, n=n_m)
    theta = theta[0] if isinstance(theta, (list, np.ndarray)) else theta
    n_w = n_w[0] if isinstance(n_w, (list, np.ndarray)) else n_w

    m12, theta_t = buildInterfaceMatrix_p(n_g, n_m, theta)
    m23, _ = buildInterfaceMatrix_p(n_m, n_w, theta_t)
    p2 = buildPropagationMatrix(medium, theta_t, omega)
    mptot = m23 @ p2 @ m12
    r = np.abs(mptot[1, 0] / mptot[1, 1])**2
    return r

def reflectancewithProtein(mthicc, pthicc, theta, n_g, n_m, n_p, n_w, omega):
    """
    Calculates the reflectance for a given set of up to 4 mediums
    """
    mthicc = mthicc[0] if isinstance(mthicc, (list, np.ndarray)) else mthicc
    mthicc = mthicc * 1e-9 if mthicc > 1e-6 else mthicc
    gmedium=Medium("gMedium", t=mthicc, n=n_m)
    theta = theta[0] if isinstance(theta, (list, np.ndarray)) else theta
    pthicc = pthicc[0] if isinstance(pthicc, (list, np.ndarray)) else pthicc
    pthicc = pthicc * 1e-9 if pthicc > 1e-6 else pthicc
    pmedium=Medium("pMedium", t=pthicc, n=n_p)
    n_w = n_w[0] if isinstance(n_w, (list, np.ndarray)) else n_w

    m12, theta2 = buildInterfaceMatrix_p(n_g, n_m, theta)
    p2 = buildPropagationMatrix(gmedium, theta2, omega)
    m23, theta3 = buildInterfaceMatrix_p(n_m, n_p, theta2)
    p3 = buildPropagationMatrix(pmedium, theta3, omega)
    m34, _ = buildInterfaceMatrix_p(pmedium.n, n_w, theta3)
    mptot = m34 @ p3 @ m23 @ p2 @ m12
    r = np.abs(mptot[1, 0] / mptot[1, 1])**2
    return r

def find_zero_reflectance_n_w_and_angle(mthicc, n_g, n_m, omega):
    """
    Finds the SPR angle for each value of the refractive index of water
    """
    n_w = np.linspace(1.3, 1.4, 100)
    thetas = []
    for n in n_w:
        result = OptimizeResult(fun=float('inf'))
        temp = differential_evolution(lambda x: reflectance(mthicc, x, n_g, n, n_m, omega), 
                bounds=[(np.radians(1), np.radians(89))],
                popsize=15,
                maxiter = 1000,
                tol=1e-6)
        if temp.fun < result.fun:
            result = temp
            thetas.append(np.degrees(result.x[0]))
        else: thetas.append(np.degrees(0))
    return thetas, result.fun, n_w

def find_zero_reflectance_p_t_and_angle(mthicc, n_g, n_m, n_p, n_w, omega):
    """
    Finds the SPR angle for each value of protein thickness
    """
    pthicc = np.linspace(0, 600, 600)
    thetas = []
    for p in pthicc:
        result = OptimizeResult(fun=float('inf'))
        temp = differential_evolution(lambda x: reflectancewithProtein(mthicc, p, x, n_g, n_m, n_p, n_w, omega), 
                bounds=[(np.radians(1), np.radians(89))],
                popsize=15,
                maxiter = 1000,
                tol=1e-6)
        if temp.fun < result.fun:
            result = temp
            thetas.append(np.degrees(result.x[0]))
        else: thetas.append(np.degrees(0))
    return thetas, result.fun, pthicc

def find_zero_reflectance_thickness_and_angle(n_g, n_w, omega, n_m):
    """
    Finds the SPR angle for each value of metal thickness
    """
    result = OptimizeResult(fun=float('inf'))
    minTheta = 0
    for theta in np.radians(np.linspace(0, 90, 1000)):
        temp = minimize(lambda x: reflectance(x, theta, n_g, n_w, n_m, omega), 45, 
                bounds=[(35, 70)],
                method='L-BFGS-B',
                options={'maxiter': 500})
        if temp.fun < result.fun:
            result = temp
            minTheta = theta
    return result.x[0], minTheta, result.fun

class Medium:
    """
    Class the defines what a medium of light is
    """
    def __init__(self, name, t=0, n=None, epsilon_inf=None, omega_p=None, gamma=None):
        self.name = name
        self.t = t  # thickness
        self.n = n  # refractive index (can be None for metals)
        self.epsilon_inf = epsilon_inf  # high-frequency dielectric constant
        self.omega_p = omega_p  # plasma frequency
        self.gamma = gamma  # damping constant

    def is_metal(self):
        return all(param is not None for param in [self.epsilon_inf, self.omega_p, self.gamma])

    def __str__(self):
        if self.is_metal():
            return f"Metal({self.name}, t={self.t}, epsilon_inf={self.epsilon_inf}, omega_p={self.omega_p}, gamma={self.gamma})"
        else:
            return f"Medium({self.name}, n={self.n}, t={self.t})"

# Function to calculate n for metals using the Drude model
def calculate_n_metal(omega, metalMed):
    """
    Function for deducing if a medium is a metal
    """
    if not metalMed.is_metal():
        raise ValueError("This medium is not a metal")
    
    epsilon_inf = metalMed.epsilon_inf
    omega_p = metalMed.omega_p
    gamma = metalMed.gamma
    epsilon = drude_model(omega, metalMed)
    return np.sqrt(epsilon)