#import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
import matplotlib
from scipy.optimize import OptimizeResult, minimize

#Universal Constants
c = 299792458  # meters per second

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
    exp_phi = np.exp(1j * phi)
    p = np.array([
        [exp_phi, 0],
        [0, 1/exp_phi]
    ], dtype=complex)
    return p

def drude_model(omega, epsilon_inf, omega_p, gamma):
    """
    Calculate the complex dielectric function using the Drude model.
    
    :param omega: angular frequency of light
    :param epsilon_inf: high-frequency dielectric constant
    :param omega_p: plasma frequency
    :param gamma: damping constant
    :return: complex dielectric function
    """
    return epsilon_inf - (omega_p**2 / (omega**2 + 1j*gamma*omega))


def reflectance(mthicc, theta, n_g, n_w, Lamda, epsilon, omega):
            #print(str(mthicc[0]))
            mthicc = mthicc[0]*1e-9
            #print(str(mthicc))
            n_m = np.sqrt(epsilon)
            m12, theta_t = buildInterfaceMatrix_p(n_g, n_m, theta)
            m23, _ = buildInterfaceMatrix_p(n_m, n_w, theta_t)
            p2 = buildPropagationMatrix(n_m, mthicc, theta_t, omega)
            mptot = m23 @ p2 @ m12
            r = np.abs(mptot[1, 0] / mptot[1, 1])**2
            return r

def find_zero_reflectance_thickness_and_angle(n_g, n_w, Lamda, epsilon):
    omega = 2 * np.pi * (c / Lamda)
    result = OptimizeResult(fun=float('inf'))
    minTheta = 0
    for theta in np.radians(np.linspace(0, 90, 1000)):
        temp = minimize(lambda x: reflectance(x, theta, n_g, n_w, Lamda, epsilon, omega), 45, 
                bounds=[(35, 70)],
                method='L-BFGS-B',
                options={'maxiter': 500})
        if temp.fun < result.fun:
            result = temp
            minTheta = theta


    return result.x[0], minTheta, result.fun