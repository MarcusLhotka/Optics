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
n_m = 0.185 #complex(0.185, 5.11) #metla
#print (str(n_m))
n_w = 1.328 # water
Lamda = 814 #wavelength
mthicc = 45 #metal THICCness

def alphaAndBeta(theta_i):
    """
    Function that returns alpha and beta (as a function of the angle of incidence)
    """
    alpha = ((np.sqrt(1-((n_g/n_m)*np.sin(theta_i))**2))/(np.cos(theta_i)))
    beta = n_m/n_g
    return alpha, beta

def reflectionAndTransmissionCoefficient(theta_i):
    """
    Function returns reflection and transmission coefficients
    """
    alpha = alphaAndBeta(theta_i)[0]
    beta = alphaAndBeta(theta_i)[1]
    reflectionCoefficient = ((alpha-beta)/(alpha+beta))
    transmissionCoefficient = (2/(alpha+beta))
    return reflectionCoefficient, transmissionCoefficient

def reflectanceAndTransmittance(theta_i):
    """
    Function returns transmittance and reflectance of the system
    """
    alpha = alphaAndBeta(theta_i)[0]
    beta = alphaAndBeta(theta_i)[1]
    reflectance = (((alpha-beta)/(alpha+beta))**2)
    transmittance = (alpha*beta*((2/(alpha+beta))**2))
    return reflectance, transmittance

# Creates array of theta_i axis values
theta_range = np.radians(np.linspace(0, 100, 100))

#create an array of transmission & reflection coefficient values for each theta_i in theta_range
reflectionCoefficient = np.array([reflectionAndTransmissionCoefficient(theta)[0] for theta in theta_range])
transmissionCoefficient = np.array([reflectionAndTransmissionCoefficient(theta)[0] for theta in theta_range])

#Create an array of transmittance & reflectance values for each theta_i in theta range
reflectance = np.array([reflectanceAndTransmittance(theta)[0] for theta in theta_range])
transmittance = np.array([reflectanceAndTransmittance(theta)[1] for theta in theta_range])

#Brewster's angle
theta_brewster = np.degrees(np.arctan(n_m/n_g))

#PLOT SETTINGS
# Tick for x-axis and y-axis
ticks_x = np.setdiff1d(np.append(np.arange(0, 90, 10),
                       np.round([theta_brewster], 3)), [50, 60])
ticks_y = np.arange(-0.4, 1, 0.2)

'''
Plot of Reflection and Transmission Coefficients
'''
figure, (graph_1, graph_2) = plt.subplots(1, 2, figsize=(12, 6))
graph_1.plot(np.degrees(theta_range), reflectionCoefficient,
             label=r'Reflection Coefficient $\frac{E_{0_R}}{E_{0_I}}$', linewidth=2, color='blue')
graph_1.plot(np.degrees(theta_range), transmissionCoefficient,
             label=r'Transmission Coefficient $\frac{E_{0_T}}{E_{0_I}}$', linewidth=2, color='orange')
graph_1.set_xlabel(r'Angle of Incidence $\theta_I$')
graph_1.set_ylabel('Reflection/Transmission Coefficients')
graph_1.set_yticks(ticks=ticks_y)
graph_1.set_title(r'''Reflection ($r=\frac{E_{0_R}}{E_{0_I}}$) and
Transmission ($t=\frac{E_{0_T}}{E_{0_I}}$) Coefficients as a function of Angle of Incidence ($\theta_i$)''', pad=20)
graph_1.axis([0, 90, -0.4, 1.0])
graph_1.tick_params(top=True, right=True, direction="in", length=7, width=0.9)
graph_1.legend(loc="upper right", bbox_to_anchor=(0.6, 0.8))


'''
Transmittance & Reflectance Plot
'''
graph_2.plot(np.degrees(theta_range), reflectance,
             label=r'Reflectance $R$', linewidth=2, color='blue')
graph_2.plot(np.degrees(theta_range), transmittance,
             label=r'Transmittance $T$', linewidth=2, color='orange')
graph_2.set_xlabel(r'Angle of Incidence $\theta_I$')
graph_2.set_ylabel('Reflectance/Transmittance')
graph_2.set_yticks(ticks=ticks_y)
graph_2.set_title(
    r'Reflectance and Transmittance as a function of Angle of Incidence ($\theta_i$)', pad=20)
graph_2.axis([0, 90, -0.05, 1.05])
graph_2.tick_params(top=True, right=True, direction="in", length=7, width=0.9)
graph_2.legend(loc="upper right", bbox_to_anchor=(0.42, 0.9))

# General Properties of Graph 1 & 2
for i in range(2):
    locals()[f"graph_{i+1}"].grid()
    locals()[f"graph_{i+1}"].annotate(r"Brewster's Angle $\theta_B$",
                                      xy=(np.degrees(np.arctan(n_m/n_g)), 0.0), xytext=(33, 0.2),
                                      arrowprops={"arrowstyle": "-|>", "color": "black"})
    locals()[f"graph_{i+1}"].set_xticks(ticks_x,
                                        [r"${angle}^\circ$".format(angle=angle) for angle in ticks_x])
    locals()[f"graph_{i+1}"].axhline(0.0,
                                     linewidth=1, color='green', linestyle='--')
    locals()[f"graph_{i+1}"].axvline(np.degrees(np.arctan(n_m/n_g)),
                                     linewidth=1, color='green', linestyle='--')

# Axis thickness
for axis in ['top', 'bottom', 'left', 'right']:
    graph_1.spines[axis].set_linewidth(0.9)
    graph_2.spines[axis].set_linewidth(0.9)

# Subplots settings
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.95,
                    top=0.84,
                    wspace=0.2,
                    hspace=0.1)

plt.show()
# Save plot to png file.
figure.savefig("fresnel_equation_plot_python.png", bbox_inches='tight')