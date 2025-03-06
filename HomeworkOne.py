"""
Fresnel Equations applied to a Prism

I would like to thank the AI assistant from Perplexity for providing guidance on the analysis of SPR data.
Perplexity AI. (2024). Assistance with SPR analysis.
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
Set default values for meduims and wavelength. Define universal constants.
"""
glass = ff.Medium("Glass", n=1.711)
water = ff.Medium("Water", n=1.328)
gold = ff.Medium("Gold", t=47e-9, epsilon_inf=5.2, omega_p=9, gamma=0.068)
protein = ff.Medium("Protein", n=1.45, t=400e-9)
c = 299792458  # meters per second
Lamda = 814e-9 #wavelength
omega = ff.omega(Lamda)

def update_plot(val):
    global glass, water, gold, Lamda, omega
    #set variables based on slider value
    glass.n = s_n_g.val
    water.n = s_water.val
    Lamda = s_Lamda.val * 1e-9  # Convert nm to m
    gold.t = s_gold.val * 1e-9  # Convert nm to m
    omega = ff.omega(Lamda)

    # Calculate gold.n using the Drude model
    gold.epsilon_inf = s_epsilon_inf.val
    gold.omega_p = s_omega_p.val * 1.602e-19 / 1.0545e-34  # Convert eV to rad/s
    gold.gamma = s_gamma.val * 1.602e-19 / 1.0545e-34  # Convert eV to rad/s
    epsilon = ff.drude_model(omega, gold)
    gold.n = ff.calculate_n_metal(omega, gold)

    theta_range = np.radians(np.linspace(0, 90, 1000))

    #Find s-polarization reflectance with protein
    """
    interface_results12 = [ff.buildInterfaceMatrix_s(glass.n, gold.n, theta) for theta in theta_range]
    m12 = np.array([result[0] for result in interface_results12])
    theta2 = np.array([result[1] for result in interface_results12])
    p2 = np.array([ff.buildPropagationMatrix(gold, theta, omega) for theta in theta2])
    interface_results23 = [ff.buildInterfaceMatrix_s(gold.n, protein.n, theta) for theta in theta2]
    m23 = np.array([result[0] for result in interface_results23])
    theta3 = np.array([result[1] for result in interface_results23])
    p3 = np.array([ff.buildPropagationMatrix(protein, theta, omega) for theta in theta3])
    m34 = np.array([ff.buildInterfaceMatrix_s(protein.n, water.n, theta)[0] for theta in theta3])
    mtot = np.array([m34[i] @ p3[i] @ m23[i] @ p2[i] @ m12[i] for i in range(len(m12))])
    rptot_complex = np.array([m[1, 0] / m[1, 1] for m in mtot])
    rptot_magnitude = np.abs(rptot_complex)
    rtot = rptot_magnitude**2
    rptot_phase = np.angle(rptot_complex, deg=True)
    """
    interface_results12 = [ff.buildInterfaceMatrix_s(glass.n, gold.n, theta) for theta in theta_range]
    m12 = np.array([result[0] for result in interface_results12])
    theta2 = np.array([result[1] for result in interface_results12])
    p2 = np.array([ff.buildPropagationMatrix(gold, theta, omega) for theta in theta2])
    interface_results23 = [ff.buildInterfaceMatrix_s(gold.n, water.n, theta) for theta in theta2]
    m23 = np.array([result[0] for result in interface_results23])
    mtot = np.array([m23[i] @ p2[i] @ m12[i] for i in range(len(m12))])
    rptot_complex = np.array([m[1, 0] / m[1, 1] for m in mtot])
    rptot_magnitude = np.abs(rptot_complex)
    rtot = rptot_magnitude**2
    rptot_phase = np.angle(rptot_complex, deg=True)

    #find p-polarization reflectance with protein
    """
    interface_results12_p = [ff.buildInterfaceMatrix_p(glass.n, gold.n, theta) for theta in theta_range]
    m12_p = np.array([result[0] for result in interface_results12_p])
    theta2_p = np.array([result[1] for result in interface_results12_p])
    p2_p = np.array([ff.buildPropagationMatrix(gold, theta, omega) for theta in theta2_p])
    interface_results23_p = [ff.buildInterfaceMatrix_p(gold.n, protein.n, theta) for theta in theta2_p]
    m23_p = np.array([result[0] for result in interface_results23_p])
    theta3_p = np.array([result[1] for result in interface_results23_p])
    p3_p = np.array([ff.buildPropagationMatrix(protein, theta, omega) for theta in theta3_p])
    m34_p = np.array([ff.buildInterfaceMatrix_p(protein.n, water.n, theta)[0] for theta in theta3_p])
    mtot_p = np.array([m34_p[i] @ p3_p[i] @ m23_p[i] @ p2_p[i] @ m12_p[i] for i in range(len(m12))])
    rptot_p_complex = np.array([m[1, 0] / m[1, 1] for m in mtot_p])
    rptot_p_magnitude = np.abs(rptot_p_complex)
    rtot_p = rptot_p_magnitude**2
    rptot_p_phase = np.angle(rptot_p_complex, deg=True)
    """
    interface_results12_p = [ff.buildInterfaceMatrix_p(glass.n, gold.n, theta) for theta in theta_range]
    m12_p = np.array([result[0] for result in interface_results12_p])
    theta2_p = np.array([result[1] for result in interface_results12_p])
    p2_p = np.array([ff.buildPropagationMatrix(gold, theta, omega) for theta in theta2_p])
    interface_results23_p = [ff.buildInterfaceMatrix_p(gold.n, water.n, theta) for theta in theta2_p]
    m23_p = np.array([result[0] for result in interface_results23_p])
    mtot_p = np.array([m23_p[i] @ p2_p[i] @ m12_p[i] for i in range(len(m12))])
    rptot_p_complex = np.array([m[1, 0] / m[1, 1] for m in mtot_p])
    rptot_p_magnitude = np.abs(rptot_p_complex)
    rtot_p = rptot_p_magnitude**2
    rptot_p_phase = np.angle(rptot_p_complex, deg=True)

    # Update text boxes
    tb_n_g.set_val(f"{s_n_g.val:.4g}")
    tb_epsilon_inf.set_val(f"{s_epsilon_inf.val:.4g}")
    tb_omega_p.set_val(f"{s_omega_p.val:.4g}")
    tb_gamma.set_val(f"{s_gamma.val:.4g}")
    tb_water.set_val(f"{s_water.val:.4g}")
    tb_Lamda.set_val(f"{s_Lamda.val:.4g}")
    tb_gold.set_val(f"{s_gold.val:.4g}")

    # Plot S and P reflectances against incident angle
    ax.clear()
    ax.plot(np.degrees(theta_range), rtot, label='s-polarization')
    ax.plot(np.degrees(theta_range), rtot_p, label='p-polarization')
    ax.set_xlabel('Incident Angle (degrees)')
    ax.set_ylabel('Reflectance')
    ax.set_ylim(0, 1.2)
    x_ticks = np.arange(0, 91, 5)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks, rotation=45, ha='right')
    ax.set_title('Reflectance vs Incident Angle')
    ax.grid(True)
    ax.legend()
    omega_text.set_val(f'$\omega$ = {omega:.4e} rad/s')

    """
    #Task 3 Code
    optimalTheta, r, n_w_range = ff.find_zero_reflectance_n_w_and_angle(gold.t, glass.n, gold.n, omega)
    #for theta in optimalTheta:
    #    print(str(theta))
    # Calculate the slope (RI sensitivity) no differential required as the sensitivity is linear for this case
    slope, _ = np.polyfit(n_w_range, optimalTheta, 1)
    #print(str(slope))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(n_w_range, optimalTheta)
    plt.xlabel('Refractive Index of Water (n_w)')
    plt.ylabel('Resonance Angle (degrees)')
    plt.title('SPR Angle vs Refractive Index of Water')
    plt.grid(True)

    plt.text(0.05, 0.95, f'RI Sensitivity = {slope:.2f} deg/RIU', 
         transform=plt.gca().transAxes, 
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
         verticalalignment='top')
    """

    #Task 4 Code
    # Calculate the theta SPR value for each protein thickness
    optimalTheta, r, pthick_range = ff.find_zero_reflectance_p_t_and_angle(gold.t, glass.n, gold.n, protein.n, water.n, omega)
    
    # Sample sensitivities at various points throughout the plot
    slope, _ = np.polyfit(pthick_range, optimalTheta, 1)
    slope0to75, intercept0to75 = np.polyfit(pthick_range[0:75], optimalTheta[0:75], 1)
    print(f"Slope from 0 to 75 nm: {slope0to75:.4f} degrees/nm")
    slope175to225, intercept175to225 = np.polyfit(pthick_range[175:225], optimalTheta[175:225], 1)
    print(f"Slope from 175 to 225 nm: {slope175to225:.4f} degrees/nm")
    slope500to600, intercept500to600 = np.polyfit(pthick_range[500:600], optimalTheta[500:600], 1)
    print(f"Slope from 500 to 600 nm: {slope500to600:.4f} degrees/nm")

    #Smooth to remove artifacts from calculations
    optimalTheta_smooth = savgol_filter(optimalTheta, window_length=51, polyorder=3)
    # Calculate sensitivity curve as sensitivity is not linear for this case
    sensitivity = np.gradient(optimalTheta_smooth, pthick_range)
    #print(len(pthick_range))
    #print(len(optimalTheta_smooth))
    #print(len(sensitivity))
    # Plotting
    plt.figure(figsize=(12, 8))
    plt.plot(pthick_range, optimalTheta_smooth, label='Data')

    # Plot slope lines
    x0to75 = np.array([0, 75])
    y0to75 = slope0to75 * x0to75 + intercept0to75
    plt.plot(x0to75, y0to75, 'r--', label=f'Slope 0-75 nm: {slope0to75:.4f} deg/nm')
    plt.text(75, y0to75[1], f'Slope: {slope0to75:.4f} deg/nm', color='r', verticalalignment='bottom')

    x175to225 = np.array([175, 225])
    y175to225 = slope175to225 * x175to225 + intercept175to225
    plt.plot(x175to225, y175to225, 'g--', label=f'Slope 175-225 nm: {slope175to225:.4f} deg/nm')
    plt.text(225, y175to225[1], f'Slope: {slope175to225:.4f} deg/nm', color='g', verticalalignment='bottom')

    x500to600 = np.array([500, 600])
    y500to600 = slope500to600 * x500to600 + intercept500to600
    plt.plot(x500to600, y500to600, 'b--', label=f'Slope 500-600 nm: {slope500to600:.4f} deg/nm')
    plt.text(600, y500to600[1], f'Slope: {slope500to600:.4f} deg/nm', color='b', verticalalignment='bottom')

    plt.xlabel('Film Thickness (nm)')
    plt.ylabel('Resonance Angle (degrees)')
    plt.title('Resonance Angle vs. Film Thickness')
    plt.grid(True)

    # Plot sensitivity vs. film thickness
    plt.figure(figsize=(10, 6))
    plt.plot(pthick_range, sensitivity)  # Convert to deg/nm
    plt.xlabel('Film Thickness (nm)')
    plt.ylabel('Sensitivity (deg/nm)')
    plt.title('Sensitivity vs. Film Thickness')
    plt.grid(True)

    fig.canvas.draw_idle()

def find_best_thickness(event):
    """
    Function activated by find best thickness button will 
    update parameters to minimize the refelctance at the SPR angle
    """
    global omega, gold, glass, water, lamda
    if gold.n is NULL: gold.n = ff.calculate_n_metal(gold)
    best_thickness, best_angle, min_reflectance = ff.find_zero_reflectance_thickness_and_angle(glass.n, water.n, omega, gold.n)
    
    print(f"Best thickness: {best_thickness:.2f} nm")
    print(f"Best angle: {np.degrees(best_angle):.2f} degrees")
    print(f"Minimum reflectance: {min_reflectance:.4e}")
    
    # Update the metal thickness slider and textbox
    s_gold.set_val(best_thickness)
    tb_gold.set_val(f"{best_thickness:.4g}")
    gold.t = s_gold.val * 1e-9

class CustomSlider(Slider):
    """
    Slider class for style
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.valtext.set_visible(False)

def update_slider(text, slider):
    """
    Function to update the value of the slider if 
    the matched text box is updated
    """
    try:
        val = float(text)
        slider.set_val(val)
    except ValueError:
        pass

def create_slider_with_textbox(ax, label, valmin, valmax, valinit):
    """
    Function to generate a new slider/textbox pair
    """
    slider = CustomSlider(ax, label, valmin, valmax, valinit=valinit)
    textbox = TextBox(plt.axes([ax.get_position().x1 + 0.02, ax.get_position().y0, 0.1, 0.03]), '')
    textbox.set_val(f"{valinit:.4g}")
    textbox.on_submit(lambda text: update_slider(text, slider))
    return slider, textbox

# Create the figure and the plot
fig, ax = plt.subplots(figsize=(12, 10))
plt.subplots_adjust(left=0.1, bottom=0.45, right=0.85, top=0.9)


# Create sliders with text boxes
slider_color = 'lightgoldenrodyellow'
slider_ax_n_g = plt.axes([0.12, 0.31, 0.65, 0.03], facecolor=slider_color)
slider_ax_epsilon_inf = plt.axes([0.12, 0.26, 0.65, 0.03], facecolor=slider_color)
slider_ax_omega_p = plt.axes([0.12, 0.21, 0.65, 0.03], facecolor=slider_color)
slider_ax_gamma = plt.axes([0.12, 0.16, 0.65, 0.03], facecolor=slider_color)
slider_ax_water = plt.axes([0.12, 0.11, 0.65, 0.03], facecolor=slider_color)
slider_ax_Lamda = plt.axes([0.12, 0.06, 0.65, 0.03], facecolor=slider_color)
slider_ax_gold = plt.axes([0.12, 0.01, 0.65, 0.03], facecolor=slider_color)

#Set initial values for sliders
s_n_g, tb_n_g = create_slider_with_textbox(slider_ax_n_g, 'n_g', 1.0, 2.5, glass.n)
s_epsilon_inf, tb_epsilon_inf = create_slider_with_textbox(slider_ax_epsilon_inf, '$\epsilon_\infty$', 1, 10, 5.2)
s_omega_p, tb_omega_p = create_slider_with_textbox(slider_ax_omega_p, '$\omega_p$ (eV)', 5, 15, 9)
s_gamma, tb_gamma = create_slider_with_textbox(slider_ax_gamma, '$\gamma$ (eV)', 0.01, 0.1, 0.068)
s_water, tb_water = create_slider_with_textbox(slider_ax_water, 'water.n', 1.0, 2.0, water.n)
s_Lamda, tb_Lamda = create_slider_with_textbox(slider_ax_Lamda, '$\lambda$ (nm)', 400, 1550, Lamda*1e9)
s_gold, tb_gold = create_slider_with_textbox(slider_ax_gold, 'Metal Thickness (nm)', 1, 100, gold.t*1e9)

# Create a TextBox for omega
omega_ax = plt.axes([0.8, 0.9, 0.15, 0.05])
# Create the TextBox with custom font size
omega_text = TextBox(omega_ax, '', initial=f'$\omega$ = {omega:.4e} rad/s')
omega_text.label.set_fontsize(18)  # Set the font size for the label
omega_text.text_disp.set_fontsize(18)  # Set the font size for the displayed text
omega_text.set_active(False)  # Make it read-only

# Create a button for finding the best thickness
button_ax = plt.axes([0.0, 0.93, 0.15, 0.075])
find_best_button = Button(button_ax, 'Find Best\nThickness')
find_best_button.on_clicked(find_best_thickness)

# Connect the sliders to the update function
s_n_g.on_changed(update_plot)
s_epsilon_inf.on_changed(update_plot)
s_omega_p.on_changed(update_plot)
s_gamma.on_changed(update_plot)
s_water.on_changed(update_plot)
s_Lamda.on_changed(update_plot)
s_gold.on_changed(update_plot)

# Initial plot
update_plot(None)

plt.show()