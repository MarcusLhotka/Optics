"""
Fresnel Equations applied to a Prism
"""

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

"""
First, we determine refractive index of medium
"""
n_g = 1.711 #Glass
#n_m = complex(0.185, 5.11) #metla 0.185
#print (str(n_m))
n_w = 1.328 # water
Lamda = 814e-9 #wavelength
mthicc = 45e-9 #metal THICCness
c = 299792458  # meters per second
omega = 2*np.pi*(c/Lamda)



def update_plot(val):
    global n_g, n_m, n_w, Lamda, mthicc, omega
    n_g = s_n_g.val
    n_w = s_n_w.val
    Lamda = s_Lamda.val * 1e-9  # Convert nm to m
    mthicc = s_mthicc.val * 1e-9  # Convert nm to m
    omega = 2 * np.pi * (c / Lamda)
    
    # Calculate n_m using the Drude model
    epsilon_inf = s_epsilon_inf.val
    omega_p = s_omega_p.val * 1.602e-19 / 1.0545e-34  # Convert eV to rad/s
    gamma = s_gamma.val * 1.602e-19 / 1.0545e-34  # Convert eV to rad/s
    epsilon = ff.drude_model(omega, epsilon_inf, omega_p, gamma)
    n_m = np.sqrt(epsilon)
    
    theta_range = np.radians(np.linspace(0, 90, 1000))
    
    interface_results = [ff.buildInterfaceMatrix_s(n_g, n_m, theta) for theta in theta_range]
    m12 = np.array([result[0] for result in interface_results])
    theta2 = np.array([result[1] for result in interface_results])
    p2 = np.array([ff.buildPropagationMatrix(n_m, mthicc, theta, omega) for theta in theta2])
    m23 = np.array([ff.buildInterfaceMatrix_s(n_m, n_w, theta)[0] for theta in theta2])
    mptot = np.array([m23[i] @ p2[i] @ m12[i] for i in range(len(m12))])
    rptot = np.array([np.abs(m[1, 0] / m[1, 1])**2 for m in mptot])
    
    interface_results_p = [ff.buildInterfaceMatrix_p(n_g, n_m, theta) for theta in theta_range]
    m12_p = np.array([result[0] for result in interface_results_p])
    theta2_p = np.array([result[1] for result in interface_results_p])
    p2_p = np.array([ff.buildPropagationMatrix(n_m, mthicc, theta, omega) for theta in theta2_p])
    m23_p = np.array([ff.buildInterfaceMatrix_p(n_m, n_w, theta)[0] for theta in theta2_p])
    mptot_p = np.array([m23_p[i] @ p2_p[i] @ m12_p[i] for i in range(len(m12_p))])
    rptot_p = np.array([np.abs(m[1, 0] / m[1, 1])**2 for m in mptot_p])


    # Update text boxes
    tb_n_g.set_val(f"{s_n_g.val:.4g}")
    tb_epsilon_inf.set_val(f"{s_epsilon_inf.val:.4g}")
    tb_omega_p.set_val(f"{s_omega_p.val:.4g}")
    tb_gamma.set_val(f"{s_gamma.val:.4g}")
    tb_n_w.set_val(f"{s_n_w.val:.4g}")
    tb_Lamda.set_val(f"{s_Lamda.val:.4g}")
    tb_mthicc.set_val(f"{s_mthicc.val:.4g}")

    ax.clear()
    ax.plot(np.degrees(theta_range), rptot, label='s-polarization')
    ax.plot(np.degrees(theta_range), rptot_p, label='p-polarization')
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
    
    fig.canvas.draw_idle()

def find_best_thickness(event):
    epsilon_inf = s_epsilon_inf.val
    omega_p = s_omega_p.val * 1.602e-19 / 1.0545e-34
    gamma = s_gamma.val * 1.602e-19 / 1.0545e-34
    epsilon = ff.drude_model(omega, epsilon_inf, omega_p, gamma)
    
    best_thickness, best_angle, min_reflectance = ff.find_zero_reflectance_thickness_and_angle(n_g, n_w, Lamda, epsilon)
    
    print(f"Best thickness: {best_thickness:.2f} nm")
    print(f"Best angle: {np.degrees(best_angle):.2f} degrees")
    print(f"Minimum reflectance: {min_reflectance:.4e}")
    
    # Update the metal thickness slider and textbox
    s_mthicc.set_val(best_thickness)
    tb_mthicc.set_val(f"{best_thickness:.4g}")


class CustomSlider(Slider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.valtext.set_visible(False)

def update_slider(text, slider):
    try:
        val = float(text)
        slider.set_val(val)
    except ValueError:
        pass

def create_slider_with_textbox(ax, label, valmin, valmax, valinit):
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
slider_ax_n_w = plt.axes([0.12, 0.11, 0.65, 0.03], facecolor=slider_color)
slider_ax_Lamda = plt.axes([0.12, 0.06, 0.65, 0.03], facecolor=slider_color)
slider_ax_mthicc = plt.axes([0.12, 0.01, 0.65, 0.03], facecolor=slider_color)

s_n_g, tb_n_g = create_slider_with_textbox(slider_ax_n_g, 'n_g', 1.0, 2.5, n_g)
s_epsilon_inf, tb_epsilon_inf = create_slider_with_textbox(slider_ax_epsilon_inf, '$\epsilon_\infty$', 1, 10, 5.2)
s_omega_p, tb_omega_p = create_slider_with_textbox(slider_ax_omega_p, '$\omega_p$ (eV)', 5, 15, 9)
s_gamma, tb_gamma = create_slider_with_textbox(slider_ax_gamma, '$\gamma$ (eV)', 0.01, 0.1, 0.068)
s_n_w, tb_n_w = create_slider_with_textbox(slider_ax_n_w, 'n_w', 1.0, 2.0, n_w)
s_Lamda, tb_Lamda = create_slider_with_textbox(slider_ax_Lamda, '$\lambda$ (nm)', 400, 1550, Lamda*1e9)
s_mthicc, tb_mthicc = create_slider_with_textbox(slider_ax_mthicc, 'Metal Thickness (nm)', 1, 100, mthicc*1e9)

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
s_n_w.on_changed(update_plot)
s_Lamda.on_changed(update_plot)
s_mthicc.on_changed(update_plot)



# Initial plot
update_plot(None)

plt.show()