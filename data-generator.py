# ===== Imports =====
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy import signal
import os

# ===== Utility functions =====
def sigmoid(x):
    # Smooth step function
    return 1 / (1 + np.exp(-x))

def g(x, mu, sigma):
    # Gaussian function
    return np.exp(-(x - mu)**2 / (2 * sigma**2))

def fft(x, y):
    # Compute centered FFT
    dx = x[1] - x[0]
    f_nq = 0.5 / dx
    n = len(x)
    f = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    s_fft = np.fft.fftshift(np.fft.fft(y))
    return f, s_fft

# ===== Project directories =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SPEKTR_DIR = os.path.join(BASE_DIR, "spektr")
VALUE_DIR = os.path.join(BASE_DIR, "values")

# Create folders if they do not exist
os.makedirs(SPEKTR_DIR, exist_ok=True)
os.makedirs(VALUE_DIR, exist_ok=True)

# Use in the rest of the code
save_spektr_dir = SPEKTR_DIR
folder_path = VALUE_DIR

# ===== Delta grid =====
numeric_array = np.linspace(0, 30, 50)
plot_z = np.sort(numeric_array)
plot_z = np.array(plot_z, dtype=np.float64)
file_path = os.path.join(folder_path, "values_of_steps.txt")

print(plot_z)
print(len(plot_z))

# ===== Sigmoid test grid =====
x = np.linspace(0, 2, 10000)
z = sigmoid(100 * (x - 1))

# ===== Load refractive index data =====
df = pd.read_csv('N-BK7.csv', decimal=".", delimiter=',')[:101]
wl = df['wl'].values.astype(float)
n = df['n'].values.astype(float)

# ===== Interpolate refractive index =====
wl_new = np.linspace(wl.min(), wl.max(), 100 * len(wl))
print(wl_new)
print(len(wl_new))

n_f = interp1d(wl, n, kind='cubic')
n = n_f(wl_new)

print(n)
print(len(n))

dn_dwl = np.gradient(n, wl_new)

# ===== Plot refractive index =====
plt.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(wl_new, n, label='n')
ax.set_xlim(min(wl_new), max(wl_new))
ax.set_ylim(min(n), max(n))
ax.set_xlabel('λ, µm', fontsize=16)
ax.set_ylabel('n(λ)', fontsize=16)
ax.grid(True)
fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
fig.savefig(os.path.join(BASE_DIR, "n_NBK7.pdf"), dpi=100, bbox_inches='tight', format='pdf')
plt.show()

# ===== Physical constants =====
th = 100.0
c = 3e8
f = c / (wl_new)
omega = 2 * np.pi * f

# ===== Phase derivative analysis =====
real_dlambda = wl_new[1] - wl_new[0]
print(real_dlambda)
z_values = plot_z
dlambda_max_list = []

for deltas in z_values:
    # Compute phase derivative
    dphi_dlambda = (2 * np.pi / wl_new**2) * (-((n - 1) * th - deltas) + wl_new * th * dn_dwl)
    grad_dphi = np.gradient(dphi_dlambda, wl_new)
    max_grad = np.max(np.abs(dphi_dlambda))

    plt.rcParams['font.family'] = 'Times New Roman'
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(wl_new, dphi_dlambda, color='blue')
    ax1.set_xlim(0.3, 2.5)
    ax1.set_xlabel('λ, µm', fontsize=16)
    ax1.set_ylabel("dφ/dλ", fontsize=16)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.grid(True)
    plt.close()

    dlambda_max = np.pi / max_grad
    dlambda_max_list.append(dlambda_max)

# ===== Plot dn/dλ =====
fig2, ax2 = plt.subplots(figsize=(5, 3))
ax2.plot(wl_new, dn_dwl, color='red')
ax2.set_xlim(0.3, 2.5)
ax2.set_ylim(-0.35, 0)
ax2.set_xlabel('λ, µm', fontsize=16)
ax2.set_ylabel("dn/dλ", fontsize=16)
ax2.tick_params(axis='both', labelsize=12)
ax2.grid(True)
plt.close()

# ===== Plot delta-lambda relation =====
plt.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(6, 6))
plt.plot(z_values, dlambda_max_list, label=r'$\text{erforderliche } \delta\lambda_\mathrm{e}$', color='blue')
plt.axhline(real_dlambda, color='red', linestyle='--', label=r'$\text{angewandte } \delta\lambda_\mathrm{a}$')
plt.xlabel(r'$\Delta$, µm', fontsize=13)
plt.ylabel(r'$\delta\lambda$, µm', fontsize=13)
plt.grid(True)
plt.legend()
plt.tight_layout()
fig.savefig(os.path.join(BASE_DIR, "dlambda.pdf"), dpi=100, bbox_inches='tight', format='pdf')
plt.show()

# ===== Gaussian spectral envelope =====
omega = 2 * np.pi * c / (wl_new * 1e-6)
wl_min = 0
wl_max = 1.5
wl_new_g = np.linspace(wl_new[0], wl_new[-1], len(wl_new))

lambda_0 = (wl_max + wl_min) / 2
sigma_lambda = (wl_max - wl_min) / 8

g_vals = g(wl_new_g, lambda_0, sigma_lambda)

plt.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(6, 4))
plt.plot(wl_new_g, g_vals, label=r'$g(\lambda)$')
plt.xlabel(r'$\lambda$, μm')
plt.ylabel(r'$g(\lambda)$')
plt.title('Gaussian over λ')
plt.grid(True)
plt.legend()
plt.show()
print(len(wl_new_g))

# ===== Noise generation =====
white = np.random.normal(0, 1, len(wl_new))
white_scale = 0.1
white *= white_scale

b, a = signal.butter(1, 0.1, 'low')
pink_noise = signal.lfilter(b, a, white)
pink_noise /= np.std(pink_noise)
pink_noise = pink_noise * (wl_new - wl_new.min()) / (wl_new.max() - wl_new.min())

freq = c / (wl_new * 1e-6)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18

plt.figure(figsize=(8, 4))
plt.plot(freq, white, color='gray', alpha=1.0, label='White noise')
plt.plot(freq, pink_noise, color='red', alpha=0.8, label='Pink noise')
plt.xlabel("ω, Hz")
plt.ylabel("Amplitude")
plt.xlim(0.15e15, 1e15)
plt.ylim(-2, 2)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ===== Noise FFT =====
fft_white = np.fft.fft(white)
fft_pink = np.fft.fft(pink_noise)
freqs_fft = np.fft.fftfreq(len(wl_new), d=(freq[1] - freq[0]))

mask = freqs_fft > 0
freqs_fft = freqs_fft[mask]
fft_white_mag = np.abs(fft_white[mask])
fft_pink_mag = np.abs(fft_pink[mask])

plt.figure(figsize=(8, 4))
plt.plot(freqs_fft, fft_white_mag, color='gray', alpha=0.9, label='White FFT')
plt.plot(freqs_fft, fft_pink_mag, color='red', alpha=0.8, label='Pink FFT')
plt.xlabel("k, 1/µm")
plt.ylabel("|F(k)|")
plt.xlim(0, 7e-14)
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ===== Main spectral loop =====
s_n_z_array = []
s_total_array = []

for z in plot_z:
    # Build signal for given delta
    s_n_z_current = []
    for n_koef, wl_val, g_val in zip(n, wl_new, g_vals):
        s_n_z = np.cos((((n_koef - 1) * 1 - z) * 2 * np.pi / wl_val))
        s_n_z_current.append(s_n_z)

    s_total = np.sum(s_n_z_current)
    s_total_array.append(s_total)

    s_n_z_noisy = (s_n_z_current + (0.05 * pink_noise) + white) * g_vals
    k_vals = 2 * np.pi / wl_new
    omega_uniform = np.linspace(k_vals[0], k_vals[-1], len(wl_new))

    f_sn, s_fft_n = fft(omega_uniform, s_n_z_noisy)
    f_sn, s_fft_n_g = fft(omega_uniform, s_n_z_current)

    s_fft_n_g = np.abs(s_fft_n_g)
    s_fft_n = np.abs(s_fft_n)

    N = len(s_n_z_noisy)
    s_fft_n = s_fft_n / N

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(wl_new, s_n_z_noisy)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(f_sn, s_fft_n)
    ax.set_xlim(-4, 4)
    ax.set_title(f"Δ = {round(z)} µm", fontname='Times New Roman')
    ax.grid(True)

    plot_filename = f'1D-Spektr-[{z}].pdf'
    plot_filepath = os.path.join(save_spektr_dir, plot_filename)

    file_name = f'1D-Spektr-[{z}].txt'
    file_path = os.path.join(save_spektr_dir, file_name)

    s_n_z_array.extend(s_n_z_current)





