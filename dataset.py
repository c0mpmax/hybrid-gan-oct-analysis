import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

###########_DATASET-CLASS_###########
class FourierDatasetDeterministic(torch.utils.data.Dataset):
    def __init__(self, a_domain_path, b_values_path, spectrum_generator):

        self.spectrum_generator = spectrum_generator
        
        self.b_values = np.loadtxt(b_values_path) 
        
        a_files = [f for f in os.listdir(a_domain_path) if f.endswith('.txt')]
        
        def extract_number(filename):
            return float(filename.split("[")[1].split("]")[0])
        
        a_files_sorted = sorted(a_files, key=extract_number)
        self.a_paths = [os.path.join(a_domain_path, f) for f in a_files_sorted]

    def __len__(self):
        return len(self.a_paths)

    def __getitem__(self, idx):
        spectrum = np.loadtxt(self.a_paths[idx])
        spectrum = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min() + 1e-8)
        if spectrum.ndim == 1:
            spectrum = spectrum.reshape(1, -1)
        
        z_value = self.b_values[idx] / 100  # Normalization

        b_spectrum = spectrum
        
        return (
            torch.tensor(spectrum, dtype=torch.float32),       # A
            torch.tensor(z_value, dtype=torch.float32),        # B
            torch.tensor(z_value, dtype=torch.float32).view(-1, 1),  # B 2D
            torch.tensor(b_spectrum, dtype=torch.float32),    # A reconstruction (deterministic)
            os.path.basename(self.a_paths[idx]) # paths
        )


import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy import signal
import matplotlib.pyplot as plt

# --- Функции ---
def g(omega, omega_0, sigma):
    return np.exp(-0.5 * ((omega - omega_0)/sigma)**2)

def fft(x, y):
    dx = x[1] - x[0]
    f_nq = 0.5 / dx  # частота Найквиста
    n = len(x)
    f = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    s_fft = np.fft.fftshift(np.fft.fft(y))
    return f, s_fft

# --- Класс генератора спектра ---
class SpectrumGenerator:
    def __init__(self, wl_file='Fused_silica.csv', th=1e-3, noise_scale=0.8, white_scale=0.05):
        self.th = th
        self.noise_scale = noise_scale
        self.white_scale = white_scale
        self.c = 3e8  # скорость света
        
        # Данные материала
        df = pd.read_csv(wl_file, decimal=".", delimiter=',')[:101]
        wl = df['wl'].values.astype(float)
        n = df['n'].values.astype(float)
        
        # Интерполяция
        self.wl_new = np.linspace(wl.min(), wl.max(), 10 * len(wl))  # 1010 точек
        n_f = interp1d(wl, n, kind='cubic')
        self.n = n_f(self.wl_new)
        
        # Частоты и Гаусс
        omega_min = 2 * np.pi * self.c / (2000e-9)
        omega_max = 2 * np.pi * self.c / (1200e-9)
        self.omega_0 = (omega_max + omega_min) / 2
        self.sigma = (omega_max - omega_min) / 8
        
        self.omega = 2 * np.pi * self.c / (self.wl_new * 1e-6)
        self.g_vals = g(self.omega, self.omega_0, self.sigma)
        
        # Розовый шум
        N = len(self.wl_new)
        white = np.random.normal(0, 1, N) * self.white_scale
        b, a = signal.butter(1, 0.1, 'low')
        pink_noise = signal.lfilter(b, a, white)
        pink_noise /= np.std(pink_noise)
        pink_noise = pink_noise * (self.wl_new - self.wl_new.min()) / (self.wl_new.max() - self.wl_new.min())
        self.pink_noise = pink_noise
        
        # Для FFT
        self.omega_uniform = np.linspace(omega_min, omega_max, len(self.wl_new))
    
    def generate(self, z):
        s_n_z_current = np.cos((((self.n - 1) * self.th - z) * 2 * np.pi / self.wl_new)) #* self.g_vals
        s_n_z_noisy = s_n_z_current + self.noise_scale * self.pink_noise
        f_sn, s_fft_n = fft(self.omega_uniform, s_n_z_noisy)
        s_fft_n = np.abs(s_fft_n)
        s_fft_n /= (s_fft_n.max() + 1e-8)
        return f_sn, np.abs(s_fft_n)