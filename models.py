import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.interpolate import interp1d
from scipy import signal
import matplotlib.pyplot as plt


###########_AtoB-MODEL_###########

class Conv1DModel(nn.Module):
    def __init__(self):
        super(Conv1DModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        
        dummy_input = torch.randn(1, 1, 1010)
        with torch.no_grad():
            dummy_output = self.conv2(self.relu(self.conv1(dummy_input)))
            self.flattened_size = dummy_output.view(1, -1).size(1)
        
        self.fc = nn.Linear(self.flattened_size, 1)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flat Layer
        x = self.fc(x)
        return x.squeeze(1) 
    
###########Determ-Model###########

def g(omega, omega_0, sigma):
    return np.exp(-0.5 * ((omega - omega_0)/sigma)**2)

def fft(x, y):
    dx = x[1] - x[0]
    f_nq = 0.5 / dx 
    n = len(x)
    f = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    s_fft = np.fft.fftshift(np.fft.fft(y))
    return f, s_fft

class SpectrumGenerator:
    def __init__(self, wl_file='N-BK7.csv', th=1e-3, noise_scale=0.8, white_scale=0.05):
        self.th = th
        self.noise_scale = noise_scale
        self.white_scale = white_scale
        self.c = 3e8  
        
        df = pd.read_csv(wl_file, decimal=".", delimiter=',')[:101]
        wl = df['wl'].values.astype(float)
        n = df['n'].values.astype(float)
        
        self.wl_new = np.linspace(wl.min(), wl.max(), 10 * len(wl)) 
        n_f = interp1d(wl, n, kind='cubic')
        self.n = n_f(self.wl_new)
        
        omega_min = 2 * np.pi * self.c / (2000e-9)
        omega_max = 2 * np.pi * self.c / (1200e-9)
        self.omega_0 = (omega_max + omega_min) / 2
        self.sigma = (omega_max - omega_min) / 8
        
        self.omega = 2 * np.pi * self.c / (self.wl_new * 1e-6)
        self.g_vals = g(self.omega, self.omega_0, self.sigma)
        
        N = len(self.wl_new)
        white = np.random.normal(0, 1, N) * self.white_scale
        b, a = signal.butter(1, 0.1, 'low')
        pink_noise = signal.lfilter(b, a, white)
        pink_noise /= np.std(pink_noise)
        pink_noise = pink_noise * (self.wl_new - self.wl_new.min()) / (self.wl_new.max() - self.wl_new.min())
        self.pink_noise = pink_noise
        
        self.omega_uniform = np.linspace(omega_min, omega_max, len(self.wl_new))
    
    def generate(self, z):
        s_n_z_current = np.cos((((self.n - 1) * self.th - z) * 2 * np.pi / self.wl_new)) #* self.g_vals
        s_n_z_noisy = s_n_z_current + self.noise_scale * self.pink_noise
        f_sn, s_fft_n = fft(self.omega_uniform, s_n_z_noisy)
        s_fft_n = np.abs(s_fft_n)
        s_fft_n /= (s_fft_n.max() + 1e-8)
        return f_sn, np.abs(s_fft_n)