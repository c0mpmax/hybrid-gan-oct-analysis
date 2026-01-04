import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from gan import A_PATH, B_PATH
from dataset import SpectrumGenerator
from dataset import FourierDatasetDeterministic
from models import Conv1DModel

# =============================
# TRAINING FUNCTION
# =============================

def train_cycleGAN_deterministic(
    a_to_b_model,
    spectrum_generator,
    train_dataset,
    val_dataset,
    num_epochs=30,
    batch_size=8,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    a_to_b_model = a_to_b_model.to(device)
    optimizer_AtoB = optim.Adam(a_to_b_model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    history = {"train_total": [], "train_cycle": [], "val_total": [], "val_cycle": []}

    for epoch in range(1, num_epochs + 1):
        a_to_b_model.train()
        train_total_loss, train_cycle_loss = 0, 0
        
        for spectra, b_value, b_value_2d, b_spectrum, _ in train_loader:
            if spectra.ndim == 2:
                spectra = spectra.unsqueeze(1)
            spectra, b_value, b_value_2d = spectra.to(device), b_value.to(device), b_value_2d.to(device)
            b_spectrum = b_spectrum.to(device)
            
            # -------- Forward A->B --------
            pred_b = a_to_b_model(spectra)  # A -> B
            
            # -------- Deterministic B->A --------
            b_to_a_spectra = []
            for pb in pred_b.detach().cpu().numpy():
                _, recon_spectrum = spectrum_generator.generate(pb.item() * 100)
                b_to_a_spectra.append(recon_spectrum)
            recon_a = torch.tensor(np.stack(b_to_a_spectra), dtype=torch.float32).unsqueeze(1).to(device)  # + add channel
            
            # -------- Forward cycle --------
            pred_b_from_true = a_to_b_model(recon_a)
            
            # -------- Losses --------
            loss_ab = mse_loss(pred_b, b_value)           # A->B supervised
            loss_ba = l1_loss(recon_a, spectra)          # B->A supervised 
            loss_cycle = mse_loss(pred_b_from_true, b_value) + l1_loss(recon_a, spectra)
            total_loss = loss_ab + loss_ba + 0.5 * loss_cycle
            
            optimizer_AtoB.zero_grad()
            total_loss.backward()
            optimizer_AtoB.step()
            
            train_total_loss += total_loss.item()
            train_cycle_loss += loss_cycle.item()
        
        # ----------------- Validation -----------------
        a_to_b_model.eval()
        val_total_loss, val_cycle_loss = 0, 0
        with torch.no_grad():
            for spectra, b_value, b_value_2d, b_spectrum, _ in val_loader:
                if spectra.ndim == 2:
                    spectra = spectra.unsqueeze(1)
                spectra, b_value, b_value_2d = spectra.to(device), b_value.to(device), b_value_2d.to(device)
                b_spectrum = b_spectrum.to(device)
                
                pred_b = a_to_b_model(spectra)
                
                # Deterministic B->A
                b_to_a_spectra = []
                for pb in pred_b.detach().cpu().numpy():
                    _, recon_spectrum = spectrum_generator.generate(pb.item() * 100)
                    b_to_a_spectra.append(recon_spectrum)
                recon_a = torch.tensor(np.stack(b_to_a_spectra), dtype=torch.float32).unsqueeze(1).to(device)
                
                pred_b_from_true = a_to_b_model(recon_a)
                
                loss_ab = mse_loss(pred_b, b_value)
                loss_ba = l1_loss(recon_a, spectra)
                loss_cycle = mse_loss(pred_b_from_true, b_value) + l1_loss(recon_a, spectra)
                total_loss = loss_ab + loss_ba + 0.5 * loss_cycle
                
                val_total_loss += total_loss.item()
                val_cycle_loss += loss_cycle.item()
        
        train_total_loss /= len(train_loader)
        train_cycle_loss /= len(train_loader)
        val_total_loss /= len(val_loader)
        val_cycle_loss /= len(val_loader)
        
        history["train_total"].append(train_total_loss)
        history["train_cycle"].append(train_cycle_loss)
        history["val_total"].append(val_total_loss)
        history["val_cycle"].append(val_cycle_loss)
        
        print(f"Epoch {epoch}/{num_epochs} | Train Total: {train_total_loss:.4f} | Val Total: {val_total_loss:.4f}")
    
    torch.save(a_to_b_model.state_dict(), "model/AtoB_model.pth")
    print("Model saved as AtoB_model.pth")
    
    return history

spectrum_generator = SpectrumGenerator(wl_file='N-BK7.csv')

train_dataset = FourierDatasetDeterministic(A_PATH, B_PATH, spectrum_generator)
val_dataset   = FourierDatasetDeterministic(A_PATH, B_PATH, spectrum_generator)

a_to_b_model = Conv1DModel()

history = train_cycleGAN_deterministic(
    a_to_b_model=a_to_b_model,
    spectrum_generator=spectrum_generator,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    num_epochs=10,   
    batch_size=8,    
    lr=1e-3
)
