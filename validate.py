import matplotlib.pyplot as plt
import torch
import os
import matplotlib as mpl

def validate_predictions(a_to_b_model, dataset, step,
                         device="cuda" if torch.cuda.is_available() else "cpu"):
    
    a_to_b_model.load_state_dict(torch.load("model/AtoB_model.pth", map_location=device))
    a_to_b_model.to(device)
    a_to_b_model.eval()

    true_vals = []
    pred_vals = []

    with torch.no_grad():
        for spectrum, b_value, _, _, filename in dataset:
            x = spectrum.unsqueeze(0).to(device)
            if x.ndim == 2:
                x = x.unsqueeze(1)

            # Normalization
            true_z = b_value.item() * 100
            
            # Prediction
            pred_z = a_to_b_model(x).item() * 100

            true_vals.append(true_z)
            pred_vals.append(pred_z)

    true_vals_plot = true_vals[::step]
    pred_vals_plot = pred_vals[::step]

    plt.figure(figsize=(6,6))
    plt.scatter(true_vals_plot, pred_vals_plot, color="blue", marker="x", label="Prediction")
    
    # Ideal prediction
    min_val, max_val = min(true_vals+pred_vals), max(true_vals+pred_vals)
    plt.plot([min_val, max_val], [min_val, max_val], color = "red", label="Truth")

    plt.xlabel("Truth")
    plt.ylabel("Prediction")
    plt.legend()
    plt.grid(True)

    plt.show()

