import torch

from pathlib import Path

from models import Conv1DModel
from models import SpectrumGenerator
from dataset import FourierDatasetDeterministic
from validate import validate_predictions

a_to_b_model = Conv1DModel()

# ===== Created by data-generator.py =====
BASE_DIR = Path(__file__).resolve().parent
A_PATH = BASE_DIR / "spektr"
B_PATH = BASE_DIR / "values" / "values_of_steps.txt"
# ===== Created by data-generator.py =====

spectrum_generator = SpectrumGenerator(wl_file='N-BK7.csv')
text_dataset = FourierDatasetDeterministic(A_PATH, B_PATH, spectrum_generator)

sample = text_dataset[50]
print("Spectrum shape:", sample[0].shape)
print("B value:", sample[1])

validate_predictions(a_to_b_model, text_dataset, step=10)
