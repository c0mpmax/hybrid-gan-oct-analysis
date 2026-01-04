# Hybrid Physics-GAN for OCT Signal Analysis

**Physics-constrained GAN for robust optical path estimation in OCT systems.**

This project reconstructs optical path shift Δ from noisy interferometric spectra by embedding a physical OCT model directly into a CycleGAN training loop.

<p align="center">
  <img src="pictures/Schema.png" width="600">
</p>

<p align="center">
  <em>Figure 1 - Schema for hybrid algorithm</em>
</p>

This repository is based on my Master’s thesis: 

**“Integrating Deterministic Models into GAN Pipelines for OCT Signal Analysis”**  

(https://github.com/c0mpmax/hybrid-gan-oct-analysis/blob/main/MASTERARBEIT.pdf)

## Problem:

A purely ML-based approach to spectral inversion is very sensitive to changes in the physical system and does not generalize when system parameters change (e.g. light source bandwidth, noise profile, interferometer geometry).

## Provided solution:

By embedding real physics into the training loop, the network is constrained to learn only physically valid mappings, making it robust to system changes.

## Method:

- Physics-based spectrum generation
- Deep learning (1D CNN)
- Cycle-consistency training (CycleGAN)

## Project structure:

data-generator.py     - Generates interferometric spectra

dataset.py            - PyTorch Dataset + spectrum normalization

models.py             - CNN + physical SpectrumGenerator

train.py              - Training loop

validate.py           - Evaluation & plotting

gan.py                - Paths and evaluation wrapper

spektr/               - Generated spectral files

values/               - Ground truth Δ values

model/                - Saved model weights

requirements.txt

## Launch stages:

pip install -r requirements.txt

python data-generator.py

python train.py 

python validate.py

## Results:

- Model is capable of generating complex interferometer signals with added dispersion and noise.
  
<p align="center">
  <img src="pictures/signal.PNG" width="600">
</p>

<p align="center">
  <em>Figure 2 — Simulated OCT interferometer signal with pink + white noise</em>
</p>

- Signal generator also produces the corresponding Fourier transform, as performed in OCT systems.
  
<p align="center">
  <img src="pictures/spectra.png" width="600">
</p>

<p align="center">
  <em>Figure 3 — Transformation of a simulated signal into a spectrum using the Fourier transform</em>
</p>

- Trained neural network, driven by the physical spectrum generator, is able to accurately predict values from spectral data.

<p align="center">
  <img src="pictures/prediction_results.PNG" width="600">
</p>

<p align="center">
  <em>Figure 4 — The result of training the CNN-network in a loop with a physical spectrum generator</em>
</p>

