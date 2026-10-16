# 🎨 Variational Autoencoder (VAE) for Image Generation on MNIST

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?logo=tensorflow)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

An implementation of a **Variational Autoencoder (VAE)** built with **TensorFlow / Keras** to synthesize handwritten digits from the MNIST dataset. The network maps high-dimensional image vectors into a smooth, continuous 2D Gaussian latent space using the reparameterization trick.

---

## 📌 Problem Statement
**Question:** Implement Variational Autoencoders (VAE) using TensorFlow for image generation.

---

## ❓ Project Overview (The 5 Ws)

* **What:** A deep generative model combining probabilistic inference with neural networks. Unlike standard deterministic autoencoders that memorize discrete points, a VAE models a probability distribution over the latent space.
* **Why:** Standard autoencoders suffer from "gaps" in the latent space where the decoder outputs garbled noise. VAEs enforce a continuous, standard normal prior $\mathcal{N}(0, I)$ via Kullback-Leibler (KL) divergence, allowing smooth interpolation and novel image generation from random samples.
* **Who:** Students, researchers, and machine learning practitioners studying deep generative modeling, representation learning, and unsupervised feature extraction.
* **Where:** Runs in Google Colab (CPU or T4 GPU), Jupyter Notebooks, or locally via CLI.
* **How:** Incorporates a custom Keras training step (`train_step`) computing the Evidence Lower Bound (ELBO) loss: Binary Cross-Entropy reconstruction loss + KL Divergence regularization.

---

## ⚙️ Architecture & Pipeline

```text
 Input Image (28x28 = 784-d vector)
                │
                ▼
      ┌──────────────────┐
      │  Encoder Network │  (Dense 256 -> ReLU)
      └─────────┬────────┘
                │
        ┌───────┴───────┐
        ▼               ▼
     z_mean         z_log_var
  (2D centroid)   (2D log-variance)
        └───────┬───────┘
                │
                ▼
   ┌──────────────────────────┐
   │ Reparameterization Trick │ ◄── Sample ε ~ N(0, I)
   │  z = μ + σ ⊙ ε           │
   └────────────┬─────────────┘
                │
                ▼
      ┌──────────────────┐
      │  Decoder Network │  (Dense 256 -> Dense 784, Sigmoid)
      └─────────┬────────┘
                │
                ▼
   Reconstructed Image (784-d)
