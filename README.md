# MNIST Classification using Variational Quantum Circuit (VQC)

This program performs classification on the **MNIST image dataset** using a **Variational Quantum Circuit (VQC)**. Image data is first reduced in dimensionality using **PCA (Principal Component Analysis)**, and the compressed features are encoded into the quantum circuit as input parameters. The circuit parameters are then optimized based on classification performance.

---

## Features

- **Quantum-Classical Hybrid Pipeline**:
  - Applies PCA for dimensionality reduction and encodes the data into a variational quantum circuit.

- **Optimization Based on Classification Accuracy**:
  - The circuit is trained to maximize classification performance.

---

## Main Functions

- **Dimensionality Reduction**:
  - Reduces feature dimensions using PCA prior to quantum encoding.

- **Quantum Circuit Data Encoding**:
  - Encodes compressed image data as input parameters to a VQC.

- **Benchmarking on MNIST**:
  - Demonstrates classification performance using the MNIST dataset.

---

## How to Use

```bash
python3 main.py  # Run classification (averaged over 5 runs)
