# FedBand: Adaptive Federated Learning Under Strict Bandwidth Constraints

Official PyTorch implementation of the paper:

**FedBand: Adaptive Federated Learning Under Strict Bandwidth Constraints**  
Taghreed Al-anazi et al., ICCCN 2025

---

## Overview

FedBand is a communication-efficient Federated Learning (FL) framework that dynamically allocates **per-client compression budgets** based on model behavior (validation loss or gradient norm).

The method enforces a **global bandwidth constraint** each round while maximizing accuracy and fairness.

This repository includes:

- CIFAR-10 experiments  
- UTMobileNetTraffic2021 experiments  
- Dynamic bandwidth allocation  
- Top-k / sparse update compression  
- Cache-based compression re-use  
- Fairness & per-client metrics  

---

## Installation

```bash
git clone https://github.com/t-alanazi/FedBand.git
cd FedBand
pip install -r requirements.txt
