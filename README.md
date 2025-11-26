# FedBand: Adaptive Federated Learning Under Strict Bandwidth Constraints

Official PyTorch implementation of the paper:

**FedBand: Adaptive Federated Learning Under Strict Bandwidth Constraints**  
Taghreed Alanazi; Abdulrahman Fahim; Muntaka Ibnath; Basak Guler; Amit Roy Chowdhury; Ananthram Swami; Evangelos E. Papalexakis; Srikanth V. Krishnamurthy, ICCCN 2025

https://ieeexplore.ieee.org/document/11133779

---

## Overview

FedBand is a communication-efficient Federated Learning (FL) framework that dynamically allocates **per-client compression budgets** based on model behavior (validation loss or gradient norm).

The method enforces a **global bandwidth constraint** each round while maximizing accuracy and fairness.

This repository includes:

- CIFAR-10 images experiments  
- UTMobileNet2021 trafic experiments  
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

## Run CIFAR-10
python3 -m fedband.run_cifar10

## Run UTMobileNet2021
python3 -m fedband.run_utmobilenet --base_path /path/to/data
