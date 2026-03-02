# Track B: Finance & QML – Updated Preparation Guide  

**Sponsor:** Quandela  

---

## Challenge Objective  

For this track, you will work on **Level 1: Future Prediction** (available via Hugging Face).  

### 🎯 Goal  

Given the dataset, your task is to:

- Predict the **next 6 rows**
- Each row contains **224 features**
- Output must therefore generate a **6 × 224 future prediction**

This is a **time-series forecasting task** using Quantum Machine Learning (QML).

---

## Required Tools & Technical Constraints  

### Primary SDK  

- **MerLin by Quandela**  
  - Installation:
    ```bash
    pip install merlinquantum
    ```

---

## Crucial Technical Constraints  

When designing your QML architecture, you must strictly respect the following limits:

### 🧪 Simulation Limits  
- Up to **20 modes**  
- Up to **10 photons**

### 🖥️ Quandela QPU Limits  
- Up to **24 modes**  
- Up to **12 photons**

🚨 **Hardware Limitation Warning:**  
The QPU **does not support amplitude encoding or state injection**.  
You must therefore design a compatible data encoding strategy (e.g., photon-number encoding, phase encoding, or other hardware-compatible approaches).
