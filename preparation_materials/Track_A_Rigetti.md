# Track A: Energy Grid Optimization - Preparation Guide

**Sponsor:** Rigetti

## Required Tools (Open Source) & Datasets

**Python Libraries:** 
*   `numpy` & `scipy` (for general scientific computation)
*   `networkx` (Crucial for graph manipulation and the classical benchmark). 

**Quantum SDKs:**
*   You may approach the quantum portion numerically using major SDKs: `Qiskit` or `Cirq`.
*   To execute on Rigetti's Ankaa-3 hardware, you **must** use `pyQuil` and the QCS SDK.

**Challenge Dataset:**
*   [South Carolina Energy Grid Dataset (Zenodo)](http://doi.org/10.5281/zenodo.18329641) - Use `read_weighted_edgelist()` from NetworkX to load the graphs.

## Study Resources (Recommended Reading)

To prepare for the advanced nature of this challenge, we have categorized the resources from foundational to cutting-edge research:

### 1. The Theory & Research Context (Advanced Approaches)
*   **Research Paper:** ["Optimization via Quantum Preconditioning"](https://arxiv.org/abs/2502.18570) by Maxime Dupont, Tina Oberoi, and Bhuvanesh Sundar. *(Note: Search ArXiv for the title if the direct PDF link is not provided in your packet).*
*   **Blog Post:** ["New quantum algorithm boosts classical optimizers"](https://medium.com/rigetti/new-quantum-algorithm-boosts-classical-optimizers-e191e28d4aff) — A high-level overview of the concepts behind the research paper.

### 2. Rigetti Specific Tutorials (Hardware/Simulation)
*   [Getting Started with QCS](https://docs.rigetti.com/qcs)
*   [pyQuil Documentation](https://pyquil-docs.rigetti.com/en/stable/) & [QCS SDK Rust/Python Docs](https://rigetti.github.io/qcs-sdk-rust/qcs_sdk.html)
*   [Rigetti Notebook: Getting Started](https://gitlab.com/rigetti/jupyterhub/forest-notebook/-/blob/main/forest-jh15/tutorials/GettingStarted.ipynb)
*   [Rigetti Notebook: MaxCut QAOA](https://gitlab.com/rigetti/jupyterhub/forest-notebook/-/blob/main/forest-jh15/tutorials/MaxCutQAOA.ipynb)

### 3. QAOA & MaxCut Fundamentals
*   [Introduction to QAOA (PennyLane)](https://pennylane.ai/qml/demos/tutorial_qaoa_intro)
*   [MaxCut with QAOA (PennyLane)](https://pennylane.ai/qml/demos/tutorial_qaoa_maxcut)
*   [QAOA Experiments (Google Cirq)](https://quantumai.google/cirq/experiments/qaoa)

### 4. Advanced QAOA Techniques (IBM)
*   [IBM QAOA Tutorial](https://quantum.cloud.ibm.com/docs/en/tutorials/quantum-approximate-optimization-algorithm)
*   [Advanced Techniques for QAOA](https://quantum.cloud.ibm.com/docs/en/tutorials/advanced-techniques-for-qaoa)
*   [Pauli Correlation Encoding for QAOA](https://quantum.cloud.ibm.com/docs/en/tutorials/pauli-correlation-encoding-for-qaoa)
