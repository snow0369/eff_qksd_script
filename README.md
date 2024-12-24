
---

# Reproducible Codes and Data for Numerical Experiments in [Efficient Strategies for Reducing Sampling Error in Quantum Krylov Subspace Diagonalization](https://arxiv.org/abs/2409.02504)

This repository contains the raw data and reproducible code used in the numerical experiments described in
the numerical experiments described in *[Efficient Strategies for Reducing Sampling Error in Quantum Krylov Subspace Diagonalization](https://arxiv.org/abs/2409.02504)*.
The scripts are organized to facilitate the reproduction of key results presented in the paper, specifically Table 1, Table 2, and Figure 2.

---

## Repository Contents

### Python Scripts
The following scripts may generate large buffer files in the `/buffer` directory to
avoid redundant heavy calculation.

- **`result_shifted_norm.py`**  
  Script to reproduce results in **Table 1**, presenting the shifted measurement norms.

- **`result_meas_cost.py`**  
  Script to reproduce results in **Table 2**, producing measurement costs based on empirical and theoretical variance.

- **`result_histogram.py`**  
  Script to reproduce **Figure 2**, generating a histogram for visualizing distributions of QKSD ground state
  energies using the measurement techniques introduced in this work.

### Supporting Scripts
- **`base_utils.py`**  
  Contains utility functions shared across the numerical experiments, including:
  - Preprocessing Hamiltonians
  - True spectrum estimation by direct diagonalization
  - Real-time propagator
  - Buffer paths management

- **`ics_interface.py`**  
  Provides an interface for interacting with iterative coefficient splitting (ICS).

- **`plot_tool.py`**  
  A helper script for generating plots to visualize results.

### Data
- **Shifted Norm:**
  - `result_shifted_norm.csv`: Numerical data corresponding to Table 1.

- **Measurement Cost Data:**  
  - `measurement_cost/total_result.csv`: Statistical data corresponding to Table 2.
  - `measurement_cost/*.pkl`: Raw data used to generate `total_result.csv`.

- **Histogram Data:**  
  - `histogram/H2O_histogram.png`: Image file of Figure 2.
  - `histogram/H2O_eig_data.json`: Eigenvalue error distribution used to generate `H2O_histogram.png`.

### Dependencies
- Python 3.9 is required.
- Install the required Python pacakges using `requirements.txt`:

```bash
pip install -r requirements.txt
```

- Install the [ofex](https://github.com/snow0369/ofex) pacakge and ensure your virtual environment can locate it.

---

## Reproducing the Results

**NOTE:** For the first run, if the buffers are not generated,
computations may take a significant amount of time (~2 days on a standard laptop to run all the scripts.)

### Table 1: Shifted Norm Results
1. Run the script `result_shifted_norm.py`:
   ```bash
   python3 result_shifted_norm.py
   ```
2. Results will be computed and saved to `result_shift_norm.csv`.

### Table 2: Measurement Costs
1. Run the script `result_meas_cost.py`:
   ```bash
   python3 result_meas_cost.py
   ```
2. Results will be computed and saved in `measurement_cost/` directory.

### Figure 2: Histogram of Eigenvalue Distributions
Run the script `result_histogram.py`:
   ```bash
   python3 result_histogram.py
   ```
2. Results will be computed and saved in `histogram/` directory.

---

## Citation
If you use this repository in your research, please cite the corresponding paper:

```
@misc{lee2024efficientstrategiesreducingsampling,
      title={Efficient Strategies for Reducing Sampling Error in Quantum Krylov Subspace Diagonalization}, 
      author={Gwonhak Lee and Seonghoon Choi and Joonsuk Huh and Artur F. Izmaylov},
      year={2024},
      eprint={2409.02504},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2409.02504}, 
}
```

---
