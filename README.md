# Scaling up Constrained Quantum Approximate Optimization
A set of python codes for benchmarking a new quantum algorithm for solving the Maximum Independent Set problem.

## Getting started
Make a virtual environment and install required packages:
```
conda create -n qenv python=3.7
conda deactivate && conda activate qenv
pip install numpy qiskit matplotlib pillow pydot
```

The different available ansatzes are found in the `ansatz/` folder. Note that `dqv_cut_ansatz.py` is currently unfinished. The `DQVA_and_QSPLIT.ipynb` notebook contains the latest functions to produce the `dqv_cut_ansatz` and simulate its execution with circuit cutting.
