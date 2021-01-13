# Scaling up Constrained Quantum Approximate Optimization
A set of python codes for benchmarking a new quantum algorithm for solving the Maximum Independent Set problem.

## Notes
1. Make a virtual environment and install required packages:
```
conda create -n qenv python=3.7
conda deactivate && conda activate qenv
pip install numpy qiskit matplotlib pillow pydot termcolor
conda config --add channels http://conda.anaconda.org/gurobi
conda install gurobi
```
2. Install the latest Qiskit helper functions (https://github.com/weiT1993/qiskit_helper_functions):
```
pip install .
```
