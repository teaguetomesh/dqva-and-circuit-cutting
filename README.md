# circuit_cutting
A circuit cutting project

## Prerequisites

```
conda create -n venv python=3.7
conda activate venv
pip install numpy qiskit matplotlib nxpd pydot pillow progressbar2 jupyterlab
conda install -c conda-forge mpich-mpicc
pip install mpi4py
conda install gurobi
```

## TODO
- [x] Update to be compatible with Qiskit 0.14
- [ ] Report IBMQ.load_account() warning
- [ ] Report Qiskit incompatibility with Python 3.8