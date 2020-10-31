# CutQC
A Python package for CutQC

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
3. Install Intel MKL to use the `C` backend:
```
source /opt/intel/bin/compilervars.sh intel64
```

## Todo
- [ ] Write the cutQC package

# HPU
A hybrid processing unit
## Todo
- [ ] DRAM
- [ ] Think about how to detect probability changes without encountering the state. 1/1 --> 1/1000? 1/10 --> 10/19?