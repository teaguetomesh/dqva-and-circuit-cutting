# circuit_cutting
A circuit cutting project

## Prerequisites

```
conda create -n venv python=3.7
conda activate venv
pip install numpy qiskit matplotlib nxpd pydot pillow progressbar2 jupyterlab
conda install -c conda-forge mpich-mpicc
pip install mpi4py
conda config --add channels http://conda.anaconda.org/gurobi
conda install gurobi
```

## Qiskit paralllel test notes:
- ```max_parallel_threads``` changes the ```parallel_state_update``` metadata, but does not affect runtime
- ```max_parallel_experiments``` affects the ```parallel_experiments``` metadata, helps when there are multiple circuits
- ```max_parallel_shots``` does not change the ```parallel_shots``` parameter, does not affect runtime

## Design choices
- Always ```assemble``` and ```run```
- APIs do not reverse default Qiskit orders. Only manually reverse when saving to output.
- Use as few default parameters as possible

## TODO
- [x] Update to be compatible with Qiskit 0.14
- [ ] Report IBMQ.load_account() warning
- [x] Edit job_submittor to use same mitigation for each cluster
- [x] Edit mitigation for robust key names
- [ ] Implement 'least_squares' mitigation (currently is pseudo inverse)
- [ ] Fix job_submittor for the new mitigation method
- [ ] Test if ```assemble``` does any transpilation