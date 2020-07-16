# circuit_cutting
A circuit cutting project

## Prerequisites

```
conda create -n cc-env python=3.7.7
conda activate cc-env
pip install numpy qiskit==0.17 matplotlib pillow pydot termcolor
conda config --add channels http://conda.anaconda.org/gurobi
conda install gurobi
conda install -c conda-forge mpich-mpicc
pip install mpi4py
```

Install Intel MKL
```
source /opt/intel/bin/compilervars.sh intel64
```

On Chai:
```
source ~/anaconda3/etc/profile.d/conda.sh
```

Local:
```
source ~/.bash_profile
```

## Commands
```
python master.py --circuit_type supremacy_grid --stage generator --size_range 6 9 --cc_size 5

python master.py --circuit_type supremacy_grid --stage evaluator --size_range 6 9 --cc_size 5 --eval_mode sv

python master.py --circuit_type supremacy_grid --stage process --size_range 6 9 --cc_size 5 --techniques 1 1 8 10 --eval_mode sv

python master.py --circuit_type supremacy_grid --stage verify --size_range 6 9 --cc_size 5 --techniques 1 1 8 10 --eval_mode sv

python master.py --circuit_type supremacy_grid --stage plot --size_range 6 9 --cc_size 5 --techniques 1 1 8 10 --eval_mode sv
```

## Qiskit paralllel test notes:
- ```max_parallel_threads``` changes the ```parallel_state_update``` metadata, but does not affect runtime
- ```max_parallel_experiments``` affects the ```parallel_experiments``` metadata, helps when there are multiple circuits
- ```max_parallel_shots``` does not change the ```parallel_shots``` parameter, does not affect runtime

## Notes
- More workers need more disk space to store reconstructed_prob

## TODO
- [x] Add time logs
- [x] Add plots
- [ ] Deep parallel
- [ ] Add `sim_percent` option for `runtime` mdoe
- [x] Add toggles for techniques
- [x] Synchronize all plot colors. Use same color for same circuit
- [x] Add evaluation backend options of `device`, `sv`, and `runtime`
- [x] Add toggle for `verify` after CC
- [x] Improve init cost from 6 to 4, adopt new `build` pattern
- [ ] Improve `horizontal_collapse` (I+Z, I-Z)
- [ ] Improve `vertical_collapse` (0+1)

## Techniques
- [ ] Reconstruct largest states? Recurse on largest bins?
- [ ] Sample, instead of reconstruct?