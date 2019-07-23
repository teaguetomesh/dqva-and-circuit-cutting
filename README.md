# circuit_cutting
## Getting Started

### Prerequisites

Requires a small change to the Qiskit source code.

After creating a python virtual environment and installing Qiskit,

```
python3 -m venv my_venv

pip install qiskit
pip install nxpd pydot
```

Modify the file at

```
my_venv/lib/python3.7/site-packages/qiskit/dagcircuit/dagnode.py
```

By adding the following text (not necessary anymore but keep it for now):

```
@cargs.setter
def cargs(self, new_cargs):
    """Sets the cargs to be the given list of cargs"""
    self.data_dict['cargs'] = new_cargs
```
### variables explanation
complete_path_map
```
key: qubit tuple in the original uncut circuit
value: list(tuple)
(sub circuit index, input qubit tuple in the sub circuit), 
...
(sub circuit index, input qubit tuple in the sub circuit) --> Last one is the final output measurement
```

## TODO
### Wei
#### Auto Cut Finder

- [x] Implement Karger auto cutter
- [ ] Implement fast Karger_stein auto cutter
- [x] Sort output cuts in reverse topological order

#### Cutter

 - [x] Easy interface for the uniter
 - [x] Cut the circuit into multiple parts
 - [ ] Check if cut_dag is legal
 - [x] Need to address one qubit entering a fragment more than once

#### Simulator
 - [ ] Implement simulator in Qiskit

#### Uniter
 - [ ] Implement uniter in python

### Teague
#### Circuit generator
 - [ ] Update to newest supremacy circuit generator

#### Simulator
 - [ ] Implement simulator in Intel QS

#### Uniter
 - [ ] Implement uniter in MPI

### Misc
 - [x] Is modification to Qiskit source code still needed?
 - [ ] Is Qiskit 0.11 going to crash the current codes?
 - [ ] Add a dependency installation file

## Future Directions
 - [ ] Weighted tensor network contraction, considering 'hardness' of each cluster.


