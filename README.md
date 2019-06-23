# circuit_cutting

## Getting Started

### Prerequisites

Requires a small change to the Qiskit source code.

After creating a python virtual environment and installing Qiskit,

```
python3 -m venv my_venv

pip install qiskit
```

Modify the file at

```
my_venv/lib/python3.7/site-packages/qiskit/dagcircuit/dagnode.py
```

By adding the following text:

```
@cargs.setter
def cargs(self, new_cargs):
    """Sets the cargs to be the given list of cargs"""
    self.data_dict['cargs'] = new_cargs
```
### TODO

```
1. Handle original_dag that already has >1 components
2. Easy interface for the uniter
```
### Future Directions

```
1. Weighted tensor network contraction, considering 'hardness' of each cluster.
```
