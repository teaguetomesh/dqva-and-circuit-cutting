# Google quantum supremacy circuit generator
```
python supremacy_generator.py -h
```
for help information

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
(sub circuit index, input qubit tuple in the sub circuit, classical bit tuple to measure to), 
...
(sub circuit index, input qubit tuple in the sub circuit, classical bit tuple to measure to) --> Last one is the final output measurement
```
input_wires_mapping
```
key: qubit tuple in the original uncut circuit
value: (tuple) (sub circuit index, input qubit tuple in the sub circuit)
```
translation_dict
```
key: (tuple) (qubit tuple in the original uncut circuit, sub circuit index)
value: (tuple) corresponding qubit tuple in the sub circuit
```
## TODO
### Wei

 - [ ] Sort input positions in reverse topological order to prevent bad user input
 - [x] Easy interface for the uniter
 - [x] Cut the circuit into multiple parts
 - [ ] Automatic algorithm to find positions to cut
 - [x] Is modification to Qiskit source code still needed?
 - [x] Implement uniter based on sub_circ, wiring, stitches interface

### Misc
 - [ ] Add a dependency installation file
 - [ ] Current codes may crash if user input circuit has registers names containing 'measure' and 'output'
## Future Directions


 - [ ] Weighted tensor network contraction, considering 'hardness' of each cluster.
