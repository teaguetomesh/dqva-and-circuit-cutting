# circuit_cutting
A circuit cutting project

## Prerequisites

After creating a python virtual environment and installing Qiskit,

```
python3 -m venv my_venv

pip install qiskit nxpd pydot matplotlib
```
Install Gurobi solver. Change your current directory to the Gurobi <installdir> (the directory that contains the file setup.py), and issue the following command:
```
python setup.py install
```
## Supremacy circuit generator
To generate a Google quantum supremacy circuit:
```
circuit_generator(circuit_dimension, layer_order, random_order, measure)
```
@args:
```
circuit_dimension: tuple of horizontal #qubits, vertical #qubits, depth. E.g. [4,4,8]
layer_order: cycling through 0-7 indexed two qubit gates layers. Default order is [0,2,1,3,4,6,5,7]
random_order: if True, will randomize the two qubit gates layer orderings. Makes layer_order irrelevant. Default is False.
measure: if True, add classical registers and append measurements. Default is False.
```
@return:
```
a Qiskit circuit object
```
## Auto cut searcher
### Randomized Searcher
A randomized cut searcher using random graph contraction. Randomly contract and calculate the clustering hardness metric as a function of 1. K = # cuts. 2. d = Maximum qubit size of the fragments.
```
find_best_cuts(circ, hw_max_qubit,num_clusters=)
```
@args:
```
circ: a Qiskit circuit object
hw_max_qubit: upper limit of the number of qubits in each fragment circuit
num_clusters: list of numbers. Number of fragments to split into. Default and minimum is 2.
```
@return:
```
Cut positions that have the minimum cluster hardness metric across all iterations. Note that the solution does not guarantee optimality but will be a valid cut.
```
- [x] Correct cluster_K calculation
- [x] Implement auto number of cluster optimization
- [x] Graph implementation using counter does not count edge directionality properly!!! Causing wrong cluster_d calculations.
### Exhaustive Searcher
- [x] Unnecessary to repeat when the same starting contraction orders are enough to contract the graph
### Mixed Integer Quadratically Constrained Programming (MIQCP) Solver in Gurobi
A quadratic programming solver to solve the circuit clustering problem.
- [x] Implement exponential objectives as piecewise linear
- [x] Enforce weakly-connectedness constraint (turns out not needed)
- [x] Start with incumbent found by the randomized searcher. https://www.gurobi.com/documentation/8.1/refman/start.html. Task removed. Randomized searcher is too slow to find a solution.
- [x] Implement auto number of cluster optimization
- [x] Output parser (try the parsed output with cutter to verify K,d calculation)
- [x] Replace exponential objective with linear? Works for 2 clusters. Linear objective is equivalent to only minimizing number of cuts, and does not address cluster qubit requirements.
- [ ] Factor in simulation runtime in objective function. Fewer qubits simulate faster.
### Cut searcher benchmark
Benchmarking the running speed of randomized cut searcher vs MIQCP solver. MIQCP solver is both much faster and offers better cuts.
## Cutter
Cut a circuit into fragments and provide necessary information for simulator and uniter.
```
cut_circuit(circ, cuts)
```
@args:
```
circ: a Qiskit circuit object
cuts: a list of tuples (wire, 0-indexed gate) to cut the circuit.
```
@returns:
```
fragments: list of fragment Qiskit circuits
K, d: (K, d) clustering of the fragments. Used to test the correctness of cut searcher.
complete_path_map:
key: qubit tuple in the original uncut circuit
value: list(tuple)
(sub circuit index, input qubit tuple in the sub circuit), 
...
(sub circuit index, input qubit tuple in the sub circuit) --> Last one indicates the final output qubit
```
### Cutter verifier
Compares the operations on each qubit in the original uncut circuit and all the fragments. Test for correctness.

## TODO
### Wei
#### Auto Cut Finder

- [x] Implement Karger randomized cutter
- [x] Sort output cuts in reverse topological order
- [x] Finish MIQCP formulation

#### Cutter

 - [x] Easy interface for the uniter
 - [x] Cut the circuit into multiple parts
 - [ ] Check if cut_dag is legal
 - [x] Need to address one qubit entering a fragment more than once

### Teague
#### Circuit generator
 - [x] Update to newest supremacy circuit generator

#### Simulator
 - [ ] Implement simulator in Intel QS

#### Uniter
 - [ ] Implement uniter in MPI

### Misc
 - [x] Is modification to Qiskit source code still needed?
 - [x] Is Qiskit 0.11 going to crash the current codes? It won't.
 - [ ] Add a dependency installation file

## Future Directions
 - [x] Weighted tensor network contraction, considering 'hardness' of each cluster. Implemented in MIQCP solver.