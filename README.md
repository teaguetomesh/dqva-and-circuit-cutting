# Scaling up Constrained Quantum Approximate Optimization

## Overview
The code in this repository was used to implement the algorithms and simulations presented in two separate papers. The first, [Approaches to constrained quantum approximate optimization](https://arxiv.org/pdf/2010.06660.pdf), introduced the Dynamic Quantum Variational Ansatz (DQVA) for solving [Maximum Independent Set](https://en.wikipedia.org/wiki/Maximal_independent_set) (MIS) with a fixed allocation of quantum resources. The second, [Quantum Divide and Conquer for Combinatorial Optimization and Distributed Computing](https://arxiv.org/abs/2107.07532), combined the DQVA with quantum circuit cutting techniques to scale up the MIS optimization to larger graph sizes.

### DQVA
The workhorse file in this repo is `mis.py`. Inside, a number of functions are defined which find approximate solutions to the MIS problem on a given input graph using the specified variational ansatz. The `ansatz/` directory contains the code for generating the specific quantum circuits.

There are also a few files which are used for benchmarking the performance of the different variational algorithms on a given set of input graphs. 
In particular, `MIS_benchmark.py` and `QAOA+_benchmark.py` were used to generate the plots in the [Approaches to constrained quantum approximate optimization](https://arxiv.org/pdf/2010.06660.pdf) paper.

An example of one of these benchmarks is shown below:

<img src="https://user-images.githubusercontent.com/20692050/125863827-500fb193-031f-4a30-a9f0-9040bdbe8aa7.png" width="800">


### Quantum Divide and Conquer

The two parts of this repository: *DQVA* and *circuit cutting*, are combined together to create the Quantum Divide and Conquer (QDC) algorithm. The QDC algorithm finds approximate solutions to the MIS problem on *large* input graphs. These problem graphs are large compared to the quantum circuits used to find the maximum independent sets. In other words the graphs contain many more nodes than the number of qubits needed to solve the MIS problem. The QDC algorithm makes use of the ansatz contained within `ansatz/dqv_cut_ansatz.py` and is executed via the `solve_mis_cut_dqva()` function within `mis.py`. The code that performs the quantum circuit cutting in contained within the `qsplit` directory.

A full description and analysis of the QDC algorithm can be found in our [paper](https://arxiv.org/abs/2107.07532), example figures are shown below:

<p float="left">
  <img src="https://user-images.githubusercontent.com/20692050/128074221-d76b9294-2b5e-4662-94fc-9f7788756ec4.png" width="400" />
  <img src="https://user-images.githubusercontent.com/20692050/128074311-ce3ac037-8aa3-4387-ae90-8a52a1781cb7.png" width="400" /> 
</p>

## Citations
If you use this code, please cite our papers:

    Zain H. Saleem, Teague Tomesh, Bilal Tariq, and Martin Suchara, Approaches to Constrained Quantum Approximate Optimization,
    arXiv preprint, arXiv:2010.06660 (2021).

and

    Zain H. Saleem, Teague Tomesh, Michael A. Perlin, Pranav Gokhale, Martin Suchara, Quantum Divide and Conquer for Combinatorial 
    Optimization and Distributed Computing, arXiv preprint, arXiv:2107.07532 (2021).
