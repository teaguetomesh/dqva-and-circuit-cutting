from typing import List, Tuple, Optional, Sequence, Dict, Any
from nptyping import NDArray
import qiskit
import numpy as np
from scipy.optimize import minimize

from qiskit_ibm_runtime import QiskitRuntimeService, Options, Sampler, Session
from quantum_serverless import QuantumServerless, get, run_qiskit_remote
from circuit_knitting_toolbox.circuit_cutting.wire_cutting import (cut_circuit_wires,
                                                                   evaluate_subcircuits,
                                                                   reconstruct_full_distribution,
                                                                   wire_cutting)


def optimize_circuit(circuit, service):
    if circuit.num_clbits == 0:
        circuit.measure_all()

    with Session(service, backend="ibmq_qasm_simulator") as session:
        sampler = Sampler(session=session)

        def f(params: List):
            # Compute the cost function
            job = sampler.run(circuit, parameter_values=params, shots=12000)
            result = job.result()
            prob_int_dist = result.quasi_dists[0].nearest_probability_distribution()
            probs = prob_int_dist.binary_probabilities(num_bits=circuit.num_qubits)

            avg_cost = 0
            for sample in probs.keys():
                x = [int(bit) for bit in list(sample)]
                # Cost function is Hamming weight
                avg_cost += probs[sample] * sum(x)

            # Return the negative of the cost for minimization
            # print('Expectation value:', avg_cost)
            return -avg_cost

        init_params = np.random.uniform(low=0.0, high=2 * np.pi, size=len(circuit.parameters))
        out = minimize(f, x0=init_params, method="COBYLA")
    return out

def local_cut_sim_reconstruct(
    circuit,
    subgraph_dict,
    cut_nodes,
    options,
    backend_names,
    service,
    num_threads,
    skip_cut = False,
    cuts = None,
):
    if not skip_cut:
        stripped_circ = wire_cutting._circuit_stripping(circuit=circuit)
        _, _, _, id_vertices = wire_cutting._read_circuit(circuit=stripped_circ)

        # The subcircuit qubit information is contained in the subgraph dict
        def parse_gate_info(gate_info):
            qubits = []
            for substr in gate_info.split('['):
                if ']' in substr:
                    qubits.append(int(substr.split(']')[0]))
            return qubits

        # Hardcoded to 2 subcircuits only
        qdca_subcircuit_2q_gates = [[], []]
        for vertex_id, gate_info in id_vertices.items():
            for qubit in parse_gate_info(gate_info):
                if qubit not in cut_nodes:
                    subcircuit_id = subgraph_dict[qubit]
                    break
            qdca_subcircuit_2q_gates[subcircuit_id].append(vertex_id)

        # LOCAL CUTTING
        cuts = cut_circuit_wires(
            circuit=circuit,
            method="manual",
            subcircuit_vertices=qdca_subcircuit_2q_gates,
        )
    else:
        if not cuts:
            raise Exception('cuts cannot be', cuts)

    # REMOTE EVALUATION via RUNTIME
    subcircuit_instance_probabilities = evaluate_subcircuits(
        cuts,
        service=service,
        backend_names=backend_names,
        options=options,
    )

    # LOCAL RECONSTRUCTION
    reconstructed_probabilities = reconstruct_full_distribution(
        circuit,
        subcircuit_instance_probabilities,
        cuts,
        num_threads=num_threads
    )

    return reconstructed_probabilities, cuts

def remote_cut_sim_reconstruct(
    circuit,
    subgraph_dict,
    cut_nodes,
    options,
    backend_names,
    serverless,
    service,
):
    stripped_circ = wire_cutting._circuit_stripping(circuit=circuit)
    _, _, _, id_vertices = wire_cutting._read_circuit(circuit=stripped_circ)

    # The subcircuit qubit information is contained in the subgraph dict
    def parse_gate_info(gate_info):
        qubits = []
        for substr in gate_info.split('['):
            if ']' in substr:
                qubits.append(int(substr.split(']')[0]))
        return qubits

    # Hardcoded to 2 subcircuits only 
    qdca_subcircuit_2q_gates = [[], []]
    for vertex_id, gate_info in id_vertices.items():
        for qubit in parse_gate_info(gate_info):
            if qubit not in cut_nodes:
                subcircuit_id = subgraph_dict[qubit]
                break
        qdca_subcircuit_2q_gates[subcircuit_id].append(vertex_id)

    # Initiate a single serverless context
    with serverless.context():
        # REMOTE CUTTING
        cuts_future = cut_circuit_wires_remote(
            circuit=circuit,
            method="manual",
            subcircuit_vertices=qdca_subcircuit_2q_gates,
        )

        cuts = get(cuts_future)

        # REMOTE EVALUATION
        service_args = service.active_account()

        options_dict = asdict(options)

        subcircuit_probabilities_future = evaluate_subcircuits_remote(
            cuts,
            service_args=service_args,
            backend_names=backend_names,
            options_dict=options_dict,
        )

        subcircuit_instance_probabilities = get(subcircuit_probabilities_future)

        # REMOTE RECONSTRUCTION
        reconstructed_probabilities_future = reconstruct_full_distribution_remote(
            circuit,
            subcircuit_instance_probabilities,
            cuts,
        )

        reconstructed_probabilities = get(reconstructed_probabilities_future)

    return reconstructed_probabilities, cuts

# Create a wrapper function to be sent to remote cluster
@run_qiskit_remote()
def cut_circuit_wires_remote(
    circuit: qiskit.QuantumCircuit,
    method: str,
    subcircuit_vertices: Optional[Sequence[Sequence[int]]] = None,
    max_subcircuit_width: Optional[int] = None,
    max_subcircuit_cuts: Optional[int] = None,
    max_subcircuit_size: Optional[int] = None,
    max_cuts: Optional[int] = None,
    num_subcircuits: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    return cut_circuit_wires(
        circuit=circuit,
        method=method,
        subcircuit_vertices=subcircuit_vertices,
        max_subcircuit_width=max_subcircuit_width,
        max_subcircuit_cuts=max_subcircuit_cuts,
        max_subcircuit_size=max_subcircuit_size,
        max_cuts=max_cuts,
        num_subcircuits=num_subcircuits,
    )

# Create a wrapper function to be sent to remote cluster
@run_qiskit_remote()
def evaluate_subcircuits_remote(
    cuts: Dict[str, Any],
    service_args: Optional[Dict[str, Any]] = None,
    backend_names: Optional[Sequence[str]] = None,
    options_dict: Optional[Dict] = None,
) -> Dict[int, Dict[int, NDArray]]:
    service = None if service_args is None else QiskitRuntimeService(**service_args)
    options = None if options_dict is None else Options(**options_dict)

    return evaluate_subcircuits(
        cuts, service=service, backend_names=backend_names, options=options
    )

@run_qiskit_remote()
def reconstruct_full_distribution_remote(
    circuit: qiskit.QuantumCircuit,
    subcircuit_instance_probabilities: Dict[int, Dict[int, NDArray]],
    cuts: Dict[str, Any],
    num_threads: int = 1,
) -> NDArray:
    return reconstruct_full_distribution(
        circuit, subcircuit_instance_probabilities, cuts
    )
