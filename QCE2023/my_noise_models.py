from qiskit.providers.aer.noise import pauli_error, NoiseModel, thermal_relaxation_error


def get_pauli_noise_func(p_Xerr=0.001, p_Zerr=0.001, p_Yerr=0.003):
    # Example error probabilities
    p_reset = 0.0
    p_meas = 0.0

    # QuantumError objects
    #error_reset = pauli_error([('X', p_reset), ('I', 1 - p_reset)])
    #error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])
    error_single_qubit = pauli_error([('X',p_Xerr), ('Z',p_Zerr), ('Y',p_Yerr), ('I', 1 - (p_Xerr + p_Zerr + p_Yerr))])
    error_two_qubit = error_single_qubit.tensor(error_single_qubit) # A chance of single-qubit error on each participating qubit

    # Add errors to noise model
    noise_pauli = NoiseModel()
    #noise_bit_flip.add_all_qubit_quantum_error(error_reset, "reset")
    #noise_bit_flip.add_all_qubit_quantum_error(error_meas, "measure")
    noise_pauli.add_all_qubit_quantum_error(error_single_qubit, ["u1", "u2", "u3", "rz", "sx", "rx", "ry"])
    noise_pauli.add_all_qubit_quantum_error(error_two_qubit, ["cx"])

    return noise_pauli


def get_thermal_noise_func(T1=50e3, T2=70e3):
    """
    T1 and T2 values are given in units of nanoseconds
    """

    # Instruction times (in nanoseconds)
    time_u1 = 0   # virtual gate
    time_u2 = 50  # (single X90 pulse)
    time_u3 = 100 # (two X90 pulses)
    time_cx = 300
    time_reset = 1000  # 1 microsecond
    time_measure = 1000 # 1 microsecond

    # Add errors to noise model
    noise_thermal = NoiseModel()
    #noise_thermal.add_all_qubit_quantum_error(thermal_relaxation_error(T1, T2, time_reset), "reset")
    #noise_thermal.add_all_qubit_quantum_error(thermal_relaxation_error(T1, T2, time_measure), "measure")
    noise_thermal.add_all_qubit_quantum_error(thermal_relaxation_error(T1, T2, time_u1), "u1")
    noise_thermal.add_all_qubit_quantum_error(thermal_relaxation_error(T1, T2, time_u2), "u2")
    noise_thermal.add_all_qubit_quantum_error(thermal_relaxation_error(T1, T2, time_u3), "u3")
    noise_thermal.add_all_qubit_quantum_error(thermal_relaxation_error(
        T1,
        T2,
        time_cx).expand(thermal_relaxation_error(T1, T2, time_cx)), "cx")

    return noise_thermal
