CIRCUIT_TYPE="$1"
DEVICE_NAME="$2"

python -m simulator.generator --min-qubit 2 --max-qubit 3 --max-clusters 5 --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE

# mpiexec -n 5 python evaluator_prob.py --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --shots-mode saturated --evaluation-method noisy_qasm_simulator
# python uniter_prob.py --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --shots-mode saturated --evaluation-method noisy_qasm_simulator

# mpiexec -n 5 python evaluator_prob.py --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --shots-mode sametotal --evaluation-method noisy_qasm_simulator
# python uniter_prob.py --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --shots-mode sametotal --evaluation-method noisy_qasm_simulator

# echo "$CIRCUIT_TYPE on $DEVICE_NAME DONE"

# python plot.py --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --evaluation-method noisy_qasm_simulator