CIRCUIT_TYPE="$1"
DEVICE_NAME="$2"

python -m simulator.generator --min-qubit 5 --max-qubit 5 --max-clusters 5 --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE

mpiexec -n 4 python -m utils.evaluator --experiment-name simulator --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --shots-mode saturated --evaluation-method statevector_simulator
python -m utils.reconstructor --experiment-name simulator --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --shots-mode saturated --evaluation-method statevector_simulator

# mpiexec -n 2 python -m utils.evaluator --experiment-name simulator --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --shots-mode sametotal --evaluation-method noisy_qasm_simulator
# python -m utils.reconstructor --experiment-name simulator --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --shots-mode sametotal --evaluation-method noisy_qasm_simulator

python -m simulator.plot --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --evaluation-method statevector_simulator