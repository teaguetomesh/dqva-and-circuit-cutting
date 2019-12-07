CIRCUIT_TYPE="$1"
DEVICE_NAME="$2"

if [ ! -d "./hardware/logs/" ]; then
  mkdir ./hardware/logs/
fi

python -m simulator.generator --min-qubit 5 --max-qubit 5 --max-clusters 5 --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE 2>&1 | tee -a ./simulator/logs/$CIRCUIT_TYPE\_$DEVICE_NAME\_logs.txt

mpiexec -n 4 python -m utils.evaluator --experiment-name simulator --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --shots-mode saturated --evaluation-method noisy_qasm_simulator 2>&1 | tee -a ./simulator/logs/$CIRCUIT_TYPE\_$DEVICE_NAME\_logs.txt
python -m utils.reconstructor --experiment-name simulator --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --shots-mode saturated --evaluation-method noisy_qasm_simulator 2>&1 | tee -a ./simulator/logs/$CIRCUIT_TYPE\_$DEVICE_NAME\_logs.txt

mpiexec -n 4 python -m utils.evaluator --experiment-name simulator --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --shots-mode sametotal --evaluation-method noisy_qasm_simulator 2>&1 | tee -a ./simulator/logs/$CIRCUIT_TYPE\_$DEVICE_NAME\_logs.txt
python -m utils.reconstructor --experiment-name simulator --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --shots-mode sametotal --evaluation-method noisy_qasm_simulator 2>&1 | tee -a ./simulator/logs/$CIRCUIT_TYPE\_$DEVICE_NAME\_logs.txt

python -m simulator.plot --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --evaluation-method noisy_qasm_simulator 2>&1 | tee -a ./simulator/logs/$CIRCUIT_TYPE\_$DEVICE_NAME\_logs.txt

# python -m utils.check_output --experiment-name simulator --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --shots-mode saturated --evaluation-method noisy_qasm_simulator