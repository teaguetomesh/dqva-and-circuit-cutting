DEVICE_NAME="$1"

python -m large_on_small.generator --device-name $DEVICE_NAME

mpiexec -n 2 python -m utils.evaluator --experiment-name large_on_small --device-name $DEVICE_NAME --circuit-type bv --shots-mode saturated --evaluation-method noisy_qasm_simulator
python -m utils.reconstructor --experiment-name large_on_small --device-name $DEVICE_NAME --circuit-type bv --shots-mode saturated --evaluation-method noisy_qasm_simulator

python -m large_on_small.plot