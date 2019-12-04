EXPERIMENT_NAME="$1"
DEVICE_NAME="$2"
CIRCUIT_TYPE="$3"
EVALUATION_METHOD="$4"

python -m $EXPERIMENT_NAME.generator --device-name $DEVICE_NAME

python -m utils.evaluator --experiment-name $EXPERIMENT_NAME --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE
# python -m utils.job_submittor --experiment-name $EXPERIMENT_NAME --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --evaluation-method $EVALUATION_METHOD
# python -m utils.uniter --experiment-name $EXPERIMENT_NAME --device-name $DEVICE_NAME --evaluation-method noisy_qasm_simulator

# python -m $EXPERIMENT_NAME.plot