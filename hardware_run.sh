CIRCUIT_TYPE="$1"
DEVICE_NAME="$2"

python -m hardware.generator --min-qubit 5 --max-qubit 5 --max-clusters 5 --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE

mpiexec -n 4 python -m utils.evaluator --experiment-name hardware --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --shots-mode saturated --evaluation-method hardware
python -m hardware.job_submittor --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --shots-mode saturated
python -m utils.reconstructor --experiment-name hardware --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --shots-mode saturated --evaluation-method hardware

mpiexec -n 4 python -m utils.evaluator --experiment-name hardware --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --shots-mode sametotal --evaluation-method hardware
python -m hardware.job_submittor --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --shots-mode sametotal
python -m utils.reconstructor --experiment-name hardware --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --shots-mode sametotal --evaluation-method hardware

python -m hardware.plot --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --evaluation-method hardware