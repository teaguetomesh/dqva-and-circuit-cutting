CIRCUIT_TYPE="$1"
DEVICE_NAME="$2"

if [ ! -d "./hardware/logs/" ]; then
  mkdir ./hardware/logs/
fi

python -m hardware.generator --min-qubit 5 --max-qubit 5 --max-clusters 5 --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE 2>&1 | tee -a ./hardware/logs/$CIRCUIT_TYPE\_$DEVICE_NAME\_logs.txt

mpiexec -n 2 python -m utils.evaluator --experiment-name hardware --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --shots-mode saturated --evaluation-method hardware
python -m hardware.job_submittor --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --shots-mode saturated 2>&1 | tee -a ./hardware/logs/$CIRCUIT_TYPE\_$DEVICE_NAME\_logs.txt
python -m utils.reconstructor --experiment-name hardware --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --shots-mode saturated --evaluation-method hardware 2>&1 | tee -a ./hardware/logs/$CIRCUIT_TYPE\_$DEVICE_NAME\_logs.txt

mpiexec -n 2 python -m utils.evaluator --experiment-name hardware --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --shots-mode sametotal --evaluation-method hardware
python -m hardware.job_submittor --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --shots-mode sametotal 2>&1 | tee -a ./hardware/logs/$CIRCUIT_TYPE\_$DEVICE_NAME\_logs.txt
python -m utils.reconstructor --experiment-name hardware --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --shots-mode sametotal --evaluation-method hardware 2>&1 | tee -a ./hardware/logs/$CIRCUIT_TYPE\_$DEVICE_NAME\_logs.txt

python -m hardware.plot --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --evaluation-method hardware 2>&1 | tee -a ./hardware/logs/$CIRCUIT_TYPE\_$DEVICE_NAME\_logs.txt

# python -m utils.check_output --experiment-name hardware --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --shots-mode saturated --evaluation-method hardware