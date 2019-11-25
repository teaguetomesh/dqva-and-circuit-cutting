CIRCUIT_TYPE="$1"
DEVICE_NAME="$2"

python generate_evaluator_input.py --min-qubit 2 --max-qubit 9 --max-clusters 3 --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE 2>&1 | tee ./logs/$CIRCUIT_TYPE\_$DEVICE_NAME\_generator_logs.txt

{
mpiexec -n 5 python evaluator_prob.py --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --shots-mode saturated --evaluation-method noisy_qasm_simulator 2>&1 | tee ./logs/$CIRCUIT_TYPE\_saturated_$DEVICE_NAME\_evaluator_logs.txt
python uniter_prob.py --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --shots-mode saturated --evaluation-method noisy_qasm_simulator 2>&1 | tee ./logs/$CIRCUIT_TYPE\_saturated_$DEVICE_NAME\_uniter_logs.txt
} &
P1=$!

{
mpiexec -n 5 python evaluator_prob.py --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --shots-mode sametotal --evaluation-method noisy_qasm_simulator 2>&1 | tee ./logs/$CIRCUIT_TYPE\_sametotal_$DEVICE_NAME\_evaluator_logs.txt
python uniter_prob.py --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --shots-mode sametotal --evaluation-method noisy_qasm_simulator 2>&1 | tee ./logs/$CIRCUIT_TYPE\_sametotal_$DEVICE_NAME\_uniter_logs.txt
} &
P2=$!

wait $P1 $P2
echo "$CIRCUIT_TYPE on $DEVICE_NAME DONE"

python plot.py --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --evaluation-method noisy_qasm_simulator