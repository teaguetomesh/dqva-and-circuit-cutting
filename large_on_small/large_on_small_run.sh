DEVICE_NAME="$1"

python generate_evaluator_input.py --device-name $DEVICE_NAME 2>&1 | tee ./logs/bv\_$DEVICE_NAME\_generator_logs.txt

{
mpiexec -n 2 python evaluator_prob.py --device-name $DEVICE_NAME --evaluation-method noisy_qasm_simulator
python uniter_prob.py --device-name $DEVICE_NAME --evaluation-method noisy_qasm_simulator 2>&1 | tee ./logs/$CIRCUIT_TYPE\_saturated_$DEVICE_NAME\_uniter_logs.txt
} &
P1=$!

{
mpiexec -n 2 python evaluator_prob.py --device-name $DEVICE_NAME --evaluation-method statevector_simulator
python uniter_prob.py --device-name $DEVICE_NAME --evaluation-method statevector_simulator 2>&1 | tee ./logs/$CIRCUIT_TYPE\_saturated_$DEVICE_NAME\_uniter_logs.txt
} &
P2=$!

wait $P1 $P2
echo "$DEVICE_NAME DONE"

python plot.py