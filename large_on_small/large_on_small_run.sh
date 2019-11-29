DEVICE_NAME="$1"

python generate_evaluator_input.py --device-name $DEVICE_NAME 2>&1 | tee ./logs/bv\_$DEVICE_NAME\_generator_logs.txt

mpiexec -n 21 python evaluator_prob.py --device-name $DEVICE_NAME --evaluation-method noisy_qasm_simulator
python uniter_prob.py --device-name $DEVICE_NAME --evaluation-method noisy_qasm_simulator 2>&1 | tee ./logs/bv\_saturated_$DEVICE_NAME\_uniter_logs.txt

echo "$DEVICE_NAME DONE"

python plot.py