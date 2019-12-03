EXPERIMENT_NAME="$1"
DEVICE_NAME="$2"

rm -r ./$EXPERIMENT_NAME/logs
mkdir ./$EXPERIMENT_NAME/logs

python -m $EXPERIMENT_NAME.generate_evaluator_input.py --device-name $DEVICE_NAME 2>&1 | tee ./$EXPERIMENT_NAME/logs/bv\_$DEVICE_NAME\_generator_logs.txt

mpiexec -n 21 python evaluator_prob.py --device-name $DEVICE_NAME --evaluation-method noisy_qasm_simulator
python -m uniter_prob.py --device-name $DEVICE_NAME --evaluation-method noisy_qasm_simulator 2>&1 | tee ./logs/bv\_saturated_$DEVICE_NAME\_uniter_logs.txt

echo "$DEVICE_NAME DONE"

python plot.py