EXPERIMENT_NAME="$1"
DEVICE_NAME="$2"

rm -r ./$EXPERIMENT_NAME/logs
mkdir ./$EXPERIMENT_NAME/logs

python -m $EXPERIMENT_NAME.generate_evaluator_input --device-name $DEVICE_NAME 2>&1 | tee ./$EXPERIMENT_NAME/logs/bv\_$DEVICE_NAME\_generator_logs.txt

mpiexec -n 5 python -m $EXPERIMENT_NAME.evaluator_prob --device-name $DEVICE_NAME --evaluation-method noisy_qasm_simulator
python -m utils.uniter_prob --device-name $DEVICE_NAME --evaluation-method noisy_qasm_simulator 2>&1 | tee ./$EXPERIMENT_NAME/logs/bv\_saturated_$DEVICE_NAME\_uniter_logs.txt

echo "$DEVICE_NAME DONE"

python -m $EXPERIMENT_NAME.plot