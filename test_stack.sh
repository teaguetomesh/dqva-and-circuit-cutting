# NOTE: toggle here to change max qc size, max clusters
echo "Generate evaluator input"
python generate_evaluator_input.py --min-qubit 9 --max-qubit 9 --max-clusters 5 --device-name ibmq_boeblingen
EVALUATOR_FILE=./benchmark_data/evaluator_input_*.p

# NOTE: toggle here to change cluster shots
echo "Running evaluator"
# mpiexec -n 5 python evaluator_prob.py --input-file $EVALUATOR_FILE --saturated-shots --evaluation-method statevector_simulator

# mpiexec -n 5 python evaluator_prob.py --input-file $EVALUATOR_FILE --saturated-shots --evaluation-method noisy_qasm_simulator

# TODO: add more exact file destinations
mpiexec -n 2 python evaluator_prob.py --input-file $EVALUATOR_FILE --saturated-shots --evaluation-method hardware
echo "Running job submittor"
JOB_SUBMITTOR_FILE=./benchmark_data/job_submittor_input_*.p
python hardware_job_submittor.py --input-file $JOB_SUBMITTOR_FILE --saturated-shots
# rm $JOB_SUBMITTOR_FILE

echo "Running reconstruction"
UNITER_INPUT_FILE=./benchmark_data/*_uniter_input_*.p
python uniter_prob.py --input-file $UNITER_INPUT_FILE
# rm $UNITER_INPUT_FILE

# rm $EVALUATOR_FILE

# python plot.py