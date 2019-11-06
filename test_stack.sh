# NOTE: toggle here to change max qc size, max clusters
python generate_evaluator_input.py --min-qubit 4 --max-qubit 4 --max-clusters 5 --device-name ibmq_boeblingen
EVALUATOR_FILE=./benchmark_data/evaluator_input_*.p

for i in {1..1};
do
    # NOTE: toggle here to change cluster shots
    # mpiexec -n 5 python evaluator_prob.py --input-file $EVALUATOR_FILE --saturated-shots --evaluation-method statevector_simulator
    # mpiexec -n 5 python evaluator_prob.py --input-file $EVALUATOR_FILE --saturated-shots --evaluation-method noisy_qasm_simulator
    mpiexec -n 5 python evaluator_prob.py --input-file $EVALUATOR_FILE --saturated-shots --evaluation-method hardware
    JOB_SUBMITTOR_FILE=./benchmark_data/job_submittor_input_*.p
    python hardware_job_submittor.py --input-file $JOB_SUBMITTOR_FILE --saturated-shots
    rm $JOB_SUBMITTOR_FILE

    # UNITER_INPUT_FILE=./benchmark_data/*_uniter_input_*.p
    # python uniter_prob.py --input-file $UNITER_INPUT_FILE
    # rm $UNITER_INPUT_FILE
done

rm $EVALUATOR_FILE

# python plot.py