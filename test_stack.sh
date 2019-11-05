# NOTE: toggle here to change max qc size, max clusters
python generate_evaluator_input.py --min-qubit 3 --max-qubit 6 --max-clusters 5 --device-name ibmq_boeblingen
EVALUATOR_FILES=./benchmark_data/evaluator_input_*.p

for i in {1..2};
do
    for f in $EVALUATOR_FILES;
    do
        # NOTE: toggle here to change cluster shots
        mpiexec -n 5 python evaluator_prob.py --input-file $f --saturated-shots --evaluation-method noisy_qasm_simulator

        UNITER_INPUT_FILE=./benchmark_data/*_uniter_input_*.p
        python uniter_prob.py --input-file $UNITER_INPUT_FILE
        rm $UNITER_INPUT_FILE
    done
done

for f in $EVALUATOR_FILES;
    do
        rm $f
    done

python plot.py