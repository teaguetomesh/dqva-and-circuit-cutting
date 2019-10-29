# evaluator_input.p
# NOTE: toggle here to change max qc size, max clusters
python generate_evaluator_input.py --min-qubit 3 --max-qubit 5 --max-clusters 5
EVALUATOR_FILES=./benchmark_data/evaluator_input.p

for i in {1..2};
do
    # NOTE: toggle here to change cluster shots
    # mpiexec -n 5 python evaluator_prob.py --input-file ./benchmark_data/evaluator_input.p --saturated-shots
    mpiexec -n 5 python evaluator_prob.py --input-file ./benchmark_data/evaluator_input.p

    UNITER_INPUT_FILE=./benchmark_data/*_uniter_input_*.p
    echo $UNITER_INPUT_FILE
    python uniter_prob.py --input-file $UNITER_INPUT_FILE
    rm $UNITER_INPUT_FILE
done

for f in $EVALUATOR_FILES;
    do
        rm $f
    done

python plot.py