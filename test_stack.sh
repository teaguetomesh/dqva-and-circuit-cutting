# evaluator_input.p
# NOTE: toggle here to change max qc size, max clusters
python generate_evaluator_input.py --min-qubit 5 --max-qubit 6 --max-clusters 6
EVALUATOR_FILES=./benchmark_data/evaluator_input.p

# NOTE: toggle here to change cluster shots
# mpiexec -n 5 python evaluator_prob.py --input-file ./benchmark_data/evaluator_input.p --saturated-shots
mpiexec -n 5 python evaluator_prob.py --input-file ./benchmark_data/evaluator_input.p

UNITER_INPUT_FILES=./benchmark_data/*_uniter_input_*.p
for f in $UNITER_INPUT_FILES;
do
    echo $f
    python uniter_prob.py --input-file $f
    rm $f
done

for f in $EVALUATOR_FILES;
    do
        rm $f
    done

python plot.py