python generate_evaluator_input.py

for i in {1..2};
do
    EVALUATOR_FILES=./noisy_benchmark_data/evaluator_input_*.p
    for f in $EVALUATOR_FILES;
    do
        echo $f
        mpiexec -n 3 python evaluator_prob.py --input-file $f --shots 1000
    done

    UNITER_FILES=./noisy_benchmark_data/uniter_input_*.p
    for f in $UNITER_FILES;
    do
        echo $f
        python uniter_prob.py --input-file $f
    done

    python data_combiner.py
done