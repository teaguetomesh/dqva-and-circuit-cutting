python generate_evaluator_input.py --fc-shots 100000 --cluster-shots 10000

for i in {1..3};
do
    EVALUATOR_FILES=./noisy_benchmark_data/evaluator_input_*.p
    for f in $EVALUATOR_FILES;
    do
        echo $f
        mpiexec -n 5 python evaluator_prob.py --input-file $f
    done

    UNITER_FILES=./noisy_benchmark_data/*_uniter_input_*.p
    for f in $UNITER_FILES;
    do
        echo $f
        python uniter_prob.py --input-file $f
        rm $f
    done

    python data_combiner.py
done