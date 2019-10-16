# evaluator_input.p
# NOTE: toggle here to change max qc size, max clusters
python generate_evaluator_input.py --max-qubit 6 --max-clusters 6
EVALUATOR_FILES=./benchmark_data/evaluator_input_*.p

# NOTE: toggle here to change number of repetitions
for i in {1..1};
do
    for f in $EVALUATOR_FILES;
    do
        echo $f
        # uniter_input.p
        # mpiexec -n 5 python evaluator_prob.py --input-file $f --saturated-shots
        mpiexec -n 5 python evaluator_prob.py --input-file $f
    done

    UNITER_INPUT_FILES=./benchmark_data/*_uniter_input_*.p
    for f in $UNITER_INPUT_FILES;
    do
        echo $f
        # uniter_output.p
        python uniter_prob.py --input-file $f
        rm $f
    done

    # plotter_input.p
    python data_combiner.py
    
    UNITER_OUTPUT_FILES=./benchmark_data/*_uniter_output_*.p
    for f in $UNITER_OUTPUT_FILES;
        do
            rm $f
        done
done

for f in $EVALUATOR_FILES;
    do
        rm $f
    done

python plot.py