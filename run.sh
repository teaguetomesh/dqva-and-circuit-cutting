# !/bin/bash

# ***
# *** "#SBATCH" lines must come before any non-blank, non-comment lines ***
# ***

# 1 nodes, 30 CPUs per node (total 30 CPUs), wall clock time of 5 hours

# SBATCH -N 1                  ## Node count
# SBATCH --ntasks-per-node=30   ## Processors per node
# SBATCH -t 0:15:00            ## Walltime


python generate_evaluator_input.py
FILES=./noisy_benchmark_data/evaluator_input_*.p

module load mpi
for f in $FILES
do
  echo "Processing $f file..."
  mpiexec -n 5 python evaluator_prob.py --cluster-idx 0 --backend qasm_simulator --noisy --num-shots 1024 --dirname noisy_benchmark_data
done