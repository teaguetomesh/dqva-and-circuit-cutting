# NOTE: toggle here to change max qc size, max clusters
echo "Generate evaluator input"
python generate_evaluator_input.py --min-qubit 5 --max-qubit 15 --max-clusters 4 --device-name ibmq_johannesburg --circuit-name bv --shots-scaling 100 2>&1 | tee ./logs/bv_generator_logs.txt

echo "Running evaluator"
mpiexec -n 2 python evaluator_prob.py --device-name ibmq_johannesburg --circuit-name bv --evaluation-method hardware
echo "Running job submittor"
python hardware_job_submittor.py --device-name ibmq_johannesburg --circuit-name bv 2>&1 | tee ./logs/bv_hw_job_submittor_logs.txt
echo "Running reconstruction"
python uniter_prob.py --device-name ibmq_johannesburg --evaluation-method hardware --circuit-name bv 2>&1 | tee ./logs/bv_uniter_logs.txt

# python plot.py

# echo "Running evaluator"
# mpiexec -n 5 python evaluator_prob.py --evaluation-method statevector_simulator --device-name ibmq_johannesburg --circuit-name bv
# echo "Running reconstruction"
# python uniter_prob.py --evaluation-method statevector_simulator --device-name ibmq_johannesburg 2>&1 | tee ./logs/uniter_logs.txt

# echo "Running evaluator"
# mpiexec -n 5 python evaluator_prob.py --evaluation-method noisy_qasm_simulator --device-name ibmq_johannesburg --circuit-name bv
# echo "Running reconstruction"
# python uniter_prob.py --evaluation-method noisy_qasm_simulator --device-name ibmq_johannesburg --circuit-name bv 2>&1 | tee ./logs/uniter_logs.txt