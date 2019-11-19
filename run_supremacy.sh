# NOTE: toggle here to change max qc size, max clusters
echo "Generate evaluator input"
python generate_evaluator_input.py --min-qubit 2 --max-qubit 19 --max-clusters 3 --device-name ibmq_johannesburg --circuit-type supremacy --shots-mode saturated 2>&1 | tee ./logs/supremacy_generator_logs.txt

echo "Running evaluator"
mpiexec -n 2 python evaluator_prob.py --device-name ibmq_johannesburg --circuit-type supremacy --shots-mode saturated --evaluation-method hardware
echo "Running job submittor"
python hardware_job_submittor.py --device-name ibmq_johannesburg --circuit-name supremacy 2>&1 | tee ./logs/supremacy_hw_job_submittor_logs.txt
echo "Running reconstruction"
python uniter_prob.py --device-name ibmq_johannesburg --evaluation-method hardware --circuit-name supremacy 2>&1 | tee ./logs/supremacy_uniter_logs.txt

# python plot.py

# echo "Running evaluator"
# mpiexec -n 5 python evaluator_prob.py --device-name ibmq_johannesburg --circuit-type supremacy --shots-mode saturated --evaluation-method statevector_simulator
# echo "Running reconstruction"
# python uniter_prob.py --device-name ibmq_johannesburg --circuit-type supremacy --shots-mode saturated --evaluation-method statevector_simulator 2>&1 | tee ./logs/uniter_logs.txt

# echo "Running evaluator"
# mpiexec -n 5 python evaluator_prob.py --device-name ibmq_johannesburg --circuit-type supremacy --shots-mode saturated --evaluation-method noisy_qasm_simulator
# echo "Running reconstruction"
# python uniter_prob.py --device-name ibmq_johannesburg --circuit-type supremacy --shots-mode saturated --evaluation-method noisy_qasm_simulator 2>&1 | tee ./logs/uniter_logs.txt