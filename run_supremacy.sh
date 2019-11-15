# NOTE: toggle here to change max qc size, max clusters
echo "Generate evaluator input"
python generate_evaluator_input.py --min-qubit 2 --max-qubit 8 --max-clusters 4 --device-name ibmq_boeblingen --circuit-name supremacy 2>&1 | tee ./logs/supremacy_generator_logs.txt

echo "Running evaluator"
mpiexec -n 2 python evaluator_prob.py --device-name ibmq_boeblingen --circuit-name supremacy --evaluation-method hardware
echo "Running job submittor"
python hardware_job_submittor.py --device-name ibmq_boeblingen --circuit-name supremacy 2>&1 | tee ./logs/supremacy_hw_job_submittor_logs.txt
echo "Running reconstruction"
python uniter_prob.py --device-name ibmq_boeblingen --evaluation-method hardware --circuit-name supremacy 2>&1 | tee ./logs/supremacy_uniter_logs.txt

# python plot.py

# echo "Running evaluator"
# mpiexec -n 5 python evaluator_prob.py --evaluation-method statevector_simulator --device-name ibmq_johannesburg --circuit-name supremacy
# echo "Running reconstruction"
# python uniter_prob.py --evaluation-method statevector_simulator --device-name ibmq_johannesburg 2>&1 | tee ./logs/uniter_logs.txt

# echo "Running evaluator"
# mpiexec -n 5 python evaluator_prob.py --evaluation-method noisy_qasm_simulator --device-name ibmq_johannesburg --circuit-name supremacy
# echo "Running reconstruction"
# python uniter_prob.py --evaluation-method noisy_qasm_simulator --device-name ibmq_johannesburg --circuit-name supremacy 2>&1 | tee ./logs/uniter_logs.txt