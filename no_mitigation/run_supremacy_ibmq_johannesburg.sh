# NOTE: toggle here to change max qc size, max clusters
echo "Generate evaluator input"
python generate_evaluator_input.py --min-qubit 2 --max-qubit 9 --max-clusters 3 --device-name ibmq_johannesburg --circuit-type supremacy 2>&1 | tee ./logs/supremacy_ibmq_johannesburg_generator_logs.txt

{
echo "Running saturated hardware evaluator"
mpiexec -n 2 python evaluator_prob.py --device-name ibmq_johannesburg --circuit-type supremacy --shots-mode saturated --evaluation-method hardware
echo "Running job submittor"
python hardware_job_submittor.py --device-name ibmq_johannesburg --circuit-type supremacy --shots-mode saturated 2>&1 | tee ./logs/supremacy_saturated_ibmq_johannesburg_hw_job_submittor_logs.txt
echo "Running reconstruction"
python uniter_prob.py --device-name ibmq_johannesburg --circuit-type supremacy --shots-mode saturated --evaluation-method hardware 2>&1 | tee ./logs/supremacy_saturated_ibmq_johannesburg_uniter_logs.txt
} &
P1=$!

{
echo "Running sametotal hardware evaluator"
mpiexec -n 2 python evaluator_prob.py --device-name ibmq_johannesburg --circuit-type supremacy --shots-mode sametotal --evaluation-method hardware
echo "Running job submittor"
python hardware_job_submittor.py --device-name ibmq_johannesburg --circuit-type supremacy --shots-mode sametotal 2>&1 | tee ./logs/supremacy_sametotal_ibmq_johannesburg_hw_job_submittor_logs.txt
echo "Running reconstruction"
python uniter_prob.py --device-name ibmq_johannesburg --circuit-type supremacy --shots-mode sametotal --evaluation-method hardware 2>&1 | tee ./logs/supremacy_sametotal_ibmq_johannesburg_uniter_logs.txt
} &
P2=$!

wait $P1 $P2
echo "supremacy ibmq_johannesburg DONE"

# echo "Running saturated noisy qasm evaluator"
# mpiexec -n 5 python evaluator_prob.py --device-name ibmq_johannesburg --circuit-type supremacy --shots-mode saturated --evaluation-method noisy_qasm_simulator
# echo "Running reconstruction"
# python uniter_prob.py --device-name ibmq_johannesburg --circuit-type supremacy --shots-mode saturated --evaluation-method noisy_qasm_simulator 2>&1 | tee ./logs/supremacy_uniter_logs.txt

# echo "Running sametotal noisy qasm evaluator"
# mpiexec -n 5 python evaluator_prob.py --device-name ibmq_johannesburg --circuit-type supremacy --shots-mode sametotal --evaluation-method noisy_qasm_simulator
# echo "Running reconstruction"
# python uniter_prob.py --device-name ibmq_johannesburg --circuit-type supremacy --shots-mode sametotal --evaluation-method noisy_qasm_simulator 2>&1 | tee ./logs/supremacy_uniter_logs.txt

# echo "Running evaluator evaluator"
# mpiexec -n 5 python evaluator_prob.py --device-name ibmq_johannesburg --circuit-type supremacy --evaluation-method statevector_simulator
# echo "Running reconstruction"
# python uniter_prob.py --device-name ibmq_johannesburg --circuit-type supremacy --evaluation-method statevector_simulator 2>&1 | tee ./logs/supremacy_uniter_logs.txt

# python plot.py