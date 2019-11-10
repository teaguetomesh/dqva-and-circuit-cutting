# NOTE: toggle here to change max qc size, max clusters
echo "Generate evaluator input"
python generate_evaluator_input.py --min-qubit 6 --max-qubit 6 --max-clusters 5 --device-name ibmq_boeblingen

# NOTE: toggle here to change cluster shots
echo "Running evaluator"
# mpiexec -n 5 python evaluator_prob.py --saturated-shots --evaluation-method statevector_simulator --device-name ibmq_boeblingen

# mpiexec -n 5 python evaluator_prob.py --saturated-shots --evaluation-method noisy_qasm_simulator --device-name ibmq_boeblingen

mpiexec -n 2 python evaluator_prob.py --saturated-shots --evaluation-method hardware --device-name ibmq_boeblingen
echo "Running job submittor"
python hardware_job_submittor.py --saturated-shots --device-name ibmq_boeblingen

echo "Running reconstruction"
python uniter_prob.py --device-name ibmq_boeblingen --evaluation-method hardware --saturated-shots

python plot.py