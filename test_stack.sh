rm -r ./logs
mkdir logs
rm -r benchmark_data
mkdir benchmark_data
rm -r plots
mkdir plots
# NOTE: toggle here to change max qc size, max clusters
echo "Generate evaluator input"
python generate_evaluator_input.py --min-qubit 3 --max-qubit 9 --max-clusters 3 --device-name ibmq_boeblingen 2>&1 | tee ./logs/generator_logs.txt

echo "Running evaluator"
# mpiexec -n 5 python evaluator_prob.py --saturated-shots --evaluation-method statevector_simulator --device-name ibmq_boeblingen

# mpiexec -n 5 python evaluator_prob.py --saturated-shots --evaluation-method noisy_qasm_simulator --device-name ibmq_boeblingen

mpiexec -n 2 python evaluator_prob.py --saturated-shots --evaluation-method hardware --device-name ibmq_boeblingen
echo "Running job submittor"
python hardware_job_submittor.py --saturated-shots --device-name ibmq_boeblingen 2>&1 | tee ./logs/hw_job_submittor_logs.txt

echo "Running reconstruction"
python uniter_prob.py --device-name ibmq_boeblingen --evaluation-method hardware --saturated-shots 2>&1 | tee ./logs/uniter_logs.txt

python plot.py 2>&1 | tee ./logs/plotter_logs.txt