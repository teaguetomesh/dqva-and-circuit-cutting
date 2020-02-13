CIRCUIT_TYPE="$1"

if [ ! -d "./large_on_small/logs/" ]; then
  mkdir ./large_on_small/logs/
fi

python -m large_on_small.generator 2>&1 | tee -a ./large_on_small/logs/$CIRCUIT_TYPE\_logs.txt

mpiexec -n 30 python -m utils.evaluator --experiment-name large_on_small --device-name ibmq_boeblingen --circuit-type $CIRCUIT_TYPE --evaluation-method statevector_simulator 2>&1 | tee -a ./large_on_small/logs/$CIRCUIT_TYPE\_logs.txt
# python -m utils.reconstructor --experiment-name large_on_small --device-name ibmq_boeblingen --circuit-type $CIRCUIT_TYPE --evaluation-method statevector_simulator 2>&1 | tee -a ./large_on_small/logs/$CIRCUIT_TYPE\_logs.txt
mpiexec -n 30 python -m utils.reconstructor_parallel --experiment-name large_on_small --device-name ibmq_boeblingen --circuit-type $CIRCUIT_TYPE --evaluation-method statevector_simulator 2>&1 | tee -a ./large_on_small/logs/$CIRCUIT_TYPE\_logs.txt

# python -m large_on_small.plot
