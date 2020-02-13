CIRCUIT_TYPE="$1"
MIN_SIZE="$2"
MAX_SIZE="$3"

if [ ! -d "./large_on_small/logs/" ]; then
  mkdir ./large_on_small/logs/
fi

python -m large_on_small.generator --circuit-type $CIRCUIT_TYPE --min-size $MIN_SIZE --max-size $MAX_SIZE 2>&1 | tee -a ./large_on_small/logs/$CIRCUIT_TYPE\_logs.txt

mpiexec -n 10 python -m utils.evaluator --experiment-name large_on_small --device-name ibmq_boeblingen --circuit-type $CIRCUIT_TYPE --evaluation-method statevector_simulator 2>&1 | tee -a ./large_on_small/logs/$CIRCUIT_TYPE\_logs.txt
# python -m utils.reconstructor --experiment-name large_on_small --device-name ibmq_boeblingen --circuit-type $CIRCUIT_TYPE --evaluation-method statevector_simulator 2>&1 | tee -a ./large_on_small/logs/$CIRCUIT_TYPE\_logs.txt
mpiexec -n 30 python -m utils.reconstructor_parallel --experiment-name large_on_small --device-name ibmq_boeblingen --circuit-type $CIRCUIT_TYPE --evaluation-method statevector_simulator 2>&1 | tee -a ./large_on_small/logs/$CIRCUIT_TYPE\_logs.txt

# python -m large_on_small.plot