CIRCUIT_TYPE="$1"
DEVICE_NAME="$2"

if [ ! -d "./large_on_small/logs/" ]; then
  mkdir ./large_on_small/logs/
fi

python -m large_on_small.generator

mpiexec -n 2 python -m utils.evaluator --experiment-name large_on_small --device-name ibmq_boeblingen --circuit-type supremacy --evaluation-method statevector_simulator
# python -m utils.reconstructor --experiment-name large_on_small --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --evaluation-method statevector_simulator
mpiexec -n 20 python -m utils.reconstructor_parallel --experiment-name large_on_small --device-name $DEVICE_NAME --circuit-type $CIRCUIT_TYPE --evaluation-method statevector_simulator

# python -m large_on_small.plot