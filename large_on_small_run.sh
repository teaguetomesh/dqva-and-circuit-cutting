DEVICE_NAME="$1"

if [ ! -d "./large_on_small/logs/" ]; then
  mkdir ./large_on_small/logs/
fi

python -m large_on_small.generator

mpiexec -n 3 python -m utils.evaluator --experiment-name large_on_small --device-name ibmq_boeblingen --circuit-type supremacy --shots-mode sametotal --evaluation-method statevector_simulator
python -m utils.reconstructor --experiment-name large_on_small --device-name $DEVICE_NAME --circuit-type bv --shots-mode saturated --evaluation-method noisy_qasm_simulator

python -m large_on_small.plot