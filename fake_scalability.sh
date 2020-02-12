CIRCUIT_TYPE="$1"

if [ ! -d "./large_on_small/logs/" ]; then
  mkdir ./large_on_small/logs/
fi

python -m large_on_small.fake_generator 2>&1 | tee -a ./large_on_small/logs/$CIRCUIT_TYPE\_logs_fake.txt

mpiexec -n 2 python -m utils.evaluator --experiment-name large_on_small --device-name fake --circuit-type $CIRCUIT_TYPE --evaluation-method fake 2>&1 | tee -a ./large_on_small/logs/$CIRCUIT_TYPE\_logs_fake.txt
# python -m utils.reconstructor --experiment-name large_on_small --device-name fake --circuit-type $CIRCUIT_TYPE --evaluation-method fake 2>&1 | tee -a ./large_on_small/logs/$CIRCUIT_TYPE\_logs_fake.txt
mpiexec -n 50 python -m utils.reconstructor_parallel --experiment-name large_on_small --device-name fake --circuit-type $CIRCUIT_TYPE --evaluation-method fake 2>&1 | tee -a ./large_on_small/logs/$CIRCUIT_TYPE\_logs_fake.txt