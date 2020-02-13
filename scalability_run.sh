CIRCUIT_TYPE="$1"
MIN_SIZE="$2"
MAX_SIZE="$3"

if [ ! -d "./scalability/logs/" ]; then
  mkdir ./scalability/logs/
fi

# python -m scalability.fake_generator --circuit-type $CIRCUIT_TYPE --min-size $MIN_SIZE --max-size $MAX_SIZE 2>&1 | tee -a ./scalability/logs/$CIRCUIT_TYPE\_logs.txt
# python -m scalability.fake_evaluator --circuit-type $CIRCUIT_TYPE 2>&1 | tee -a ./scalability/logs/$CIRCUIT_TYPE\_logs.txt

python -m scalability.fake_reconstructor --circuit-type $CIRCUIT_TYPE 2>&1 | tee -a ./scalability/logs/$CIRCUIT_TYPE\_logs.txt
# mpiexec -n 30 python -m scalability.fake_reconstructor_parallel --circuit-type $CIRCUIT_TYPE 2>&1 | tee -a ./scalability/logs/$CIRCUIT_TYPE\_logs.txt