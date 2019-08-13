python generate_circ_clusters.py
FILES=./data/cluster*.p

start=$(date +%s.%N)
for f in $FILES
do
  echo "Processing $f file..."
  mpiexec -n 4 python simulator_init.py --cluster-file $f
done
end=$(date +%s.%N)
runtime=$(python -c "print(${end} - ${start})")
echo "Runtime was $runtime"