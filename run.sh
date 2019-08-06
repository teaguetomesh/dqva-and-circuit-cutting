python generate_circ_clusters.py
FILES=./data/cluster*.p
for f in $FILES
do
  echo "Processing $f file..."
  mpiexec -n 4 python simulator.py --cluster-file $f
done