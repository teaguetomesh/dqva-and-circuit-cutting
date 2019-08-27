python generate_circ_clusters.py
FILES=./data/cluster_*_circ.p

start=`date +%s`
for f in $FILES
do
  echo "Processing $f file..."
  mpiexec -n 8 python simulator_init.py --cluster-file $f
done
end=`date +%s`
runtime=$((end-start))
echo "Runtime was $runtime"

start=`date +%s`
for f in $FILES
do
  echo "Processing $f file..."
  mpiexec -n 8 python simulator_init_meas.py --cluster-file $f
done
end=`date +%s`
runtime=$((end-start))
echo "Runtime was $runtime"