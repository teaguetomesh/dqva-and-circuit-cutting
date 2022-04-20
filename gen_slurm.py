#!/usr/bin/env python
import sys, os

planted = False
ncuts = 2
N = 26
ncom = 2
pin = 20
pout = 2
nfrags = 2
shots = 60000
partition_alg = 'klb'
optimizer = 'COBYLA'

for graph_num in range(1, 61):
    if planted:
        path = f"N{N}_{ncom}com/cuts{ncuts}/G{graph_num}/"
    else:
        path = f"N{N}_d3/cuts{ncuts}/G{graph_num}/"
    if not os.path.isdir(path):
        os.makedirs(path)

    for rep in range(1, 11):
        if planted:
            slurmfn = f"cuts{ncuts}N{N}G{graph_num}_{ncom}com_{nfrags}frags_{shots}shots_rep{rep}.sh"
        else:
            slurmfn = f"cuts{ncuts}N{N}G{graph_num}_d3_{nfrags}frags_{shots}shots_rep{rep}.sh"

        with open(path+slurmfn, 'w') as sf:
            sf.write("#!/usr/bin/env bash\n")
            sf.write(f"#SBATCH -o /n/fs/qcteague/dqva-and-circuit-cutting/slurm_output/dqva_{slurmfn.strip('.sh')}.out\n")
            sf.write("#SBATCH -p defq\n")
            sf.write("#SBATCH --nodes=1\n")
            sf.write("#SBATCH --ntasks=1\n")
            sf.write("#SBATCH --cpus-per-task=4\n")
            sf.write("#SBATCH -t 48:00:00\n")
            sf.write("#SBATCH --mail-type=end\n")
            sf.write("#SBATCH --mem=8000M\n")
            sf.write("#SBATCH --mail-user=ttomesh@cs.princeton.edu\n")
            sf.write("source /n/fs/qcteague/dqva-and-circuit-cutting/cutEnv/bin/activate\n")

            if planted:
                pythoncommand = f"python /n/fs/qcteague/dqva-and-circuit-cutting/run_dqva_and_cutting.py -p /n/fs/qcteague/dqva-and-circuit-cutting/ --graph \"benchmark_graphs/N{N}_com{ncom}_pin{pin}_pout{pout}_graphs/G{graph_num}.txt\" --numcuts {ncuts} --shots {shots} --rounds 10 --rep {rep} --numfrags {nfrags} --optimizer {optimizer} --graphalg klb --resultdir MICRO_results"
            else:
                pythoncommand = f"python /n/fs/qcteague/dqva-and-circuit-cutting/run_dqva_and_cutting.py -p /n/fs/qcteague/dqva-and-circuit-cutting/ --graph \"benchmark_graphs/N{N}_d3_graphs/G{graph_num}.txt\" --numcuts {ncuts} --shots {shots} --rounds 10 --rep {rep} --numfrags {nfrags} --optimizer {optimizer} --graphalg klb --resultdir MICRO_results"

            sf.write(pythoncommand)
