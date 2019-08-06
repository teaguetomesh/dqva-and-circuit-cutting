#!/usr/bin/env python

# To run:
#
# for i in $(seq 10 30); do ./benchmark_ansatz.py -q $i; done
#

import argparse
import timeit
import logging
import numpy as np
import csv
from pathlib import Path
from qiskit import Aer, execute
from qcg.generators import gen_hwea


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", type=int, default=5, help="number of qubits")
    parser.add_argument("-t","--threads", type=int, default=1,
                        help="number of threads qiskit aer is using")
    parser.add_argument("-l","--layers", type=int, default=1,
                        help="number of layers to use in HWEA")
    parser.add_argument("--save",type=str,
                        help="saves summarized results as a csv, with name as parameter. If csv exists, it appends to the end")
    parser.add_argument("--loop",type=str,default=None,
                        help="option to loop over nthreads or nqubits")
    args = parser.parse_args()
    return args


def runjob(nq, nlayers, nthreads, savefn):
    backend = Aer.get_backend("qasm_simulator")

    qc = gen_hwea(nq,nlayers,parameters='seeded',measure=True)

    start_time = timeit.default_timer()
    qobj = execute(qc, backend=backend,
                   backend_options = {"max_parallel_threads" : nthreads})
    res = qobj.result()
    runtime = timeit.default_timer() - start_time
    print("Finished {} qubits in: {:.2f} sec on {} threads".format(nq,
                                                                   runtime,
                                                                   nthreads))
    header = ['nqubits', 'nthreads', 'runtime (sec)', 'nlayers']
    results = [nq, nthreads, runtime, nlayers]
    if savefn:
        if Path(savefn).exists():
            write_header = False
        else:
            write_header = True

        with open(savefn, 'a') as csvfile:
            out = csv.writer(csvfile)
            if write_header:
                out.writerow(header)
            out.writerow(results)


def main():
    logging.basicConfig(level=logging.INFO)

    args = parse_args()

    if args.loop == 'qubits':
        for q in range(1,args.q + 1):
            runjob(q, args.layers, args.threads, args.save)
    elif args.loop == 'threads':
        for t in range(1,args.threads + 1):
            runjob(args.q, args.layers, t, args.save)
    else:
        runjob(args.q, args.layers, args.threads, args.save)


if __name__ == '__main__':
    main()



