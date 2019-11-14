from qiskit.compiler import transpile, assemble
from helper_fun import load_IBMQ, get_evaluator_info
from qiskit.providers.aer import noise
from qiskit import IBMQ
from qiskit.providers.jobstatus import JobStatus
import argparse
from qiskit.visualization import plot_gate_map, plot_error_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='check hardware jobs')
    parser.add_argument('--cancel-jobs',action="store_true",help='cancel all running jobs')
    args = parser.parse_args()

    provider = load_IBMQ()

    for x in provider.backends():
        if 'qasm' not in str(x):
            evaluator_info = get_evaluator_info(circ=None,device_name=str(x),fields=['properties'])
            num_qubits = len(evaluator_info['properties'].qubits)
            print('%s: %d-qubit, max %d jobs * %d shots'%(x,num_qubits,x.configuration().max_experiments,x.configuration().max_shots))
            # plot_error_map(x).savefig('./plots/%s_error_map.png'%str(x))
            if num_qubits==20:
                for job in x.jobs():
                    if job.status() in [JobStatus['DONE'],JobStatus['CANCELLED'],JobStatus['ERROR']]:
                        print(job.creation_date(),job.status(),job.error_message(),job.job_id())
                    else:
                        print(job.creation_date(),job.status(),job.job_id())
                        if args.cancel_jobs:
                            job.cancel()
                            print('cancelled')