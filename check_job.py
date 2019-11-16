from qiskit.compiler import transpile, assemble
from helper_fun import load_IBMQ, get_evaluator_info
from qiskit.providers.aer import noise
from qiskit import IBMQ
from qiskit.providers.jobstatus import JobStatus
import argparse
from qiskit.visualization import plot_gate_map, plot_error_map
import datetime
from datetime import timedelta

def format_time():
    t = datetime.datetime.now(datetime.timezone.utc)
    delta = timedelta(days=0,seconds=0,microseconds=0,milliseconds=0,minutes=0,hours=5,weeks=0)
    t = t - delta
    s = t.strftime('%Y-%m-%dT%H:%M:%S.%f')
    tail = s[-7:]
    f = round(float(tail), 3)
    temp = "%.3f" % f
    return "%s%sZ" % (s[:-7], temp[1:])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='check hardware jobs')
    parser.add_argument('--cancel-jobs',action="store_true",help='cancel all running jobs')
    args = parser.parse_args()

    provider = load_IBMQ()

    past_5_hrs = format_time()
    terminal_status = [JobStatus['DONE'],JobStatus['CANCELLED'],JobStatus['ERROR']]

    for x in provider.backends():
        if 'qasm' not in str(x):
            evaluator_info = get_evaluator_info(circ=None,device_name=str(x),fields=['properties'])
            num_qubits = len(evaluator_info['properties'].qubits)
            if num_qubits==20:
                print('%s: %d-qubit, max %d jobs * %d shots'%(x,num_qubits,x.configuration().max_experiments,x.configuration().max_shots))
                for job in x.jobs():
                    if args.cancel_jobs and job.status() not in terminal_status:
                        print(job.creation_date(),job.status(),job.queue_position(),job.job_id())
                        job.cancel()
                        print('cancelled')
                    if job.status() not in terminal_status:
                        print(job.creation_date(),job.status(),job.queue_position(),job.job_id())
                    elif job.creation_date()>past_5_hrs:
                        print(job.creation_date(),job.status(),job.error_message(),job.job_id())
                print('-'*100)