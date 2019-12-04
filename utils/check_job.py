from qiskit.compiler import transpile, assemble
from utils.helper_fun import load_IBMQ, get_evaluator_info
from qiskit.providers.aer import noise
from qiskit import IBMQ
from qiskit.providers.jobstatus import JobStatus
import argparse
from qiskit.visualization import plot_gate_map, plot_error_map
import datetime
from datetime import timedelta
import time

def format_time(hours):
    t = datetime.datetime.now(datetime.timezone.utc)
    delta = timedelta(days=0,seconds=0,microseconds=0,milliseconds=0,minutes=0,hours=hours,weeks=0)
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

    terminal_status = [JobStatus['DONE'],JobStatus['CANCELLED'],JobStatus['ERROR']]

    time_now = datetime.datetime.now()

    for x in provider.backends():
        if 'qasm' not in str(x):
            evaluator_info = get_evaluator_info(circ=None,device_name=str(x),fields=['properties'])
            num_qubits = len(evaluator_info['properties'].qubits)
            if num_qubits==20:
                print('%s: %d-qubit, max %d jobs * %d shots'%(x,num_qubits,x.configuration().max_experiments,x.configuration().max_shots))
                print('Most recently QUEUED:')
                limit = 5 if str(x)=='ibmq_poughkeepsie' else 5
                total_queued = 0
                for job in x.jobs(limit=200,status='QUEUED'):
                    if total_queued < 5:
                        print(job.creation_date(),job.status(),job.queue_position(),job.job_id())
                    total_queued += 1
                print('Total queued = {:d}. Time stamp: {}'.format(total_queued,time_now))
                print('Most recently DONE:')
                for job in x.jobs(limit=5,status=JobStatus['DONE']):
                    print(job.creation_date(),job.status(),job.error_message(),job.job_id())
                if args.cancel_jobs:
                    [print('Warning!!! Cancelling jobs! 5 seconds count down') for i in range(5)]
                    time.sleep(5)
                    for job in x.jobs(limit=500,status='QUEUED'):
                        print(job.creation_date(),job.status(),job.queue_position(),job.job_id())
                        job.cancel()
                        print('cancelled')
                print('-'*100)