from qiskit.compiler import transpile, assemble
from utils.helper_fun import load_IBMQ, get_evaluator_info
from qiskit.providers.aer import noise
from qiskit import IBMQ
from qiskit.providers.jobstatus import JobStatus
import argparse
from qiskit.visualization import plot_gate_map, plot_error_map
from datetime import timedelta, datetime, timezone
import time

def format_time(hours):
    t = datetime.now(timezone.utc)
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

    time_now = datetime.now(timezone.utc)
    time_delta = format_time(hours=24)

    for x in provider.backends():
        if 'qasm' not in str(x) and 'tokyo' not in str(x):
            evaluator_info = get_evaluator_info(circ=None,device_name=str(x),fields=['properties'])
            num_qubits = len(evaluator_info['properties'].qubits)
            if num_qubits==20:
                print('%s: %d-qubit, max %d jobs * %d shots'%(x,num_qubits,x.configuration().max_experiments,x.configuration().max_shots))
                queued_jobs = []
                run_jobs = []
                done_jobs = []
                error_jobs = []
                total_queued = 0
                for job in x.jobs(limit=200):
                    if job.status() == JobStatus['QUEUED']:
                        queued_jobs.append(job)
                    elif job.status() == JobStatus['RUNNING']:
                        run_jobs.append(job)
                    elif job.status() == JobStatus['DONE'] and job.creation_date()>time_delta:
                        done_jobs.append(job)
                    elif job.status() == JobStatus['ERROR'] and job.creation_date()>time_delta:
                        error_jobs.append(job)
                print_ctr = 0
                print('Most recently QUEUED:')
                for job in queued_jobs[-10:]:
                    print(job.creation_date(),job.status(),job.queue_position(),'ETA:',job.queue_info().estimated_complete_time-time_now)
                print('Total queued = {:d}.'.format(len(queued_jobs)))
                print('RUNNING:')
                for job in run_jobs:
                    print(job.creation_date(),job.status(),job.queue_position())
                print('Most recently DONE:')
                for job in done_jobs[-3:]:
                    print(job.creation_date(),job.status(),job.error_message(),job.job_id())
                print('Most recently ERROR:')
                for job in error_jobs[-3:]:
                    print(job.creation_date(),job.status(),job.error_message(),job.job_id())
                if args.cancel_jobs:
                    for i in range(5):
                        print('Warning!!! Cancelling jobs! 5 seconds count down')
                        time.sleep(1.2)
                    for job in queued_jobs:
                        print(job.creation_date(),job.status(),job.queue_position(),job.job_id())
                        job.cancel()
                        print('cancelled')
                print('-'*100)