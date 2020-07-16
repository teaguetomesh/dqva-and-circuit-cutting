from qiskit.compiler import transpile, assemble
from utils.helper_fun import load_IBMQ
from qiskit.providers.aer import noise
from qiskit import IBMQ
from qiskit.providers.jobstatus import JobStatus
import argparse
from qiskit.visualization import plot_gate_map, plot_error_map
from datetime import timedelta, datetime, timezone
import time
import subprocess

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

    subprocess.run(['rm','-r','./devices'])
    subprocess.run(['mkdir','./devices'])

    provider = load_IBMQ()

    terminal_status = [JobStatus['DONE'],JobStatus['CANCELLED'],JobStatus['ERROR']]
    devices_to_check = ['ibmq_cambridge', 'ibmq_paris', 'ibmq_johannesburg', 'ibmq_16_melbourne', 'ibmqx2']

    time_now = datetime.now(timezone.utc)
    delta = timedelta(days=0,seconds=0,microseconds=0,milliseconds=0,minutes=0,hours=6,weeks=0)
    time_delta = time_now - delta

    for x in provider.backends():
        if str(x) in devices_to_check:
            device = provider.get_backend(str(x))
            properties = device.properties()
            num_qubits = len(properties.qubits)
            print('%s: %d-qubit, max %d jobs * %d shots'%(x,num_qubits,x.configuration().max_experiments,x.configuration().max_shots))
            try:
                fig = plot_error_map(device)
                fig.savefig('./devices/%s.png'%str(x),dpi=400)
            except:
                print('...%s fails to plot error map...'%x)
            jobs_to_cancel = []
            print('Most recently QUEUED:')
            print_ctr = 0
            for job in x.jobs(limit=50,status=JobStatus['QUEUED']):
                if print_ctr<5:
                    print(job.creation_date(),job.status(),job.queue_position(),job.job_id(),'ETA:',job.queue_info().estimated_complete_time-time_now)
                jobs_to_cancel.append(job)
                print_ctr+=1
            print('RUNNING:')
            for job in x.jobs(limit=5,status=JobStatus['RUNNING']):
                print(job.creation_date(),job.status(),job.queue_position())
                jobs_to_cancel.append(job)
            print('Most recently DONE:')
            for job in x.jobs(limit=5,status=JobStatus['DONE'],start_datetime=time_delta):
                print(job.creation_date(),job.status(),job.error_message(),job.job_id())
            print('Most recently ERROR:')
            for job in x.jobs(limit=5,status=JobStatus['ERROR'],start_datetime=time_delta):
                print(job.creation_date(),job.status(),job.error_message(),job.job_id())
            if args.cancel_jobs:
                for i in range(5):
                    print('Warning!!! Cancelling jobs! %d seconds count down'%(5-i))
                    time.sleep(1.2)
                for job in jobs_to_cancel:
                    print(job.creation_date(),job.status(),job.queue_position(),job.job_id())
                    job.cancel()
                    print('cancelled')
            print('-'*100)