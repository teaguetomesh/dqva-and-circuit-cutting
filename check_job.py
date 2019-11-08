from qcg.generators import gen_supremacy, gen_hwea
from qiskit.compiler import transpile, assemble
from helper_fun import load_IBMQ, readout_mitigation, get_evaluator_info
from qiskit.providers.aer import noise
from qiskit import IBMQ
from qiskit.providers.jobstatus import JobStatus

circ = gen_supremacy(2,2,8)
provider = load_IBMQ()

for x in provider.backends():
    if 'qasm' not in str(x) and str(x) == 'ibmq_boeblingen':
        evaluator_info = get_evaluator_info(circ=circ,device_name=str(x),fields=['properties'])
        num_qubits = len(evaluator_info['properties'].qubits)
        print('%s: %d-qubit, max %d jobs * %d shots'%(x,num_qubits,x.configuration().max_experiments,x.configuration().max_shots))
        for job in x.jobs():
            print(job.creation_date(),job.status(),job.job_id())
            # if job.status() == JobStatus['RUNNING']:
            #     job.cancel()