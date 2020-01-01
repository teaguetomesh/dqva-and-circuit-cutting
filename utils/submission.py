import math
import copy
from utils.helper_fun import get_evaluator_info, apply_measurement
from qiskit.compiler import transpile, assemble
from qiskit import Aer

class ScheduleItem:
    def __init__(self,max_experiments,max_shots):
        self.max_experiments = max_experiments
        self.max_shots = max_shots
        self.circ_list = []
        self.shots = 0
        self.total_circs = 0
    
    def update(self, key, circ, shots):
        circ_shots = max(shots,self.shots)
        circ_shots = min(circ_shots,self.max_shots)
        reps = math.ceil(shots/circ_shots)
        if self.total_circs + reps > self.max_experiments:
            return True
        else:
            self.circ_list.append({'key':key,'circ':circ,'reps':reps})
            self.shots = circ_shots
            self.total_circs += reps
            return False

class Scheduler:
    def __init__(self,circ_dict,device_name):
        self.circ_dict = circ_dict
        self.device_name = device_name
        self.check_input()

    def check_input(self):
        assert isinstance(self.circ_dict,dict)
        for key in self.circ_dict:
            value = self.circ_dict[key]
            try:
                circ = value['circ']
                shots = value['shots']
            except (AttributeError, TypeError):
                raise AssertionError('Input circ_dict should have circ and shots')

    def get_schedule(self):
        circ_dict = copy.deepcopy(self.circ_dict)
        evaluator_info = get_evaluator_info(circ=None,device_name=self.device_name,fields=['properties','device'])
        device_size = len(evaluator_info['properties'].qubits)
        device_max_shots = evaluator_info['device'].configuration().max_shots
        device_max_experiments = int(evaluator_info['device'].configuration().max_experiments)
        schedule = []
        schedule_item = ScheduleItem(max_experiments=device_max_experiments,max_shots=device_max_shots)
        while len(circ_dict)>0:
            key = list(circ_dict.keys())[0]
            circ = circ_dict[key]['circ']
            shots = circ_dict[key]['shots']
            # print('adding %d qubit circuit with %d shots to job'%(len(circ.qubits),shots))
            overflow = schedule_item.update(key,circ,shots)
            if overflow:
                # print('OVERFLOW, has %d total circuits * %d shots'%(job.total_circs,job.shots))
                schedule.append(schedule_item)
                schedule_item = ScheduleItem(max_experiments=device_max_experiments,max_shots=device_max_shots)
            else:
                # print('Did not overflow, has %d total circuits * %d shots'%(job.total_circs,job.shots))
                del circ_dict[key]
        if schedule_item.total_circs>0:
            schedule.append(schedule_item)
        return schedule

    def submit_schedule(self,schedule):
        print('*'*20,'Submitting jobs','*'*20)
        jobs = []
        for idx, schedule_item in enumerate(schedule):
            print('Submitting job %d/%d'%(idx+1,len(schedule)))
            # print('Has %d total circuits * %d shots, %d circ_list elements'%(job.total_circs,job.shots,len(job.circ_list)))
            job_circuits = []
            for element in schedule_item.circ_list:
                key = element['key']
                circ = element['circ']
                reps = element['reps']
                print('Key {} {:d} qubit circuit * {:d} reps'.format(key,len(circ.qubits),reps))
                
                evaluator_info = get_evaluator_info(circ=circ,device_name=self.device_name,
                fields=['device','basis_gates','coupling_map','properties','initial_layout'])

                qc=apply_measurement(circ)
                mapped_circuit = transpile(qc,
                backend=evaluator_info['device'], basis_gates=evaluator_info['basis_gates'],
                coupling_map=evaluator_info['coupling_map'],backend_properties=evaluator_info['properties'],
                initial_layout=evaluator_info['initial_layout'])

                circs_to_add = [mapped_circuit]*reps
                job_circuits += circs_to_add
            
            assert len(job_circuits) == schedule_item.total_circs
            qobj = assemble(job_circuits, backend=evaluator_info['device'], shots=schedule_item.shots)
            hw_job = Aer.get_backend('qasm_simulator').run(qobj)
            jobs.append(hw_job)
            print('Job {:d}/{:d} {} --> submitted {:d} circuits * {:d} shots'.format(idx+1,len(schedule),hw_job.job_id(),len(job_circuits),schedule_item.shots))
        return jobs

    def retrieve(self,schedule,jobs):
        print('*'*20,'Retrieving jobs','*'*20)
        assert len(schedule) == len(jobs)
        circ_dict = copy.deepcopy(self.circ_dict)
        for job_idx in range(len(jobs)):
            schedule_item = schedule[job_idx]
            hw_job = jobs[job_idx]
            print('Retrieving job {:d}/{:d}, job_id {} --> {:d} circuits'.format(job_idx+1,len(jobs),hw_job.job_id(),schedule_item.total_circs))
            hw_result = hw_job.result()
            result_idx = 0
            for element in schedule_item.circ_list:
                experiment_hw_counts = hw_result.get_counts(result_idx)