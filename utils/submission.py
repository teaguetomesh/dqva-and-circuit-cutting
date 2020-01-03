import math
import copy
from utils.helper_fun import get_evaluator_info, apply_measurement, combine_dict, dict_to_prob, memory_to_dict
from qiskit.compiler import transpile, assemble
from qiskit import Aer, execute

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
        total_reps = math.ceil(shots/circ_shots)
        reps_vacant = self.max_experiments - self.total_circs
        reps_to_add = min(total_reps,reps_vacant)
        self.circ_list.append({'key':key,'circ':circ,'reps':reps_to_add})
        self.shots = circ_shots
        self.total_circs += reps_to_add
        shots_remaining = (total_reps - reps_to_add)*self.shots
        return shots_remaining

class Scheduler:
    def __init__(self,circ_dict,device_name):
        self.circ_dict = circ_dict
        self.device_name = device_name
        self.check_input()
        evaluator_info = get_evaluator_info(circ=None,device_name=self.device_name,fields=['properties','device'])
        self.device_max_shots = evaluator_info['device'].configuration().max_shots
        self.device_max_experiments = int(evaluator_info['device'].configuration().max_experiments)

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
        schedule = []
        schedule_item = ScheduleItem(max_experiments=self.device_max_experiments,max_shots=self.device_max_shots)
        key_idx = 0
        while key_idx<len(circ_dict):
            key = list(circ_dict.keys())[key_idx]
            circ = circ_dict[key]['circ']
            shots = circ_dict[key]['shots']
            # print('adding %d qubit circuit with %d shots to job'%(len(circ.qubits),shots))
            shots_remaining = schedule_item.update(key,circ,shots)
            if shots_remaining>0:
                # print('OVERFLOW, has %d total circuits * %d shots'%(job.total_circs,job.shots))
                schedule.append(schedule_item)
                schedule_item = ScheduleItem(max_experiments=self.device_max_experiments,max_shots=self.device_max_shots)
                circ_dict[key]['shots'] = shots_remaining
            else:
                # print('Did not overflow, has %d total circuits * %d shots'%(job.total_circs,job.shots))
                key_idx += 1
        if schedule_item.total_circs>0:
            schedule.append(schedule_item)
        return schedule

    def submit_schedule(self,schedule,real_device=False):
        print('*'*20,'Submitting jobs','*'*20)
        jobs = []
        for idx, schedule_item in enumerate(schedule):
            # print('Submitting job %d/%d'%(idx+1,len(schedule)))
            # print('Has %d total circuits * %d shots, %d circ_list elements'%(schedule_item.total_circs,schedule_item.shots,len(schedule_item.circ_list)))
            job_circuits = []
            for element in schedule_item.circ_list:
                key = element['key']
                circ = element['circ']
                reps = element['reps']
                # print('Key {} {:d} qubit circuit * {:d} reps'.format(key,len(circ.qubits),reps))
                
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
            if real_device:
                hw_job = evaluator_info['device'].run(qobj,backend_options={'memory':True})
            else:
                hw_job = Aer.get_backend('qasm_simulator').run(qobj,backend_options={'memory':True})
            jobs.append(hw_job)
            print('Submitting job {:d}/{:d} {} --> {:d} circuits, {:d} * {:d} shots'.format(idx+1,len(schedule),hw_job.job_id(),len(schedule_item.circ_list),len(job_circuits),schedule_item.shots),flush=True)
        return jobs

    def retrieve(self,schedule,jobs):
        print('*'*20,'Retrieving jobs','*'*20)
        assert len(schedule) == len(jobs)
        circ_dict = copy.deepcopy(self.circ_dict)
        for key in circ_dict:
            circ_dict[key]['hw'] = {}
        memories = {}
        for job_idx in range(len(jobs)):
            schedule_item = schedule[job_idx]
            hw_job = jobs[job_idx]
            print('Retrieving job {:d}/{:d} {} --> {:d} circuits, {:d} * {:d} shots'.format(job_idx+1,len(jobs),hw_job.job_id(),len(schedule_item.circ_list),schedule_item.total_circs,schedule_item.shots),flush=True)
            hw_result = hw_job.result()
            start_idx = 0
            for element in schedule_item.circ_list:
                key = element['key']
                circ = element['circ']
                reps = element['reps']
                end_idx = start_idx + reps
                # print('Getting {:d}-{:d} ({:d}/{:d}) circuits, key {} : {:d} qubit'.format(start_idx,end_idx-1,end_idx,schedule_item.total_circs,key,len(circ.qubits)),flush=True)
                for result_idx in range(start_idx,end_idx):
                    # experiment_hw_counts = hw_result.get_counts(result_idx)
                    experiment_hw_memory = hw_result.get_memory(result_idx)
                    if key in memories:
                        memories[key] += experiment_hw_memory
                    else:
                        memories[key] = experiment_hw_memory
                start_idx = end_idx
        for key in circ_dict:
            full_circ = circ_dict[key]['circ']
            shots = circ_dict[key]['shots']
            memory = memories[key]
            mem_dict = memory_to_dict(memory=memory[:shots])
            hw_prob = dict_to_prob(distribution_dict=mem_dict,reverse=True)
            circ_dict[key]['hw'] = copy.deepcopy(hw_prob)
            # print('Key {} has {:d} qubit circuit, hw has {:d}/{:d} shots'.format(key,len(full_circ.qubits),sum(hw.values()),shots))
            # print('Expecting {:d} shots, got {:d} shots'.format(shots,sum(mem_dict.values())),flush=True)
            assert len(circ_dict[key]['hw']) == 2**len(full_circ.qubits)
            assert sum(mem_dict.values())==shots
            assert abs(sum(circ_dict[key]['hw'])-1)<1e-10
        self.circ_dict = copy.deepcopy(circ_dict)