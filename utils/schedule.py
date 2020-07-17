"""
Job submission/simulating backend
Input:
circ_dict (dict): circ (not transpiled), shots, evaluator_info (optional)
"""

import math
import copy
import numpy as np
from utils.helper_fun import get_evaluator_info, apply_measurement
from utils.conversions import dict_to_array, memory_to_dict
from qiskit.compiler import transpile, assemble
from qiskit import Aer, execute
import random

class ScheduleItem:
    def __init__(self,max_experiments,max_shots):
        self.max_experiments = max_experiments
        self.max_shots = max_shots
        self.circ_list = []
        self.shots = 0
        self.total_circs = 0
    
    def update(self, key, circ, shots):
        reps_vacant = self.max_experiments - self.total_circs
        if reps_vacant>0:
            circ_shots = max(shots,self.shots)
            circ_shots = min(circ_shots,self.max_shots)
            total_reps = math.ceil(shots/circ_shots)
            reps_to_add = min(total_reps,reps_vacant)
            circ_list_item = {'key':key,'circ':circ,'reps':reps_to_add}
            self.circ_list.append(circ_list_item)
            self.shots = circ_shots
            self.total_circs += reps_to_add
            shots_remaining = shots - reps_to_add * self.shots
        else:
            shots_remaining = shots
        return shots_remaining

class DeviceScheduler:
    def __init__(self,circ_dict,device_name):
        self.circ_dict = copy.deepcopy(circ_dict)
        self.device_name = device_name
        evaluator_info = get_evaluator_info(circ=None,device_name=self.device_name,fields=['device','properties'])
        self.device_size = len(evaluator_info['properties'].qubits)
        self.check_input()
        self.schedule = self.get_schedule(device_max_shots=evaluator_info['device'].configuration().max_shots,
        device_max_experiments=evaluator_info['device'].configuration().max_experiments)

    def check_input(self):
        assert isinstance(self.circ_dict,dict)
        for key in self.circ_dict:
            value = self.circ_dict[key]
            if 'circuit' not in value or 'shots' not in value:
                raise Exception('Input circ_dict should have `circuit`, `shots` for key {}'.format(key))
            elif value['circuit'].n_qubits > self.device_size:
                raise Exception('Input `circuit` for key {} has {:d}-q ({:d}-q device)'.format(key,value['circuit'].n_qubits,self.device_size))

    def get_schedule(self,device_max_shots,device_max_experiments):
        circ_dict = copy.deepcopy(self.circ_dict)
        schedule = []
        schedule_item = ScheduleItem(max_experiments=device_max_experiments,max_shots=device_max_shots)
        key_idx = 0
        while key_idx<len(circ_dict):
            key = list(circ_dict.keys())[key_idx]
            circ = circ_dict[key]['circuit']
            shots = circ_dict[key]['shots']
            # print('adding %d qubit circuit with %d shots to job'%(len(circ.qubits),shots))
            shots_remaining = schedule_item.update(key,circ,shots)
            if shots_remaining>0:
                # print('OVERFLOW, has %d total circuits * %d shots'%(job.total_circs,job.shots))
                schedule.append(schedule_item)
                schedule_item = ScheduleItem(max_experiments=device_max_experiments,max_shots=device_max_shots)
                circ_dict[key]['shots'] = shots_remaining
            else:
                # print('Did not overflow, has %d total circuits * %d shots'%(job.total_circs,job.shots))
                circ_dict[key]['shots'] = shots_remaining
                key_idx += 1
        if schedule_item.total_circs>0:
            schedule.append(schedule_item)
        return schedule

    def run(self):
        print('*'*20,'Submitting jobs','*'*20,flush=True)
        jobs = []
        for idx, schedule_item in enumerate(self.schedule):
            # print('Submitting job %d/%d'%(idx+1,len(schedule)))
            # print('Has %d total circuits * %d shots, %d circ_list elements'%(schedule_item.total_circs,schedule_item.shots,len(schedule_item.circ_list)))
            job_circuits = []
            for element in schedule_item.circ_list:
                key = element['key']
                circ = element['circ']
                reps = element['reps']
                # print('Key {}, {:d} qubit circuit * {:d} reps'.format(key,len(circ.qubits),reps))
                
                if len(circ.clbits)>0:
                    # Assume circ was alredy mapped
                    mapped_circuit = circ
                else:
                    device_evaluator_info = get_evaluator_info(circ=circ,device_name=self.device_name,fields=['device','initial_layout'])
                    qc=apply_measurement(circ=circ)
                    mapped_circuit = transpile(qc,backend=device_evaluator_info['device'],initial_layout=device_evaluator_info['initial_layout'])

                # print('scheduler:')
                # print(circ)
                # print(mapped_circuit)
                circs_to_add = [mapped_circuit]*reps
                job_circuits += circs_to_add
            
            assert len(job_circuits) == schedule_item.total_circs
            
            
            qobj = assemble(job_circuits, backend=device_evaluator_info['device'], shots=schedule_item.shots,memory=True)
            hw_job = device_evaluator_info['device'].run(qobj)
            jobs.append(hw_job)
            print('Submitting job {:d}/{:d} {} --> {:d} circuits, {:d} * {:d} shots'.format(idx+1,len(self.schedule),hw_job.job_id(),len(schedule_item.circ_list),len(job_circuits),schedule_item.shots),flush=True)
        self.jobs = jobs

    def retrieve(self,force_prob):
        print('*'*20,'Retrieving jobs','*'*20)
        assert len(self.schedule) == len(self.jobs)
        circ_dict = copy.deepcopy(self.circ_dict)
        memories = {}
        for job_idx in range(len(self.jobs)):
            schedule_item = self.schedule[job_idx]
            hw_job = self.jobs[job_idx]
            print('Retrieving job {:d}/{:d} {} --> {:d} circuits, {:d} * {:d} shots'.format(
                job_idx+1,len(self.jobs),hw_job.job_id(),
                len(schedule_item.circ_list),schedule_item.total_circs,schedule_item.shots),flush=True)
            hw_result = hw_job.result()
            start_idx = 0
            for element_ctr, element in enumerate(schedule_item.circ_list):
                key = element['key']
                circ = element['circ']
                reps = element['reps']
                end_idx = start_idx + reps
                # print('{:d}: getting {:d}-{:d}/{:d} circuits, key {} : {:d} qubit'.format(element_ctr,start_idx,end_idx-1,schedule_item.total_circs-1,key,len(circ.qubits)),flush=True)
                for result_idx in range(start_idx,end_idx):
                    experiment_hw_memory = hw_result.get_memory(result_idx)
                    if key in memories:
                        memories[key] += experiment_hw_memory
                    else:
                        memories[key] = experiment_hw_memory
                start_idx = end_idx
        for key in circ_dict:
            full_circ = circ_dict[key]['circuit']
            shots = circ_dict[key]['shots']
            memory = memories[key]
            mem_dict = memory_to_dict(memory=memory[:shots])
            hw_prob = dict_to_array(distribution_dict=mem_dict,force_prob=force_prob)
            circ_dict[key]['prob'] = copy.deepcopy(hw_prob)
            # print('Key {} has {:d} qubit circuit, hw has {:d}/{:d} shots'.format(key,len(full_circ.qubits),sum(hw.values()),shots))
            # print('Expecting {:d} shots, got {:d} shots'.format(shots,sum(mem_dict.values())),flush=True)
            if len(full_circ.clbits)>0:
                assert len(circ_dict[key]['prob']) == 2**len(full_circ.clbits)
            else:
                assert len(circ_dict[key]['prob']) == 2**len(full_circ.qubits)
        self.circ_dict = copy.deepcopy(circ_dict)

class SimulatorScheduler:
    def __init__(self,circ_dict,device_name):
        self.circ_dict = copy.deepcopy(circ_dict)
        self.device_name = device_name
        self.check_input()
        self.schedule = self.get_schedule(device_max_shots=int(2**100),device_max_experiments=int(2**20))

    def check_input(self):
        assert isinstance(self.circ_dict,dict)
        for key in self.circ_dict:
            value = self.circ_dict[key]
            if 'circuit' not in value:
                raise Exception('Input circ_dict should have `circuit` for key {}'.format(key))
            elif self.device_name=='qasm' and 'shots' not in value:
                raise Exception('Input circ_dict should have `shots` for key {}'.format(key))
            elif self.device_name=='qasm' and value['shots']>int(2**100):
                raise Exception('Input circ_dict have too many shots for key {}'.format(key))

    def get_schedule(self,device_max_shots,device_max_experiments):
        circ_dict = copy.deepcopy(self.circ_dict)
        schedule = []
        schedule_item = ScheduleItem(max_experiments=device_max_experiments,max_shots=device_max_shots)
        key_idx = 0
        while key_idx<len(circ_dict):
            key = list(circ_dict.keys())[key_idx]
            circ = circ_dict[key]['circuit']
            shots = circ_dict[key]['shots']
            # print('adding %d qubit circuit with %d shots to job'%(len(circ.qubits),shots))
            shots_remaining = schedule_item.update(key,circ,shots)
            if shots_remaining>0:
                # print('OVERFLOW, has %d total circuits * %d shots'%(job.total_circs,job.shots))
                schedule.append(schedule_item)
                schedule_item = ScheduleItem(max_experiments=device_max_experiments,max_shots=device_max_shots)
                circ_dict[key]['shots'] = shots_remaining
            else:
                # print('Did not overflow, has %d total circuits * %d shots'%(job.total_circs,job.shots))
                circ_dict[key]['shots'] = shots_remaining
                key_idx += 1
        if schedule_item.total_circs>0:
            schedule.append(schedule_item)
        return schedule

    def run(self):
        print('*'*20,'Submitting jobs','*'*20,flush=True)
        jobs = []
        for idx, schedule_item in enumerate(self.schedule):
            # print('Submitting job %d/%d'%(idx+1,len(schedule)))
            # print('Has %d total circuits * %d shots, %d circ_list elements'%(schedule_item.total_circs,schedule_item.shots,len(schedule_item.circ_list)))
            job_circuits = []
            for element in schedule_item.circ_list:
                key = element['key']
                circ = element['circ']
                reps = element['reps']
                # print('Key {}, {:d} qubit circuit * {:d} reps'.format(key,len(circ.qubits),reps))
                
                if len(circ.clbits)>0:
                    # Assume circ was alredy mapped
                    mapped_circuit = circ
                else:
                    qc=apply_measurement(circ=circ)
                    mapped_circuit = qc

                # print('scheduler:')
                # print(circ)
                # print(mapped_circuit)
                circs_to_add = [mapped_circuit]*reps
                job_circuits += circs_to_add
            
            assert len(job_circuits) == schedule_item.total_circs
            
            hw_job = execute(job_circuits, Aer.get_backend('qasm_simulator'), shots=schedule_item.shots)
            jobs.append(hw_job)
            print('Submitting job {:d}/{:d} {} --> {:d} circuits, {:d} * {:d} shots'.format(idx+1,len(self.schedule),hw_job.job_id(),len(schedule_item.circ_list),len(job_circuits),schedule_item.shots),flush=True)
        self.jobs = jobs

    def retrieve(self):
        print('*'*20,'Retrieving jobs','*'*20)
        assert len(self.schedule) == len(self.jobs)
        circ_dict = copy.deepcopy(self.circ_dict)
        memories = {}
        for job_idx in range(len(self.jobs)):
            schedule_item = self.schedule[job_idx]
            hw_job = self.jobs[job_idx]
            print('Retrieving job {:d}/{:d} {} --> {:d} circuits, {:d} * {:d} shots'.format(
                job_idx+1,len(self.jobs),hw_job.job_id(),
                len(schedule_item.circ_list),schedule_item.total_circs,schedule_item.shots),flush=True)
            hw_result = hw_job.result()
            start_idx = 0
            for element_ctr, element in enumerate(schedule_item.circ_list):
                key = element['key']
                circ = element['circ']
                reps = element['reps']
                assert reps==1
                end_idx = start_idx + reps
                # print('{:d}: getting {:d}-{:d}/{:d} circuits, key {} : {:d} qubit'.format(element_ctr,start_idx,end_idx-1,schedule_item.total_circs-1,key,len(circ.qubits)),flush=True)
                experiment_hw_memory = hw_result.get_counts(start_idx)
                memories[key] = experiment_hw_memory
                start_idx = end_idx
        for key in circ_dict:
            full_circ = circ_dict[key]['circuit']
            shots = circ_dict[key]['shots']
            memory = memories[key]
            circ_dict[key]['prob'] = copy.deepcopy(memory)
            # print('Key {} has {:d} qubit circuit, hw has {:d}/{:d} shots'.format(key,len(full_circ.qubits),sum(hw.values()),shots))
            # print('Expecting {:d} shots, got {:d} shots'.format(shots,sum(memory.values())),flush=True)
            assert shots==sum(memory.values())
        self.circ_dict = copy.deepcopy(circ_dict)