import math
import copy
from utils.helper_fun import get_evaluator_info

class Job:
    def __init__(self,max_experiments,max_shots):
        self.max_experiments = max_experiments
        self.max_shots = max_shots
        self.circ_list = []
        self.shots = 0
        self.total_circs = 0
    
    def update(self, circ, shots):
        circ_shots = max(shots,self.shots)
        circ_shots = min(circ_shots,self.max_shots)
        reps = math.ceil(shots/circ_shots)
        if self.total_circs + reps > self.max_experiments:
            return True
        else:
            self.circ_list.append((circ,reps))
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
        job = Job(max_experiments=device_max_experiments,max_shots=device_max_shots)
        while len(circ_dict)>0:
            key = list(circ_dict.keys())[0]
            circ = circ_dict[key]['circ']
            shots = circ_dict[key]['shots']
            # print('adding circuit with %d shots to job'%shots)
            overflow = job.update(circ,shots)
            if overflow:
                # print('OVERFLOW, has %d total circuits * %d shots'%(job.total_circs,job.shots))
                schedule.append(job)
                job = job(max_experiments=device_max_experiments,max_shots=device_max_shots)
            else:
                # print('Did not overflow, has %d total circuits * %d shots'%(job.total_circs,job.shots))
                del circ_dict[key]
        if job.total_circs>0:
            schedule.append(job)
        return schedule

    def submit(self,schedule):
        jobs = []
        return jobs

    def retrieve(self,jobs):
        return circ_dict