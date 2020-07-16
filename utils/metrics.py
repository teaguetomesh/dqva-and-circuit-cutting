import numpy as np

def chi2_distance(target,obs):
    assert len(target)==len(obs)
    obs = np.absolute(obs)
    obs = obs / sum(obs)
    # assert abs(sum(obs)-1)<1e-5
    distance = 0
    for x,y in zip(target,obs):
        if x-y==0:
            distance += 0
        else:
            distance += np.power(x-y,2)/(x+y)
    # distance /= len(target)
    return distance

def fidelity(target,obs):
    assert len(target)==len(obs)
    epsilon = 1e-20
    obs = np.absolute(obs)
    obs = obs / sum(obs)
    # assert abs(sum(obs)-1)<1e-5
    fidelity = 0
    for t,o in zip(target,obs):
        if t > 1e-16:
            fidelity += o
    return fidelity