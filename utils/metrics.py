import numpy as np
from scipy import stats

def kl_divergence(target,obs):
    assert len(target)==len(obs)
    epsilon = 1e-20
    obs = [abs(x) if x!=0 else epsilon for x in obs]
    kl = stats.entropy(pk=target,qk=obs)
    return kl

def entropy(prob_dist):
    epsilon = 1e-20
    prob_dist = [abs(x) if x!=0 else epsilon for x in prob_dist]
    ent = stats.entropy(pk=prob_dist)
    return ent

def fidelity(target,obs):
    assert len(target)==len(obs)
    epsilon = 1e-20
    obs = [abs(x) if x!=0 else epsilon for x in obs]
    sum_of_prob = sum(obs)
    obs = [x/sum_of_prob for x in obs]
    if abs(sum(obs)-1) > 1e-10:
        print('sum of obs =',sum(obs))
    fidelity = 0
    for t,o in zip(target,obs):
        if t > 1e-16:
            fidelity += o
    return fidelity

def chi2_distance(target,obs):
    assert len(target)==len(obs)
    obs = np.absolute(obs)
    obs = obs / sum(obs)
    assert abs(sum(obs)-1)<1e-10
    distance = 0
    for x,y in zip(target,obs):
        if x-y==0:
            distance += 0
        else:
            distance += np.power(x-y,2)/(x+y)
    return distance