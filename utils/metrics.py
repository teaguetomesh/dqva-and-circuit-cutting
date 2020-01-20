import numpy as np

def cross_entropy(target,obs):
    assert len(target)==len(obs)
    epsilon = 1e-20
    obs = [abs(x) if x!=0 else epsilon for x in obs]
    sum_of_prob = sum(obs)
    obs = [x/sum_of_prob for x in obs]
    assert abs(sum(obs)-1) < 1e-10
    h = 0
    for p,q in zip(target,obs):
        if p==0:
            h += 0
        else:
            h += p*np.log(p/q)
    return h

def entropy(prob_dist):
    epsilon = 1e-20
    prob_dist = [abs(x) if x!=0 else epsilon for x in prob_dist]
    sum_of_prob = sum(prob_dist)
    prob_dist = [x/sum_of_prob for x in prob_dist]
    assert abs(sum(prob_dist)-1) < 1e-10
    h = 0
    for p in prob_dist:
        if p==0:
            h += 0
        else:
            h -= p*np.log(p)
    return h

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
        if t!= 0:
            fidelity += o
    return fidelity