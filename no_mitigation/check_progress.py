import pickle
import matplotlib.pyplot as plt
import numpy as np
from helper_fun import cross_entropy, fidelity

f = open('./benchmark_data/evaluator_input_ibmq_poughkeepsie_supremacy.p', 'rb' )
plotter_input = pickle.load(f)
print(plotter_input.keys())