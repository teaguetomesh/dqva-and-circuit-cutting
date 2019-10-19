import pickle
import matplotlib.pyplot as plt

f = open('./plots/classical_sametotal.p', 'rb')
fig = pickle.load(f)
plt.show()