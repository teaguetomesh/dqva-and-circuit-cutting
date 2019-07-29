from qcg.generators import gen_supremacy
import pickle
from qiskit import execute, Aer

def tensorproduct(dict1, dict2):
    dict3 = {}
    for key1 in dict1.keys():
        for key2 in dict2.keys():
            dict3[key1+key2] = dict1[key1] * dict2[key2]
    return dict3


def main():
    d1 = {'00':0.5,'11':0.5}

    # Generate some sample dictionaries as a test problem
    # I will want to pickle these so I don't have to 
    # simulate every time, but forget that for now.
    circ1 = gen_supremacy(4,4,8,barriers=False,measure=True)
    circ2 = gen_supremacy(4,4,8,barriers=False,measure=True)

    dict_list = []
    for circ in [circ1, circ2]:
        simulator = Aer.get_backend('qasm_simulator')
        result = execute(circ, simulator, shots=8192).result()
        dict_list += [result.get_counts(circ)]

    # in our uniter we have to do as many tensor products as
    # we have fragments. Here we will simulate have multiple
    # fragments by repeatedly using the same dictionaries

    # do the first tensor product
    tprod = tensorproduct(dict_list[0],dict_list[1])

    # now for each remaining fragment, perform the tensorproduct
    for rem_dict in [dict_list[0], dict_list[1]]:
        tprod = tensorproduct(tprod, rem_dict)

    print('FINISHED')


if __name__ == '__main__':
    main()
