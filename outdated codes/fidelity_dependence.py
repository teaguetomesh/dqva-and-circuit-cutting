from qcg.generators import gen_supremacy, gen_hwea
from helper_fun import evaluate_circ, cross_entropy, get_evaluator_info, fidelity
import matplotlib
import matplotlib.pyplot as plt
from plot import heatmap, annotate_heatmap
import numpy as np

def find_saturated_shots(circ,accuracy):
    ground_truth = evaluate_circ(circ=circ,backend='statevector_simulator',evaluator_info=None)
    min_ce = cross_entropy(target=ground_truth,obs=ground_truth)
    qasm_prob = [0 for i in ground_truth]
    shots_increment = min(1024,10*int(np.power(2,len(circ.qubits))))
    evaluator_info = {}
    evaluator_info['num_shots'] = shots_increment
    counter = 0.0
    while 1:
        counter += 1.0
        qasm_prob_batch = evaluate_circ(circ=circ,backend='noiseless_qasm_simulator',evaluator_info=evaluator_info)
        qasm_prob = [(x*(counter-1)+y)/counter for x,y in zip(qasm_prob,qasm_prob_batch)]
        ce = cross_entropy(target=ground_truth,obs=qasm_prob)
        diff = (ce-min_ce)/min_ce
        if diff < accuracy:
            return int(counter*shots_increment)
        # if counter%50==49:
        #     print('current diff:',diff,'current shots:',int(counter*shots_increment))

def evaluate_full_circ(circ, total_shots, device_name):
    print('Evaluate full circuit, %d shots'%total_shots)
    sv_noiseless_fc = evaluate_circ(circ=circ,backend='statevector_simulator',evaluator_info=None)

    print('Evaluating fc qasm, %d shots'%total_shots)
    evaluator_info = {'num_shots':total_shots}
    qasm_noiseless_fc = evaluate_circ(circ=circ,backend='noiseless_qasm_simulator',evaluator_info=evaluator_info)

    print('Evaluating fc qasm + noise, %d shots'%total_shots)
    evaluator_info = get_evaluator_info(circ=circ,device_name=device_name,
    fields=['device','basis_gates','coupling_map','properties','initial_layout','noise_model'])
    evaluator_info['num_shots'] = total_shots
    qasm_noisy_fc = evaluate_circ(circ=circ,backend='noisy_qasm_simulator',evaluator_info=evaluator_info)

    ground_truth_ce = cross_entropy(target=sv_noiseless_fc,obs=sv_noiseless_fc)
    qasm_noiseless_ce = cross_entropy(target=sv_noiseless_fc,obs=qasm_noiseless_fc)
    qasm_noise_ce = cross_entropy(target=sv_noiseless_fc,obs=qasm_noisy_fc)
    noise_effect_ce = (qasm_noise_ce - ground_truth_ce)/(qasm_noiseless_ce - ground_truth_ce)

    ground_truth_fid = fidelity(target=sv_noiseless_fc,obs=sv_noiseless_fc)
    assert ground_truth_fid == 1
    qasm_noiseless_fid = fidelity(target=sv_noiseless_fc,obs=qasm_noiseless_fc)
    assert qasm_noiseless_fid == 1
    qasm_noise_fid = fidelity(target=sv_noiseless_fc,obs=qasm_noisy_fc)
    noise_effect_fid = qasm_noise_fid

    return noise_effect_fid

if __name__ == '__main__':

    dimension_l = [[1,3],[2,2],[1,5],[2,3],[1,7],[2,4],[3,3]]
    depths = [1,2,3,4,5]
    widths = []
    percent_dict = {}
    num_trials = 3
    for trial in range(num_trials):
        for dimension in dimension_l:
            for depth in depths:
                # print('-'*100)
                i,j = dimension
                width = i*j
                widths.append(width)
                case = (width,depth)
                # print('Case',case)

                # full_circ = gen_supremacy(i,j,depth)
                full_circ = gen_hwea(i*j,depth)
                num_shots = find_saturated_shots(circ=full_circ,accuracy=5e-3)
                
                percent = evaluate_full_circ(circ=full_circ, total_shots=num_shots, device_name='ibmq_boeblingen')
                if case in percent_dict:
                    percent_dict[case] = percent_dict[case] + percent
                else:
                    percent_dict[case] = percent
                # print('-'*100)

    for case in percent_dict:
        percent_dict[case] = percent_dict[case]/num_trials
    print(percent_dict)

    widths = list(set(widths))
    depths.sort(reverse=True)
    data_map = np.zeros((len(depths), len(widths)))
    for depth in depths:
        for width in widths:
            case = (width,depth)
            percent = percent_dict[case]
            row_idx = depths.index(depth)
            col_idx = widths.index(width)
            data_map[row_idx, col_idx] = percent

    fig, ax = plt.subplots(figsize=(10,10))

    im, cbar = heatmap(data=data_map, row_labels=depths, col_labels=widths, ax=ax,
                    cmap="YlGn", cbarlabel="Fidelity")
    texts = annotate_heatmap(im, valfmt="{x:.3f}")
    ax.set_xlabel('Circuit width')
    ax.set_ylabel('Circuit depth')
    fig.tight_layout()
    plt.savefig('fidelity_dependence.png',dpi=400)