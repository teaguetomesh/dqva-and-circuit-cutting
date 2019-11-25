import pickle
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from helper_fun import cross_entropy, fidelity
import os
import argparse

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text_to_fill = valfmt(data[i, j], None) if data[i, j]!=0 else 'DNE'
            text = im.axes.text(j, i, text_to_fill, **kw)
            texts.append(text)

    return texts

def plot_tradeoff(best_cc,circuit_type,figname):
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.xlabel('Number of qubits')
    if circuit_type == 'supremacy':
        plt.plot([fc for fc in best_cc], [best_cc[fc]['ce_percent'] for fc in best_cc], 'bX')
        plt.ylabel('Cross entropy reduction (%)')
        plt.ylim(0,100)
    elif circuit_type == 'bv' or circuit_type=='hwea':
        plt.plot([fc for fc in best_cc], [best_cc[fc]['fid_percent'] for fc in best_cc], 'bX')
        plt.ylabel('Fidelity improvement (%)')
    plt.xticks([x for x in best_cc])
    plt.subplot(122)
    plt.plot([fc for fc in best_cc], [best_cc[fc]['uniter_time'] for fc in best_cc], 'r*')
    plt.xlabel('Number of qubits')
    plt.ylabel('Reconstruction time (s)')
    plt.xticks([x for x in best_cc])
    plt.tight_layout()
    plt.savefig('%s_tradeoff.png'%figname[:-2],dpi=400)
    plt.close()

def plot_3d_bar(plotter_input,hw_qubits,fc_qubits):
    searcher_times = [plotter_input[case]['searcher_time'] for case in plotter_input]
    classical_times = [plotter_input[case]['classical_time'] for case in plotter_input]
    quantum_times = [plotter_input[case]['quantum_time'] for case in plotter_input]
    uniter_times = [plotter_input[case]['uniter_time'] for case in plotter_input]
    ce_percent_changes = [plotter_input[case]['ce_percent_reduction'] for case in plotter_input]
    fid_percent_changes = [plotter_input[case]['fid_percent_improvement'] for case in plotter_input]
    fig_scale = 4.5
    fig = plt.figure(figsize=(3*fig_scale,2*fig_scale))
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.bar3d(hw_qubits, fc_qubits, np.zeros(len(plotter_input)), dx, dy, searcher_times)
    ax1.set_xlabel('hardware qubits')
    ax1.set_ylabel('full circuit qubits')
    ax1.set_zlabel('searcher time (seconds)')
    ax1 = fig.add_subplot(232, projection='3d')
    ax1.bar3d(hw_qubits, fc_qubits, np.zeros(len(plotter_input)), dx, dy, classical_times)
    ax1.set_zlim3d(0, 1.2*max(classical_times)+1)
    ax1.set_xlabel('hardware qubits')
    ax1.set_ylabel('full circuit qubits')
    ax1.set_zlabel('classical evaluator time (seconds)')
    ax1 = fig.add_subplot(233, projection='3d')
    ax1.bar3d(hw_qubits, fc_qubits, np.zeros(len(plotter_input)), dx, dy, quantum_times)
    ax1.set_zlim3d(0, 1.2*max(quantum_times)+1)
    ax1.set_xlabel('hardware qubits')
    ax1.set_ylabel('full circuit qubits')
    ax1.set_zlabel('quantum evaluator time (seconds)')
    ax1 = fig.add_subplot(234, projection='3d')
    ax1.bar3d(hw_qubits, fc_qubits, np.zeros(len(plotter_input)), dx, dy, uniter_times)
    ax1.set_xlabel('hardware qubits')
    ax1.set_ylabel('full circuit qubits')
    ax1.set_zlabel('reconstructor time (seconds)')
    ax1 = fig.add_subplot(235, projection='3d')
    ax1.bar3d(hw_qubits, fc_qubits, np.zeros(len(plotter_input)), dx, dy, ce_percent_changes)
    ax1.set_zlim3d(min(0,1.2*min(ce_percent_changes)), max(0,1.2*max(ce_percent_changes)))
    ax1.set_xlabel('hardware qubits')
    ax1.set_ylabel('full circuit qubits')
    ax1.set_zlabel('cross entropy gap reduction due to cutting (%)')
    ax1 = fig.add_subplot(236, projection='3d')
    ax1.bar3d(hw_qubits, fc_qubits, np.zeros(len(plotter_input)), dx, dy, fid_percent_changes)
    ax1.set_zlim3d(min(0,1.2*min(fid_percent_changes)), max(0,1.2*max(fid_percent_changes)))
    ax1.set_xlabel('hardware qubits')
    ax1.set_ylabel('full circuit qubits')
    ax1.set_zlabel('Fidelity improvement due to cutting (%)')
    plt.savefig('%s.png'%figname[:-2],dpi=400)
    plt.close()

def plot_heatmap(plotter_input,hw_qubits,fc_qubits,circuit_type,figname):
    hw_qubits_unique = list(np.unique(hw_qubits))
    fc_qubits_unique = list(np.unique(fc_qubits))
    fc_qubits_unique.sort(reverse=True)
    reduction_map = np.zeros((len(fc_qubits_unique), len(hw_qubits_unique)))
    for fc_qubit in fc_qubits_unique:
        for hw_qubit in hw_qubits_unique:
            case = (hw_qubit,fc_qubit)
            if circuit_type == 'supremacy':
                percent = plotter_input[case]['ce_percent_reduction'] if case in plotter_input else 0
            elif circuit_type == 'bv' or circuit_type=='hwea':
                percent = plotter_input[case]['fid_percent_improvement'] if case in plotter_input else 0
            row_idx = fc_qubits_unique.index(fc_qubit)
            col_idx = hw_qubits_unique.index(hw_qubit)
            reduction_map[row_idx, col_idx] = percent

    fig, ax = plt.subplots(figsize=(10,10))

    im, cbar = heatmap(reduction_map, fc_qubits_unique, hw_qubits_unique, ax=ax,
                    cmap="YlGn", cbarlabel="Cross Entropy Loss Reduction [%]" if circuit_type == 'supremacy' or circuit_type == 'qft' else "Fidelity Improvement [%]")
    texts = annotate_heatmap(im, valfmt="{x:.1f}")
    ax.set_xlabel('Hardware qubits')
    ax.set_ylabel('Full circuit qubits')

    metric_type = 'ce' if (circuit_type == 'supremacy' or circuit_type == 'qft') else 'fid'
    fig.tight_layout()
    plt.savefig('{}_{}_map.png'.format(figname[:-2],metric_type),dpi=1000)
    plt.close()

def plot_fid_bar(saturated_best_cc,sametotal_best_cc,circuit_type,figname):
    sametotal_fc_size = list(sametotal_best_cc.keys())
    saturated_fc_size = list(saturated_best_cc.keys())
    all_fc_size = list(set().union(sametotal_fc_size,saturated_fc_size))
    all_fc_size.sort()
    std = []
    saturated_cutting = []
    sametotal_cutting = []
    for fc_size in all_fc_size:
        has_std = False
        if fc_size in saturated_best_cc:
            std.append(saturated_best_cc[fc_size]['qasm_noise_ce'] if circuit_type=='supremacy' else saturated_best_cc[fc_size]['qasm_noise_fid'])
            has_std = True
            saturated_cutting.append(saturated_best_cc[fc_size]['cutting_ce'] if circuit_type=='supremacy' else saturated_best_cc[fc_size]['cutting_fid'])
        else:
            saturated_cutting.append(0)
        if fc_size in sametotal_best_cc:
            if not has_std:
                std.append(sametotal_best_cc[fc_size]['qasm_noise_ce'] if circuit_type=='supremacy' else sametotal_best_cc[fc_size]['qasm_noise_fid'])
            sametotal_cutting.append(sametotal_best_cc[fc_size]['cutting_ce'] if circuit_type=='supremacy' else sametotal_best_cc[fc_size]['cutting_fid'])
        else:
            sametotal_cutting.append(0)

    n_groups = len(all_fc_size)
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.2
    opacity = 0.8

    if circuit_type == 'supremacy':
        plt.ylabel('\u0394H')
        plt.title('\u0394H Reduction')
    elif circuit_type == 'bv' or circuit_type=='hwea':
        plt.ylim(0,1)
        plt.ylabel('Fidelity')
        plt.title('Fidelity Improvement')
    else:
        std = None
        cutting = None

    rects1 = plt.bar(index, std, bar_width,
    alpha=opacity,
    color='b',
    label='Standard Mode')

    rects2 = plt.bar(index + bar_width, saturated_cutting, bar_width,
    alpha=opacity,
    color='g',
    label='Cutting Mode, Saturated')

    rects3 = plt.bar(index + 2*bar_width, sametotal_cutting, bar_width,
    alpha=opacity,
    color='r',
    label='Cutting Mode, Sametotal')

    plt.xlabel('Full circuit size')
    plt.xticks(index + bar_width, all_fc_size)
    plt.legend()

    plt.tight_layout()
    plt.savefig('%s_improvement.png'%figname[:-2],dpi=400)
    plt.close()

def read_data(filename):
    f = open(filename, 'rb' )
    plotter_input = pickle.load(f)
    circuit_type = filename.split('/')[2]
    figname = './plots/'+filename.split('/')[-1].replace('_plotter_input','')
    print('plotting',figname)

    hw_qubits = [case[0] for case in plotter_input]
    fc_qubits = [case[1] for case in plotter_input]
    dx = [0.2 for x in plotter_input]
    dy = [0.2 for x in plotter_input]

    for case in plotter_input:
        ground_truth_ce = cross_entropy(target=plotter_input[case]['evaluations']['sv_noiseless'],
        obs= plotter_input[case]['evaluations']['sv_noiseless'])
        qasm_ce = cross_entropy(target=plotter_input[case]['evaluations']['sv_noiseless'],
        obs= plotter_input[case]['evaluations']['qasm'])
        qasm_noise_ce = cross_entropy(target=plotter_input[case]['evaluations']['sv_noiseless'],
        obs= plotter_input[case]['evaluations']['qasm+noise'])
        cutting_ce = cross_entropy(target=plotter_input[case]['evaluations']['sv_noiseless'],
        obs= plotter_input[case]['evaluations']['cutting'])
        ce_percent_change = 100*(qasm_noise_ce - cutting_ce)/(qasm_noise_ce - ground_truth_ce)
        assert ce_percent_change <= 100+1e-10 and ce_percent_change == plotter_input[case]['ce_percent_reduction']
        plotter_input[case]['ce_comparisons'] = (qasm_noise_ce-ground_truth_ce,cutting_ce-ground_truth_ce)

        ground_truth_fid = fidelity(target=plotter_input[case]['evaluations']['sv_noiseless'],
        obs= plotter_input[case]['evaluations']['sv_noiseless'])
        qasm_fid = fidelity(target=plotter_input[case]['evaluations']['sv_noiseless'],
        obs= plotter_input[case]['evaluations']['qasm'])
        qasm_noise_fid = fidelity(target=plotter_input[case]['evaluations']['sv_noiseless'],
        obs= plotter_input[case]['evaluations']['qasm+noise'])
        cutting_fid = fidelity(target=plotter_input[case]['evaluations']['sv_noiseless'],
        obs= plotter_input[case]['evaluations']['cutting'])
        fid_percent_change = 100*(cutting_fid-qasm_noise_fid)/qasm_noise_fid
        if circuit_type == 'bv' or circuit_type=='hwea':
            assert fid_percent_change == plotter_input[case]['fid_percent_improvement']
            assert abs(ground_truth_fid-1)<1e-10 and abs(qasm_fid-1)<1e-10 and qasm_noise_fid<=1 and qasm_noise_fid<=1 and cutting_fid<=1
        plotter_input[case]['fid_comparisons'] = (qasm_noise_fid,cutting_fid)

        print('case {}: ce percentage reduction = {:.3f}, fidelity improvement = {:.3f}, reconstruction time: {:.3e}'.format(case,ce_percent_change,fid_percent_change,plotter_input[case]['uniter_time']))
    print('*'*50)

    best_cc = {}
    for case in plotter_input:
        ce_percent = plotter_input[case]['ce_percent_reduction']
        fid_percent = plotter_input[case]['fid_percent_improvement']
        uniter_time = plotter_input[case]['uniter_time']
        qasm_noise_fid, cutting_fid = plotter_input[case]['fid_comparisons']
        qasm_noise_ce, cutting_ce = plotter_input[case]['ce_comparisons']
        hw, fc = case
        if circuit_type == 'supremacy':
            if (fc in best_cc and ce_percent>best_cc[fc]['ce_percent']) or (fc not in best_cc):
                best_cc[fc] = {'ce_percent':ce_percent,'uniter_time':uniter_time,'best_case':case,'qasm_noise_ce':qasm_noise_ce,'cutting_ce':cutting_ce}
        elif circuit_type == 'bv' or circuit_type=='hwea':
            if (fc in best_cc and fid_percent>best_cc[fc]['fid_percent']) or (fc not in best_cc):
                best_cc[fc] = {'fid_percent':fid_percent,'uniter_time':uniter_time,'best_case':case,'qasm_noise_fid':qasm_noise_fid,'cutting_fid':cutting_fid}
        else:
            raise Exception('Illegal circuit type:',circuit_type)
    [print(best_cc[fc]) for fc in best_cc]
    plot_tradeoff(best_cc,circuit_type,figname)
    plot_heatmap(plotter_input,hw_qubits,fc_qubits,circuit_type,figname)
    return best_cc, hw_qubits, fc_qubits

def get_filename(device_name,circuit_type,shots_mode,evaluation_method):
    dirname = './benchmark_data/{}/'.format(circuit_type)
    if evaluation_method == 'statevector_simulator':
        filename = 'classical_plotter_input_{}_{}.p'.format(device_name,circuit_type)
    elif evaluation_method == 'noisy_qasm_simulator':
        filename = 'quantum_plotter_input_{}_{}_{}.p'.format(device_name,circuit_type,shots_mode)
    else:
        raise Exception('Illegal evaluation method :',evaluation_method)
    return dirname+filename

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MPI evaluator.')
    parser.add_argument('--device-name', metavar='S', type=str,help='which evaluator device input file to run')
    parser.add_argument('--circuit-type', metavar='S', type=str,help='which circuit input file to run')
    parser.add_argument('--evaluation-method', metavar='S', type=str,help='which evaluator backend to use')
    args = parser.parse_args()

    assert args.circuit_type in ['supremacy','hwea','bv','qft','sycamore']

    saturated_filename = get_filename(device_name=args.device_name,circuit_type=args.circuit_type,shots_mode='saturated',evaluation_method=args.evaluation_method)
    sametotal_filename = get_filename(device_name=args.device_name,circuit_type=args.circuit_type,shots_mode='sametotal',evaluation_method=args.evaluation_method)

    dirname = './plots'
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    saturated_best_cc, hw_qubits, fc_qubits = read_data(filename=saturated_filename)
    sametotal_best_cc, hw_qubits, fc_qubits = read_data(filename=sametotal_filename)

    if args.evaluation_method == 'noisy_qasm_simulator':
        evaluation_header = 'quantum'
    elif args.evaluation_method == 'statevector_simulator':
        evaluation_header = 'classical'
    else:
        evaluation_header = None
    figname = '{}/{}_{}_improvement.png'.format(dirname,evaluation_header,args.device_name)
    plot_fid_bar(saturated_best_cc=saturated_best_cc, sametotal_best_cc=sametotal_best_cc,circuit_type=args.circuit_type,figname=figname)
    print('-'*100)