import pickle
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from helper_fun import cross_entropy, fidelity
import os

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

def plot_tradeoff(best_cc,circuit_type):
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

def plot_heatmap(plotter_input,hw_qubits,fc_qubits,circuit_type):
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
    texts = annotate_heatmap(im, valfmt="{x:.3f} %")
    ax.set_xlabel('Hardware qubits')
    ax.set_ylabel('Full circuit qubits')

    metric_type = 'ce' if (circuit_type == 'supremacy' or circuit_type == 'qft') else 'fid'
    fig.tight_layout()
    plt.savefig('{}_{}_map.png'.format(figname[:-2],metric_type),dpi=1000)
    plt.close()

def plot_fid_bar(best_cc,circuit_type):
    n_groups = len(best_cc)
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    if circuit_type == 'supremacy':
        vanilla = [best_cc[fc]['hw_fc_ce'] for fc in best_cc]
        cutting = [best_cc[fc]['cutting_ce'] for fc in best_cc]
        plt.ylabel('\u0394H')
        plt.title('\u0394H Reduction')
    elif circuit_type == 'bv' or circuit_type=='hwea':
        vanilla = [best_cc[fc]['hw_fc_fid'] for fc in best_cc]
        cutting = [best_cc[fc]['cutting_fid'] for fc in best_cc]
        plt.ylim(0,1)
        plt.ylabel('Fidelity')
        plt.title('Fidelity Improvement')
    else:
        vanilla = None
        cutting = None

    rects1 = plt.bar(index, vanilla, bar_width,
    alpha=opacity,
    color='b',
    label='Vanilla')

    rects2 = plt.bar(index + bar_width, cutting, bar_width,
    alpha=opacity,
    color='g',
    label='Cutting')

    plt.xlabel('Full circuit size')
    plt.xticks(index + bar_width, list(best_cc.keys()))
    plt.legend()

    plt.tight_layout()
    plt.savefig('%s_improvement.png'%figname[:-2],dpi=400)
    plt.close()


if __name__ == '__main__':
    dirname = './plots'
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    all_files = glob.glob('./benchmark_data/*/*_plotter_input_*.p')
    for filename in all_files:
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
            hw_fc_ce = cross_entropy(target=plotter_input[case]['evaluations']['sv_noiseless'],
            obs= plotter_input[case]['evaluations']['hw'])
            cutting_ce = cross_entropy(target=plotter_input[case]['evaluations']['sv_noiseless'],
            obs= plotter_input[case]['evaluations']['cutting'])
            ce_percent_change = 100*(hw_fc_ce - cutting_ce)/(hw_fc_ce - ground_truth_ce)
            assert ce_percent_change <= 100+1e-10 and ce_percent_change == plotter_input[case]['ce_percent_reduction']
            plotter_input[case]['ce_comparisons'] = (hw_fc_ce-ground_truth_ce,cutting_ce-ground_truth_ce)

            ground_truth_fid = fidelity(target=plotter_input[case]['evaluations']['sv_noiseless'],
            obs= plotter_input[case]['evaluations']['sv_noiseless'])
            qasm_fid = fidelity(target=plotter_input[case]['evaluations']['sv_noiseless'],
            obs= plotter_input[case]['evaluations']['qasm'])
            qasm_noise_fid = fidelity(target=plotter_input[case]['evaluations']['sv_noiseless'],
            obs= plotter_input[case]['evaluations']['qasm+noise'])
            hw_fc_fid = fidelity(target=plotter_input[case]['evaluations']['sv_noiseless'],
            obs= plotter_input[case]['evaluations']['hw'])
            cutting_fid = fidelity(target=plotter_input[case]['evaluations']['sv_noiseless'],
            obs= plotter_input[case]['evaluations']['cutting'])
            fid_percent_change = 100*(cutting_fid-hw_fc_fid)/hw_fc_fid
            if circuit_type == 'bv' or circuit_type=='hwea':
                assert fid_percent_change == plotter_input[case]['fid_percent_improvement']
                assert abs(ground_truth_fid-1)<1e-10 and abs(qasm_fid-1)<1e-10 and qasm_noise_fid<=1 and hw_fc_fid<=1 and cutting_fid<=1
            plotter_input[case]['fid_comparisons'] = (hw_fc_fid,cutting_fid)

            print('case {}: ce percentage reduction = {:.3f}, fidelity improvement = {:.3f}, reconstruction time: {:.3e}'.format(case,ce_percent_change,fid_percent_change,plotter_input[case]['uniter_time']))
        print('*'*50)

        best_cc = {}
        for case in plotter_input:
            ce_percent = plotter_input[case]['ce_percent_reduction']
            fid_percent = plotter_input[case]['fid_percent_improvement']
            uniter_time = plotter_input[case]['uniter_time']
            hw_fc_fid, cutting_fid = plotter_input[case]['fid_comparisons']
            hw_fc_ce, cutting_ce = plotter_input[case]['ce_comparisons']
            hw, fc = case
            if circuit_type == 'supremacy':
                if (fc in best_cc and ce_percent>best_cc[fc]['ce_percent']) or (fc not in best_cc):
                    best_cc[fc] = {'ce_percent':ce_percent,'uniter_time':uniter_time,'best_case':case,'hw_fc_ce':hw_fc_ce,'cutting_ce':cutting_ce}
            elif circuit_type == 'bv' or circuit_type=='hwea':
                if (fc in best_cc and fid_percent>best_cc[fc]['fid_percent']) or (fc not in best_cc):
                    best_cc[fc] = {'fid_percent':fid_percent,'uniter_time':uniter_time,'best_case':case,'hw_fc_fid':hw_fc_fid,'cutting_fid':cutting_fid}
            else:
                raise Exception('Illegal circuit type:',circuit_type)
        [print(best_cc[fc]) for fc in best_cc]

        plot_tradeoff(best_cc,circuit_type)
        plot_3d_bar(plotter_input,hw_qubits,fc_qubits)
        plot_heatmap(plotter_input,hw_qubits,fc_qubits,circuit_type)
        plot_fid_bar(best_cc,circuit_type)
        print('-'*100)