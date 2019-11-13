import pickle
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from helper_fun import cross_entropy

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
            text = im.axes.text(j, i, valfmt(data[i, j], None) if data[i, j]!=0 else 'DNE', **kw)
            texts.append(text)

    return texts

def initialize_dict(cases):
    empty_dict = {}
    for case in cases:
        empty_dict[case] = 0.0
    return empty_dict

if __name__ == '__main__':
    all_files = glob.glob('./benchmark_data/*_plotter_input_*.p')
    for filename in all_files:
        f = open(filename, 'rb' )
        plotter_inputs = []
        while 1:
            try:
                plotter_inputs.append(pickle.load(f))
            except EOFError:
                break
        evaluator_type = filename.split('/')[-1].split('_')[0]
        figname = './plots/'+filename.split('/')[-1].replace('_plotter_input','')

        hw_qubits = [case[0] for case in plotter_inputs[0]]
        fc_qubits = [case[1] for case in plotter_inputs[0]]
        dx = [0.2 for x in plotter_inputs[0]]
        dy = [0.2 for x in plotter_inputs[0]]

        searcher_time_avg = initialize_dict(plotter_inputs[0].keys())
        classical_time_avg = initialize_dict(plotter_inputs[0].keys())
        quantum_time_avg = initialize_dict(plotter_inputs[0].keys())
        uniter_time_avg = initialize_dict(plotter_inputs[0].keys())
        ground_truth_avg = initialize_dict(plotter_inputs[0].keys())
        qasm_avg = initialize_dict(plotter_inputs[0].keys())
        qasm_noise_avg = initialize_dict(plotter_inputs[0].keys())
        hw_fc_avg = initialize_dict(plotter_inputs[0].keys())
        cutting_avg = initialize_dict(plotter_inputs[0].keys())
        percent_change_avg = initialize_dict(plotter_inputs[0].keys())

        for i, plotter_input in enumerate(plotter_inputs):
            print('repetition ',i)
            # Iterate over repetitions
            for case in plotter_input:
                searcher_time_avg[case] += plotter_input[case]['searcher_time']
                classical_time_avg[case] += plotter_input[case]['classical_time']
                quantum_time_avg[case] += plotter_input[case]['quantum_time']
                uniter_time_avg[case] += plotter_input[case]['uniter_time']

                case_ground_truth = cross_entropy(target=plotter_input[case]['evaluations']['sv_noiseless'],
                obs= plotter_input[case]['evaluations']['sv_noiseless'])
                
                case_qasm = cross_entropy(target=plotter_input[case]['evaluations']['sv_noiseless'],
                obs= plotter_input[case]['evaluations']['qasm'])
                
                case_qasm_noise = cross_entropy(target=plotter_input[case]['evaluations']['sv_noiseless'],
                obs= plotter_input[case]['evaluations']['qasm+noise'])

                case_hw_fc = cross_entropy(target=plotter_input[case]['evaluations']['sv_noiseless'],
                obs= plotter_input[case]['evaluations']['hw'])

                case_cutting = cross_entropy(target=plotter_input[case]['evaluations']['sv_noiseless'],
                obs= plotter_input[case]['evaluations']['cutting'])

                case_percent_change = 100*(case_hw_fc - case_cutting)/(case_hw_fc - case_ground_truth)
                # case_percent_change = (case_hw_fc - case_ground_truth)/(case_cutting - case_ground_truth)
                print('case {}: percentage reduction = {}, reconstruction time: {:.3e}'.format(case,
                case_percent_change,plotter_input[case]['uniter_time']))
                assert case_percent_change == plotter_input[case]['percent_reduction']
                
                ground_truth_avg[case] += case_ground_truth
                qasm_avg[case] += case_qasm
                qasm_noise_avg[case] += case_qasm_noise
                hw_fc_avg[case] += case_hw_fc
                cutting_avg[case] += case_cutting
                percent_change_avg[case] += case_percent_change

                # print('case {} reduction:{},time:{}'.format(case,case_percent_change,plotter_input[case]['uniter_time']))
            print('*'*50)
        
        num_repetitions = len(plotter_inputs)
        for dictionary in (searcher_time_avg,classical_time_avg,quantum_time_avg,uniter_time_avg,ground_truth_avg,qasm_avg,qasm_noise_avg,hw_fc_avg,cutting_avg,percent_change_avg):
            for case in dictionary:
                dictionary[case] = dictionary[case]/num_repetitions

        best_cc = {}
        for case in percent_change_avg:
            percent = percent_change_avg[case]
            uniter_time = uniter_time_avg[case]
            hw, fc = case
            if (fc in best_cc and percent>best_cc[fc][0]) or (fc not in best_cc):
                best_cc[fc] = (percent,uniter_time,case)
        [print('Full circuit size {:d}. Best case {}. Cross entropy reduction = {:.3f}%. Reconstruction time = {:.3e} seconds.'.format(fc,best_cc[fc][2],best_cc[fc][0],best_cc[fc][1])) for fc in best_cc]

        plt.figure(figsize=(10,5))
        plt.subplot(121)
        plt.plot([fc for fc in best_cc], [best_cc[fc][0] for fc in best_cc], 'bX')
        plt.xlabel('Number of qubits')
        plt.ylabel('Cross entropy reduction (%)')
        plt.xticks([x for x in best_cc])
        plt.ylim(0,100)
        plt.subplot(122)
        plt.plot([fc for fc in best_cc], [best_cc[fc][1] for fc in best_cc], 'r*')
        plt.xlabel('Number of qubits')
        plt.ylabel('Reconstruction time (s)')
        plt.xticks([x for x in best_cc])
        plt.tight_layout()
        plt.savefig('%s_tradeoff.png'%figname[:-2],dpi=400)
        plt.close()

        print('plotting %s, %d times average'%(figname,len(plotter_inputs)))

        fig_scale = 4.5
        fig = plt.figure(figsize=(3*fig_scale,2*fig_scale))
        ax1 = fig.add_subplot(231, projection='3d')
        ax1.bar3d(hw_qubits, fc_qubits, np.zeros(len(plotter_input)), dx, dy, [searcher_time_avg[case] for case in searcher_time_avg])
        ax1.set_xlabel('hardware qubits')
        ax1.set_ylabel('full circuit qubits')
        ax1.set_zlabel('searcher time (seconds)')
        ax1 = fig.add_subplot(232, projection='3d')
        ax1.bar3d(hw_qubits, fc_qubits, np.zeros(len(plotter_input)), dx, dy, [classical_time_avg[case] for case in classical_time_avg])
        ax1.set_zlim3d(0, 1.2*max(classical_time_avg.values())+1)
        ax1.set_xlabel('hardware qubits')
        ax1.set_ylabel('full circuit qubits')
        ax1.set_zlabel('classical evaluator time (seconds)')
        ax1 = fig.add_subplot(233, projection='3d')
        ax1.bar3d(hw_qubits, fc_qubits, np.zeros(len(plotter_input)), dx, dy, [quantum_time_avg[case] for case in quantum_time_avg])
        ax1.set_zlim3d(0, 1.2*max(quantum_time_avg.values())+1)
        ax1.set_xlabel('hardware qubits')
        ax1.set_ylabel('full circuit qubits')
        ax1.set_zlabel('quantum evaluator time (seconds)')
        ax1 = fig.add_subplot(234, projection='3d')
        ax1.bar3d(hw_qubits, fc_qubits, np.zeros(len(plotter_input)), dx, dy, [uniter_time_avg[case] for case in uniter_time_avg])
        ax1.set_xlabel('hardware qubits')
        ax1.set_ylabel('full circuit qubits')
        ax1.set_zlabel('reconstructor time (seconds)')
        ax1 = fig.add_subplot(235, projection='3d')
        ax1.bar3d(hw_qubits, fc_qubits, np.zeros(len(plotter_input)), dx, dy, [percent_change_avg[case] for case in percent_change_avg])
        ax1.set_zlim3d(min(0,1.2*min(percent_change_avg.values())), max(0,1.2*max(percent_change_avg.values())))
        ax1.set_xlabel('hardware qubits')
        ax1.set_ylabel('full circuit qubits')
        ax1.set_zlabel('cross entropy gap reduction due to cutting (%)')
        # pickle.dump(fig,open('%s'%figname, 'wb'))
        plt.savefig('%s.png'%figname[:-2],dpi=400)
        print('-'*100)
        plt.close()

        hw_qubits_unique = list(np.unique(hw_qubits))
        fc_qubits_unique = list(np.unique(fc_qubits))
        fc_qubits_unique.sort(reverse=True)
        reduction_map = np.zeros((len(fc_qubits_unique), len(hw_qubits_unique)))
        for fc_qubit in fc_qubits_unique:
            for hw_qubit in hw_qubits_unique:
                case = (hw_qubit,fc_qubit)
                percent = percent_change_avg[case] if case in percent_change_avg else 0
                row_idx = fc_qubits_unique.index(fc_qubit)
                col_idx = hw_qubits_unique.index(hw_qubit)
                # print('case {}, position {}, percent = {}'.format(case,(row_idx, col_idx),percent))
                reduction_map[row_idx, col_idx] = percent

        fig, ax = plt.subplots(figsize=(10,10))

        im, cbar = heatmap(reduction_map, fc_qubits_unique, hw_qubits_unique, ax=ax,
                        cmap="YlGn", cbarlabel="Cross Entropy Loss Reduction [%]")
        texts = annotate_heatmap(im, valfmt="{x:.3f} %")
        ax.set_xlabel('Hardware qubits')
        ax.set_ylabel('Full circuit qubits')

        fig.tight_layout()
        plt.savefig('%s_ce_map.png'%figname[:-2],dpi=400)
        plt.close()