import pickle
import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from utils.helper_fun import cross_entropy, fidelity, get_filename, read_file
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
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom",size=18)

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
        spine.set_visible(True)
        spine.set_linewidth(3)
        spine.set_color('0.9')

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
    # if threshold is not None:
    #     threshold = im.norm(threshold)
    # else:
    #     threshold = im.norm(data.max())/2.0

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
            # kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text_to_fill = valfmt(data[i, j], None) if data[i, j]!=0 else '-'
            text = im.axes.text(j, i, text_to_fill, fontsize=14, color='white', **kw)
            text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                       path_effects.Normal()])
            texts.append(text)

    return texts

def plot_tradeoff(best_cc,circuit_type,filename,mitigated):
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.xlabel('Number of qubits',size=12)
    if circuit_type == 'supremacy':
        plt.plot([fc for fc in best_cc], [best_cc[fc]['ce_percent'] for fc in best_cc], 'bX')
        plt.ylabel('\u0394H reduction (%)',size=12)
        plt.ylim(0,100)
    elif circuit_type == 'bv' or circuit_type=='hwea':
        plt.plot([fc for fc in best_cc], [best_cc[fc]['fid_percent'] for fc in best_cc], 'bX')
        plt.ylabel('Fidelity improvement (%)',size=12)
    plt.xticks([x for x in best_cc])
    plt.subplot(122)
    plt.plot([fc for fc in best_cc], [best_cc[fc]['reconstructor_time'] for fc in best_cc], 'r*')
    plt.xlabel('Number of qubits',size=12)
    plt.ylabel('Reconstruction time (s)',size=12)
    plt.xticks([x for x in best_cc])
    plt.tight_layout()
    plt.savefig('%s_tradeoff%s.png'%(filename[:-2],'_mitigated' if mitigated else ''),dpi=400)
    plt.close()

def plot_heatmap(plotter_input,hw_qubits,fc_qubits,circuit_type,filename,mitigated):
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

    data =np.ma.masked_where(reduction_map==0, reduction_map)

    im, cbar = heatmap(data, fc_qubits_unique, hw_qubits_unique, ax=ax,
                    cmap="YlGn", cbarlabel="\u0394H Reduction, higher is better [%]" if circuit_type == 'supremacy' or circuit_type == 'qft' else "Fidelity Improvement [%]")
    texts = annotate_heatmap(im, valfmt="{x:.1f}")
    ax.set_xlabel('Hardware qubits',fontsize=18,labelpad=10)
    ax.set_ylabel('Full circuit qubits',fontsize=18,labelpad=10)

    fig.tight_layout()
    plt.savefig('%s_heatmap%s.png'%(filename[:-2],'_mitigated' if mitigated else ''),dpi=400)
    plt.close()

def plot_fid_bar(saturated_best_cc,sametotal_best_cc,circuit_type,dirname,device_name,mitigated):
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
            std.append(saturated_best_cc[fc_size]['hw_ce'] if circuit_type=='supremacy' else saturated_best_cc[fc_size]['hw_fid'])
            has_std = True
            saturated_cutting.append(saturated_best_cc[fc_size]['cutting_ce'] if circuit_type=='supremacy' else saturated_best_cc[fc_size]['cutting_fid'])
        else:
            saturated_cutting.append(-1)
        if fc_size in sametotal_best_cc:
            if not has_std:
                std.append(sametotal_best_cc[fc_size]['hw_ce'] if circuit_type=='supremacy' else sametotal_best_cc[fc_size]['hw_fid'])
            sametotal_cutting.append(sametotal_best_cc[fc_size]['cutting_ce'] if circuit_type=='supremacy' else sametotal_best_cc[fc_size]['cutting_fid'])
        else:
            sametotal_cutting.append(-1)

    n_groups = len(all_fc_size)
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.2
    opacity = 0.8

    if circuit_type == 'supremacy':
        ax.set_ylabel('\u0394H, lower is better',size=12)
        # plt.title('\u0394H Reduction')
    elif circuit_type == 'bv' or circuit_type=='hwea':
        ax.set_ylim(0,1)
        ax.set_ylabel('Fidelity, higher is better',size=12)
        # plt.title('Fidelity Improvement')
    else:
        std = None
        cutting = None

    rects1 = ax.bar(index, std, bar_width,
    alpha=opacity,
    color='b',
    label='Standard Mode')

    rects2 = ax.bar(index + bar_width, saturated_cutting, bar_width,
    alpha=opacity,
    color='g',
    label='Cutting Mode, Saturated')

    rects3 = ax.bar(index + bar_width + bar_width, sametotal_cutting, bar_width,
    alpha=opacity,
    color='r',
    label='Cutting Mode, Sametotal')

    ax.set_xlabel('Full circuit size',size=12)
    plt.xticks(index + bar_width, all_fc_size)
    plt.legend()

    plt.tight_layout()
    plt.savefig('%s/%s_shots_comparison%s.png'%(dirname,device_name,'_mitigated' if mitigated else ''),dpi=400)
    plt.close()

def process_data(filename,circuit_type,mitigated):
    plotter_input = read_file(filename)
    print('Processing',filename)

    hw_qubits = [case[0] for case in plotter_input]
    fc_qubits = [case[1] for case in plotter_input]

    for case in plotter_input:
        ground_truth_ce = cross_entropy(target=plotter_input[case]['sv'],
        obs= plotter_input[case]['sv'])
        qasm_ce = cross_entropy(target=plotter_input[case]['sv'],
        obs= plotter_input[case]['qasm'])
        qasm_noise_ce = cross_entropy(target=plotter_input[case]['sv'],
        obs= plotter_input[case]['qasm+noise'])
        hw_ce = cross_entropy(target=plotter_input[case]['sv'],
        obs= plotter_input[case]['%shw'%('mitigated_' if mitigated else '')])
        cutting_ce = cross_entropy(target=plotter_input[case]['sv'],
        obs= plotter_input[case]['%scutting'%('mitigated_' if mitigated else '')])
        ce_percent_change = 100*(hw_ce - cutting_ce)/hw_ce
        assert ce_percent_change <= 100+1e-10
        plotter_input[case]['ce_comparisons'] = (hw_ce,cutting_ce)
        plotter_input[case]['ce_percent_reduction'] = ce_percent_change

        ground_truth_fid = fidelity(target=plotter_input[case]['sv'],
        obs= plotter_input[case]['sv'])
        qasm_fid = fidelity(target=plotter_input[case]['sv'],
        obs= plotter_input[case]['qasm'])
        qasm_noise_fid = fidelity(target=plotter_input[case]['sv'],
        obs= plotter_input[case]['qasm+noise'])
        hw_fid = fidelity(target=plotter_input[case]['sv'],
        obs= plotter_input[case]['%shw'%('mitigated_' if mitigated else '')])
        cutting_fid = fidelity(target=plotter_input[case]['sv'],
        obs= plotter_input[case]['%scutting'%('mitigated_' if mitigated else '')])
        fid_percent_change = 100*(cutting_fid-hw_fid)/hw_fid
        if circuit_type == 'bv' or circuit_type=='hwea':
            assert abs(ground_truth_fid-1)<1e-10 and abs(qasm_fid-1)<1e-10 and qasm_noise_fid<=1 and qasm_noise_fid<=1 and cutting_fid<=1
        plotter_input[case]['fid_comparisons'] = (hw_fid,cutting_fid)
        plotter_input[case]['fid_percent_improvement'] = fid_percent_change

        print('case {}: ce percentage reduction = {:.3f}, fidelity improvement = {:.3f}, reconstruction time: {:.3e}'.format(case,ce_percent_change,fid_percent_change,plotter_input[case]['reconstructor_time']))
    print('*'*25,'Best Cases','*'*25)

    best_cc = {}
    for case in plotter_input:
        ce_percent = plotter_input[case]['ce_percent_reduction']
        fid_percent = plotter_input[case]['fid_percent_improvement']
        reconstructor_time = plotter_input[case]['reconstructor_time']
        hw_fid, cutting_fid = plotter_input[case]['fid_comparisons']
        hw_ce, cutting_ce = plotter_input[case]['ce_comparisons']
        hw, fc = case
        if circuit_type == 'supremacy':
            if (fc in best_cc and ce_percent>best_cc[fc]['ce_percent']) or (fc not in best_cc):
                best_cc[fc] = {'ce_percent':ce_percent,'reconstructor_time':reconstructor_time,'best_case':case,'hw_ce':hw_ce,'cutting_ce':cutting_ce}
        elif circuit_type == 'bv' or circuit_type=='hwea':
            if (fc in best_cc and fid_percent>best_cc[fc]['fid_percent']) or (fc not in best_cc):
                best_cc[fc] = {'fid_percent':fid_percent,'reconstructor_time':reconstructor_time,'best_case':case,'hw_fid':hw_fid,'cutting_fid':cutting_fid}
        else:
            raise Exception('Illegal circuit type:',circuit_type)
    [print(best_cc[fc]) for fc in best_cc]
    print('*'*50)
    return plotter_input, best_cc, hw_qubits, fc_qubits

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MPI evaluator.')
    parser.add_argument('--device-name', metavar='S', type=str,help='which evaluator device input file to run')
    parser.add_argument('--circuit-type', metavar='S', type=str,help='which circuit input file to run')
    parser.add_argument('--evaluation-method', metavar='S', type=str,help='which evaluator backend to use')
    args = parser.parse_args()

    assert args.circuit_type in ['supremacy','hwea','bv','qft','sycamore']

    print('-'*50,'Plot','-'*50,flush=True)

    for mitigated in [True, False]:
        dirname, saturated_filename = get_filename(experiment_name='hardware',circuit_type=args.circuit_type,device_name=args.device_name,field='plotter_input',evaluation_method=args.evaluation_method,shots_mode='saturated')
        saturated_plotter_input, saturated_best_cc, saturated_hw_qubits, saturated_fc_qubits = process_data(filename=dirname+saturated_filename,circuit_type=args.circuit_type,mitigated=mitigated)
        plot_tradeoff(best_cc=saturated_best_cc,circuit_type=args.circuit_type,filename=dirname+saturated_filename,mitigated=mitigated)
        plot_heatmap(plotter_input=saturated_plotter_input,hw_qubits=saturated_hw_qubits,fc_qubits=saturated_fc_qubits,circuit_type=args.circuit_type,filename=dirname+saturated_filename,mitigated=mitigated)

        dirname, sametotal_filename = get_filename(experiment_name='hardware',circuit_type=args.circuit_type,device_name=args.device_name,field='plotter_input',evaluation_method=args.evaluation_method,shots_mode='sametotal')
        sametotal_plotter_input, sametotal_best_cc, sametotal_hw_qubits, sametotal_fc_qubits = process_data(filename=dirname+sametotal_filename,circuit_type=args.circuit_type,mitigated=mitigated)
        plot_tradeoff(best_cc=sametotal_best_cc,circuit_type=args.circuit_type,filename=dirname+sametotal_filename,mitigated=mitigated)
        plot_heatmap(plotter_input=sametotal_plotter_input,hw_qubits=sametotal_hw_qubits,fc_qubits=sametotal_fc_qubits,circuit_type=args.circuit_type,filename=dirname+sametotal_filename,mitigated=mitigated)
        
        plot_fid_bar(saturated_best_cc=saturated_best_cc, sametotal_best_cc=sametotal_best_cc,circuit_type=args.circuit_type,dirname=dirname,device_name=args.device_name,mitigated=mitigated)
        print('-'*100)