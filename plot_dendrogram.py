# coding:utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from scipy.cluster.hierarchy import dendrogram, linkage

sys.setrecursionlimit(2000)


""" FUNCTIONS """


def get_cmap(n, name='hsv'):
    """ Returns a function that maps each index
    in 0, 1, ..., n-1 to a distinct RGB color;
    the keyword argument name must be a standard mpl colormap name. """
    return plt.cm.get_cmap(name, n)


def get_iso(state):
    """ This function returns the ISO code
    of the given state. """
    return len_states[np.where(len_states == state)[0][0], 2]


def get_len(state):
    """ This function returns the number of counties
    in the given state. """
    return int(len_states[np.where(len_states == state)[0][0], 1])


def loop_plot_dendrogram_by_state():
    """ This function iterates over the states and methods and
    weight functions to create 4 dendrograms to each state. """
    for state in states:
        if state != 'District of Columbia':
            plot_dendrogram_by_state(state)


def plot_dendrogram_by_mat(mat, weight_fc='', method='', state='',
                           label=np.array([])):
    """  plot_dendrogram function
     input: state, method
            state: symmetric n-square matrix of distances
            method: string to identify if reciprocal or non-reciprocal
            label: (optional) list of labels to be used as nodes in dendrogram
    output: pdf file of dendrogram of distance matrix mat """

    t = np.triu(mat, k=1)
    tu = t[t > 0]
    z = linkage(tu)
    fig_width = 15

    if len(mat) > 200:
        fig_width = 25
    else:
        if len(mat) > 100:
            fig_width = 20
    if state == 'USA':
        fig_width = 400

    fig = plt.figure(figsize=(fig_width, 8))
    fig.add_subplot(111)
    plt.title('Hierarchical Clustering Dendrogram of %s (wf: %s, meth: %s)' %
              (state, weight_fc, method))
    dendrogram(
        z,
        color_threshold=0.9 * max(z[:, 2]),
        leaf_rotation=90,
        leaf_font_size=7,
        labels=label,
    )
    ax = plt.gca()
    xlbls = ax.get_xmajorticklabels()
    label_colors = {i: j for i, j in zip(label, colors)}
    for lbl in xlbls:
        lbl.set_color(label_colors[lbl.get_text()])
    plt.savefig('dendrograms/%s_u%s_%s.pdf' % (weight_fc, method, state),
                bbox_inches="tight")
    plt.close(fig)


def plot_dendrogram_by_state(state):
    print 'Creating all dendrograms for %s' % state
    for method in ['R', 'NR']:
        for weight_fc in ['col', 'row']:
            mat = np.loadtxt('ultra_metric/%s_u%s_%s.csv' % (weight_fc, method,
                             state.replace(' ', '_')), dtype=float,
                             delimiter=',')
            plot_dendrogram_by_mat(mat, weight_fc=weight_fc, method=method,
                                   state=state, label=dict_states_cts[state])


def plot_dendrogram_national():
    """ This function creates the national migration flow dendrogram. """
    print 'Creating the national migration dendrogram'
    for method in ['NR']:
        for weight_fc in ['col', 'row']:
            mat = np.loadtxt('ultra_metric/%s_u%s.csv' % (weight_fc, method),
                             dtype=float, delimiter=',')
            plot_dendrogram_by_mat(mat, weight_fc=weight_fc, method=method,
                                   state='USA', label=all_counties)
            #break
        #break


def use_pandas():
    global dict_states_cts, states

    print 'Defining variables dict_states_cts, states'

    df = pd.read_csv(raw_data_usa_file, encoding='latin1', header=None,
                     delim_whitespace=False, index_col=None, sep=',',
                     names=used_fields)
    data = df[df[7] != '-'][used_fields]
    states = data[4].unique().tolist()[:-9]

    """ create dictionary with counties of each state
    key: state
    value: list of counties of state """
    dict_states_cts = {}
    for state in states:
        tmp = data[data[4] == state][5].unique()
        tmp = [cty.replace(' County', '').replace(' Municipio', '').replace(
                           ' city', '').replace(' Parish', '') for cty in tmp]
        dict_states_cts[state] = tmp


""" MAIN CODE """

raw_data_usa_file = 'raw_data_usa.csv'
len_states = np.loadtxt('prepare_data_states_num_counties.csv', dtype=str,
                        delimiter=',')
used_fields = [4, 5, 6, 7, 8, 10]

use_pandas()

""" List of ISO codes + county names,
    with commom words deleted to make them shorter """
all_counties = []
for state in states:
    for county in dict_states_cts[state]:
        tmp = u'%s %s' % (get_iso(state), county)
        tmp = tmp.replace(' County', '').replace(' Municipio', '').replace(
                          ' city', '').replace(' Parish', '').replace(
                          ' Census Area', '')
        all_counties.append(tmp)

""" Creates a list of colors; one color to each state """
colors = []
cmap = get_cmap(len(states))
for i, state in enumerate(states):
    colors += [cmap(i)] * get_len(state)

# required files: 'ultra_metric/{col,row}_u{R,NR}_{state}.csv'
# loop_plot_dendrogram_by_state()

# required files: 'ultra_metric/{col,row}_u{R,NR}.csv'
# plot_dendrogram_national()
