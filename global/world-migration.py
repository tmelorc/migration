# coding:utf-8

""" Author: Thiago de Melo """

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage

""" FUNCTIONS """


def delete_zero_col_row(mat):
    """ This function delete null row/columns. """
    idx = []
    for i in range(len(mat)):
        if max(mat[i]) == 0:
            idx.append(i)

    mat = np.delete(mat, idx, 0)
    mat = np.delete(mat, idx, 1)
    return mat


def max_prod(mat1, mat2):
    """ max_prod function
     input: mat1, mat2 square matrices of same order
    output: square matrix with (i,j) entry equals to
            min_max(mat1, mat2, i, j) """
    n = len(mat1)
    tmp = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            tmp[i, j] = min_max(mat1, mat2, i, j)
            tmp[j, i] = min_max(mat2, mat1, j, i)
    return tmp


def min_max(mat1, mat2, i, j):
    """ min_max function
     input: mat1, mat2 square matrices of same order
            i, j positive integers
    output: the minimum of the maximum pointwise vector
            from mat1(row i) and mat2(col j) """
    return min(np.maximum(mat1[i, :], mat2[:, j]))


def plot_dendrogram_by_mat(mat, weight_fc='', method='', label=np.array([])):
    """  plot_dendrogram function
     input: mat
     optional: weight_fc, method, label
        mat: symmetric n-square matrix of distances
        weight_fc: (optional) function to be used (row or col)
        method: (optional) string to identify if reciprocal or non-reciprocal
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

    fig = plt.figure(figsize=(fig_width, 8))
    fig.add_subplot(111)
    plt.title('Hierarchical Clustering Dendrogram (wf: %s, meth: %s)' %
              (weight_fc, method))
    dendrogram(
        z,
        color_threshold=0.9 * max(z[:, 2]),
        leaf_rotation=-90,
        leaf_font_size=7,
        labels=label,
    )
    ax = plt.gca()
    xlbls = ax.get_xmajorticklabels()
    label_colors = {i: j for i, j in zip(label, colors)}
    for lbl in xlbls:
        lbl.set_color(label_colors[lbl.get_text()])
    plt.savefig('%s_u%s.pdf' % (weight_fc, method), bbox_inches="tight")
    plt.close(fig)


def plot_dendrogram_international():
    """ This function creates the international migration flow dendrograms. """
    print 'Creating the international migration dendrogram'
    for method in ['R', 'NR']:
        for weight_fc in ['col', 'row']:
            mat = np.loadtxt('%s_u%s.csv' % (weight_fc, method),
                             dtype=float, delimiter=',')
            plot_dendrogram_by_mat(mat, weight_fc=weight_fc, method=method,
                                   label=iso_codes)


def power(mat):
    """ power function
    computes the iterated max_prod
     input: mat n-squared matrix
    output: mat**(n-1) """
    tmp = mat
    n = len(tmp)
    k = 1
    while 2 ** (k - 1) < n:
        # print k,2**k, n
        tmp = max_prod(tmp, tmp)
        k += 1
    return tmp


def u_nr(mat, output_file='uNR.csv'):
    """ non-reciprocal ultra metric function
     input: mat squared matrix
    output: ultra metric matrix by means of
            non-reciprocal clustering method """
    print 'Computing non-reciprocal ultra metric. Wait... ',
    tmp1 = power(mat)
    tmp2 = tmp1.T
    tmp = np.maximum(tmp1, tmp2)
    np.savetxt(output_file.replace(' ', '_'), tmp, fmt='%f',
               delimiter=',')
    print 'saved in %s' % output_file.replace(' ', '_')
    return tmp


def u_r(mat, output_file='uR.csv'):
    """ reciprocal ultra metric function
     input: mat squared matrix
    output: ultra metric matrix by means of
            reciprocal clustering method """
    print 'Computing     reciprocal ultra metric. Wait... ',
    a_bar = np.maximum(mat, mat.T)

    tmp = power(a_bar)
    np.savetxt(output_file.replace(' ', '_'), tmp, fmt='%f',
               delimiter=',')
    print 'saved in %s' % output_file.replace(' ', '_')
    return tmp


def ultrametric_international(mat):
    """ This function computes the reciprocal and non reciprocal ultrametric
    for the global migration flow around the world. """
    w_mat1, w_mat2 = weight_fc(mat)
    u_r(w_mat1, output_file='col_uR.csv')
    u_nr(w_mat1, output_file='col_uNR.csv')
    u_r(w_mat2, output_file='row_uR.csv')
    u_nr(w_mat2, output_file='row_uNR.csv')


def weight_fc(mat):
    """ weight function
     input: 1 square matrix
    output: 2 square matrices """
    # global i, j, n
    n = len(mat)
    tmp1 = np.zeros((n, n), dtype=float)
    tmp2 = np.zeros((n, n), dtype=float)
    for j in range(n):
        sum_col = max(mat[:, j].sum(), 1)
        sum_row = max(mat[j, :].sum(), 1)
        for i in range(n):
            tmp1[i, j] = 1 - float(mat[i, j]) / sum_col
            tmp2[i, j] = 1 - float(mat[i, j]) / sum_row
        tmp1[j, j] = 0
        tmp2[j, j] = 0
    # del i, j, n
    return tmp1, tmp2


""" MAIN CODE """

""" Data from database: Global Bilateral Migration
    Last Updated: 06/28/2011 """
raw_data_file = 'worldbankdata.csv'
df = pd.read_csv(raw_data_file, header=0, delim_whitespace=False,
                 index_col=0, sep=',')
iso_codes = df['Country Origin Code'].unique()
iso_codes_continents = np.loadtxt('codes.csv', dtype=str, delimiter=',')
iso_continents = sorted(list(set(iso_codes_continents[:, 1])))
num_countries = len(iso_codes)
num_continents = len(iso_continents)
migration_matrix = df['value'].values.astype(float).reshape(num_countries,
                                                            num_countries)
migration_matrix = delete_zero_col_row(migration_matrix)
np.savetxt('migration_matrix.csv', migration_matrix, fmt='%d', delimiter=',')


""" Creates a list of colors; one color to each region.
['AF', 'AS', 'CA', 'EU', 'NA', 'OC', 'SA'] """
colors = []
palette = ['orange', 'red', 'black', 'green', 'cyan', 'magenta', 'blue']
for i, country in enumerate(iso_codes):
    cont = iso_codes_continents[i, 1]
    j = iso_continents.index(cont)
    colors.append(palette[j])

ultrametric_international(migration_matrix)

plot_dendrogram_international()

""" END OF FILE """
