# coding:utf-8

import numpy as np
from itertools import combinations
from my_shortest_path import dijkstra as dj

""" FUNCTIONS """


def loop_ultrametric_inner_state():
    """ Iterates the ultrametric_inner_state()
    for all states except District of Columbia. """
    for state in states:
        if state != 'District of Columbia':
            ultrametric_inner_state(state)


def max_prod(mat1, mat2=''):
    """ max_prod function
     input: mat1, mat2 square matrices of same order
    output: square matrix with (i,j) entry equals to
            min_max(mat1, mat2, i, j) """
    N = len(mat1)
    #tmp = np.zeros((n, n), dtype=float)
    '''
    for i in range(n):
        print 'max_prod: ', i
        for j in range(i, n):
            tmp[i, j] = min_max(mat1, mat2, i, j)
            tmp[j, i] = min_max(mat2, mat1, j, i)
    '''
    tmp = np.asarray([min_max(mat1, p) for p in pairs])
    tri = np.zeros((N, N), dtype=int)
    tri[np.triu_indices(N, 1)] = tmp
    tri[np.tril_indices(N, -1)] = tmp
    #np.savetxt('kk.csv', tri, fmt='%d', delimiter=',')
    return tri


def min_max(mat, p):
    """ min_max function
     input: mat1, mat2 square matrices of same order
            i, j positive integers
    output: the minimum of the maximum pointwise vector
            from mat1(row i) and mat2(col j) """
    i, j = p[0], p[1]
    return min(np.maximum(mat[i, :], mat[:, j]))


def power(mat):
    """ power function
    computes the iterated max_prod
     input: mat n-squared matrix
    output: mat**(n-1) """
    tmp = mat
    n = len(tmp)
    k = 1
    while 2 ** (k - 1) < n:
        print k,2**k, n
        tmp = max_prod(tmp)
        k += 1
    return tmp


def u_nr(mat, output_file='uNR.csv'):
    """ non-reciprocal ultra metric function
     input: mat squared matrix
    output: ultra metric matrix by means of
            non-reciprocal clustering method """
    print 'Computing non-reciprocal ultra metric. Wait... ',
    dj_mat = dj(mat)
    tmp = np.maximum(dj_mat, dj_mat.T)
    #tmp1 = power(mat)
    #tmp2 = tmp1.T
    #tmp = np.maximum(tmp1, tmp2)
    np.savetxt('ultra_metric/' + output_file.replace(' ', '_'), tmp, fmt='%f',
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
    tmp = dj(a_bar)
    np.savetxt('ultra_metric/' + output_file.replace(' ', '_'), tmp, fmt='%f',
               delimiter=',')
    print 'saved in %s' % output_file.replace(' ', '_')
    return tmp


def ultrametric_inner_state(state):
    """ This function computes the reciprocal and non reciprocal ultrametric
    for the inner migration flow in state. """
    print '\n** %s' % state
    mat = np.loadtxt('migration_matrices/mat_%s_to_%s.csv' % (state, state),
                     dtype=float, delimiter=',')
    
    w_mat1, w_mat2 = weight_fc(mat)
    u_r(w_mat1, output_file='col_uR_%s.csv' % state)
    u_nr(w_mat1, output_file='col_uNR_%s.csv' % state)
    u_r(w_mat2, output_file='row_uR_%s.csv' % state)
    u_nr(w_mat2, output_file='row_uNR_%s.csv' % state)

def ultrametric_national():
    """ This function computes the both reciprocal and non-reciprocal
    ultrametric for the national migration flow.
    This function takes hours to finish if matrix dimension is too big. """
    global pairs
    print 'This function needs time to finish.'
    mat = np.loadtxt(migration_matrix_file, dtype=float, delimiter=',')
    pairs = list(combinations(range(mat.shape[0]), 2))
    w_mat1, w_mat2 = weight_fc(mat)
    u_r(w_mat1, output_file='col_uR.csv')
    u_nr(w_mat1, output_file='col_uNR.csv')
    u_r(w_mat2, output_file='row_uR.csv')
    u_nr(w_mat2, output_file='row_uNR.csv')


def weight_fc(mat):
    """ weight function
     input: 1 square matrix
    output: 2 square matrices """
    n = len(mat)
    tmp1 = np.zeros((n, n), dtype=float)
    tmp2 = np.zeros((n, n), dtype=float)
    for j in range(n):
        sum_col = max(mat[:, j].sum(), 1)
        for i in range(n):
            sum_row = max(mat[i, :].sum(), 1)
            tmp1[i, j] = 1 - float(mat[i, j]) / sum_col
            tmp2[i, j] = 1 - float(mat[i, j]) / sum_row
        tmp1[j, j] = 0
        tmp2[j, j] = 0
    return tmp1, tmp2


""" MAIN CODE """

migration_matrix_file = 'migration_matrix.csv'
states = np.loadtxt('prepare_data_states_num_counties.csv', dtype=str,
                    delimiter=',')[:, 0]

# loop_ultrametric_inner_state()

""" This function takes hours to finish. """
# ultrametric_national()

