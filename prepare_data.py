# coding:utf-8
import numpy as np
import pandas as pd

""" Raw data file downloaded from
    https://www2.census.gov/programs-surveys/demo/tables/
        geographic-mobility/2015/county-to-county-migration-2011-2015/
        county-to-county-migration-flows/Net_Gross_US.txt
    and adjust the path to Net_Gross_US.txt on net_gross_file variable.

    State/U.S. Island Area/Foreign Region Name of Geography A
    County Name of Geography A
    State/U.S. Island Area/Foreign Region Name of Geography B
    County Name of Geography B
    Flow from Geography B to Geography A  Estimate
    Flow from Geography B to Geography A  MOE
    Counterflow from Geography A to Geography B  Estimate
    Counterflow from Geography A to Geography B  MOE
    Net Migration from Geography B to Geography A  Estimate
    Net Migration from Geography B to Geography A  MOE
    Gross Migration between Geography A and Geography B  Estimate
    Gross Migration between Geography A and Geography B  MOE
"""


""" FUNCTIONS """


def filter_line(s1, c1, s2, c2, x, y):
    print 'filtering data to %s, %s, %s, %s' % (s1, c1, s2, c2)
    from_ = (s1.decode('latin1'), c1.decode('latin1'))
    to_ = (s2.decode('latin1'), c2.decode('latin1'))
    from_idx = list_all_states_counties.index(from_)
    to_idx = list_all_states_counties.index(to_)
    migration_matrix[from_idx, to_idx] = y
    migration_matrix[to_idx, from_idx] = x


def get_len(state):
    """ This function returns the number of counties
    in the given state. """
    return int(len_states[np.where(len_states == state)[0][0], 1])


def print_fields(line):
    data = ''
    for cols in fields[used_fields]:
        val = line[cols[0]-1:cols[1]].strip()
        if val != '.':
            data += val+','
        else:
            data += val.replace('.', '')+','
    return data[:-1]


def use_pandas():
    global states, list_all_states_counties, num_counties
    print 'Defining variables states, list_all_states_counties, num_counties'
    df = pd.read_csv(raw_data_usa_file, encoding='latin1', header=None,
                     delim_whitespace=False, index_col=None, sep=',',
                     names=used_fields)
    data = df[df[7] != '-'][used_fields]
    states = data[4].unique().tolist()
    states = states[:-9]
    list_all_states_counties = []
    for state in states:
        tmp = data[data[4] == state][5].unique().tolist()
        list_all_states_counties += [(state, cty) for i, cty in enumerate(tmp)]

    num_counties = len(list_all_states_counties)


def write_raw_data():
    print 'Generating the file %s.\nWait...' % raw_data_usa_file
    with open(raw_data_usa_file, 'w') as f:
        with open(net_gross_file, 'rb') as net:
            for l in net:
                f.write('%s\n' % print_fields(l))
            net.close()
        f.close()


def migration_from_states():
    """ This function filter the migration data restricted to a state
    output: file with migration flow from the given state to USA. """
    migration_matrix = np.loadtxt(migration_matrix_file, dtype=int,
                                  delimiter=',')
    for i, state in enumerate(states):
        print 'filtering migration data from %s to USA' % state
        len_state = get_len(state)
        len_prev = sum([get_len(s) for s in states[:i]])
        tmp = migration_matrix[len_prev:len_prev+len_state]
        tmp_single = migration_matrix[len_prev:len_prev+len_state,
                                      len_prev:len_prev+len_state]
        np.savetxt('./migration_matrices/mat_%s_to_USA.csv' % state, tmp,
                   fmt='%d', delimiter=',')
        np.savetxt('./migration_matrices/mat_%s_to_%s.csv' % (state, state),
                   tmp_single, fmt='%d', delimiter=',')


def migration_usa():
    global migration_matrix
    migration_matrix = np.zeros((num_counties, num_counties))
    with open(raw_data_usa_file, 'r') as f:
        for line in f:
            tmp = line.split(',')
            if tmp[1] != '-' and tmp[3] != '-':
                filter_line(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5])
        f.close()
    np.savetxt(migration_matrix_file, migration_matrix, fmt='%d',
               delimiter=',')


""" MAIN CODE """

""" Required files """
net_gross_file = 'Net_Gross_US.txt'
fields = np.loadtxt('prepare_data_fields.csv', dtype=int, delimiter=',')
len_states = np.loadtxt('prepare_data_states_num_counties.csv', dtype=str,
                        delimiter=',')

""" Output files """
raw_data_usa_file = 'raw_data_usa.csv'
migration_matrix_file = 'migration_matrix.csv'

used_fields = [4, 5, 6, 7, 8, 10]

""" This function creates a huge ISO-8859-1 file. """
# write_raw_data()

""" This function defines variables
    states, list_all_states_counties, num_counties. """
# use_pandas()

""" This function creates a huge UTF-8 file
    containing the migration matrix. """
# migration_usa()

""" This function creates the migration data files in csv folder.
    To each state, there are files with migration
    from state to USA and from states to itself. """
# migration_from_states()
