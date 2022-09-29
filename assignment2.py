# 2018253061_현동윤

import math
import re
import numpy as np
import sys

# input_file = sys.argv[1]


###################################################################
#                       D E F I N I T I O N                       #
###################################################################

K = 10
INIT_SLICE = 50

GENES = 500
TIME_POINTS = 12
COUNT = 0

Clusters = [[] for x in range(K)]
Centers = [[0.0 for y in range(TIME_POINTS)] for z in range(K)]

# dataset = [[0.0 for i in range(12)] for j in range(500)]
Dataset = []

###################################################################
#                       D E F I N I T I O N                       #
###################################################################


###################################################################
#                       I N I T    D A T A                        #
###################################################################
f = open('data.txt', 'r')

for line in f:
    test = list(map(float, line.replace('\n', '').split('\t')))
    Dataset.append(test)


def init_data():
    init_center()
    init_cluster()


def init_cluster():
    for idx_cluster in range(K):
        for idx_data_in_cluster in range(INIT_SLICE):
            Clusters[idx_cluster].append(idx_data_in_cluster + idx_cluster * INIT_SLICE)


def init_center():
    for idx_cluster in range(K):
        Centers[idx_cluster] = calc_center_list_with(idx_cluster)

###################################################################
#                       I N I T    D A T A                        #
###################################################################


def calc_center_list_with(idx_cluster):
    center_list = calc_sum_cluster_of(idx_cluster)
    center_list = calc_div_cluster_for(INIT_SLICE, center_list)

    return center_list


def calc_sum_cluster_of(idx_cluster):
    sum_cluster = [0.0 for i in range(TIME_POINTS)]

    idx_start = idx_cluster * INIT_SLICE
    for idx_data_in_cluster in range(INIT_SLICE):
        idx_data = idx_start + idx_data_in_cluster
        sum_cluster = add_data_element_in(sum_cluster, idx_data)

    return sum_cluster


def add_data_element_in(sum_element_list, idx_data):
    for idx_element in range(TIME_POINTS):
        sum_element_list[idx_element] += Dataset[idx_data][idx_element]

    return sum_element_list


def calc_div_cluster_for(div_num, sum_list):
    for idx_element in range(TIME_POINTS):
        sum_list[idx_element] /= div_num

    return sum_list


def clustering():
    clear_cluster()
    for idx_data in range(GENES):
        assign_cluster(idx_data)


def clear_cluster():
    for i in range(K):
        Clusters[i].clear()


def assign_cluster(idx_data):
    idx_cluster = find_cluster_idx(idx_data)
    Clusters[idx_cluster].append(idx_data)


def find_cluster_idx(idx_data):
    dist_list = get_dist_list(idx_data)
    shortest_dist = min(dist_list)

    return dist_list.index(shortest_dist)


def get_dist_list(idx_data):
    dist = [0.0 for i in range(K)]

    for idx_center in range(K):
        dist[idx_center] = calc_distance(idx_data, idx_center)

    return dist


def calc_distance(idx_data, idx_center):
    dist_sum = 0.0
    for element in range(TIME_POINTS):
        dist_sum += pow(Dataset[idx_data][element] - Centers[idx_center][element], 2)

    return dist_sum ** (1 / 2)

###################################################################
#                             T E S T                             #
###################################################################


def show_centers():
    for x in Centers:
        print(np.round(x, 2))


def show_clusters():
    i = 1
    for x in Clusters:
        print('Cluster', i, x)
        i += 1


if __name__ == '__main__':
    init_data()
    # clustering()

    show_centers()
    show_clusters()
