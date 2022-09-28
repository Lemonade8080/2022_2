import math
import re
import numpy as np

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

f = open('dataset.txt', 'r')

for line in f:
    test = list(map(float, line.replace('\n', '').split('\t')))
    Dataset.append(test)


def init_center():


    divide_center_with(INIT_SLICE)
    show_centers()

    for cluster_idx in range(K):
        for data_idx in range(INIT_SLICE):
            sum_center_with(cluster_idx, data_idx)


def sum_center_with():
    for cluster_idx in range(K):
        for j in range(INIT_SLICE):
            for element in range(TIME_POINTS):
                Centers[cluster_idx][element] += Dataset[j + cluster_idx * INIT_SLICE][element]

def add_data_with(data_idx_list):
    data_size


def divide_center_with(cluster_size):
    for cluster_idx in range(K):
        for element in range(TIME_POINTS):
            Centers[cluster_idx][element] /= cluster_size


def show_centers():
    for x in Centers:
        print(np.round(x, 2))


if __name__ == '__main__':
    init_center()


def calc_dist(data_idx, center_idx):
    dist_sum = 0.0
    for i in range(TIME_POINTS):
        dist_sum += pow(Dataset[data_idx][i] - Centers[center_idx][i], 2)

    return dist_sum ** (1 / 2)


def no_name1(data_idx):
    closest_dist = [0.0 for i in range(K)]

    for center_idx in range(K):
        closest_dist[center_idx] = calc_dist(data_idx, center_idx)

    closest_index = closest_dist.index(min(closest_dist))
    Clusters[closest_index].append(data_idx)


def clustering():
    for row in range(GENES):
        no_name1(row)


def no_name3(length, cluster_idx):
    element_sum = [0.0 for j in range(TIME_POINTS)]
    for i in range(length):
        for j in range(TIME_POINTS):
            element_sum[j] += Dataset[Clusters[cluster_idx][i]][j]


def no_name2():
    cluster_size = [0.0 for i in range(K)]
    for j in range(K):
        cluster_size[j] = len(Clusters[j])

    for k in range(K):
        no_name3(len(Clusters[k]), k)





# init_center()
# for x in Centers:
#     print(np.round(x, 2))
#
# clustering()
# for x in Clusters:
#     print(x)


# def farthest_dist(center):
#     new_center = 0.0
#     distance = 0.0
#     for j in range(data_length):
#         x = calc_dist(dataset[j], center)
#         if x > distance:
#             new_center = dataset[j]
#             distance = x
#
#     return new_center
#
#
# def calc_center_avg(center_list, num):
#     center_sum = 0.0
#     for m in range(num):
#         center_sum += center_list[m]
#
#     return center_sum/num
#
#
# def select_center():
#     center_list = [0.0 for k in range(K)]
#     center_avg = 0.0
#     # center_list[0] = farthest_dist(center_avg)
#     # center_avg = calc_center_avg(center_list, 1)
#     #
#     # center_list[1] = farthest_dist(center_avg)
#     # center_avg = calc_center_avg(center_list, 2)
#
#
#
#     for l in range(K):
#         center_avg = calc_center_avg(center_list, l+1)
#         center_list[l] = farthest_dist(center_avg)
#
#     print(center_list, center_avg)
#
