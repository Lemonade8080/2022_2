import math
import sys
import numpy as np
from copy import deepcopy


K = 10
DATA_COUNT = 500  # GENE 개수

center_list = np.zeros((K, 12), dtype=float)  # 중심 [ 12 개의 디멘션 ]
cluster_list = np.zeros(DATA_COUNT)           # 군집 [ 500 개의 데이터의 군집 ]
dataset = []


def main(args):
    read_file(args)
    assignment2()
    write_file()


def assignment2():  # cluster 구하기 -> center 구하기 -> 중심값 변화 구하기

    move = True
    init_center()

    while move:
        old_center_list = deepcopy(center_list)
        clustering()
        move = calc_distance(old_center_list, center_list).any()


def clustering():
    update_clusters()
    update_centers()


def init_center():
    init_cluster = [i // 50 for i in range(500)]

    for idx in range(K):
        data_sum = [dataset[cluster_idx] for cluster_idx in range(DATA_COUNT) if init_cluster[cluster_idx] == idx]
        center_list[idx] = np.round(np.mean(data_sum, axis=0), 3)


def update_clusters():
    for cluster_idx in range(DATA_COUNT):
        distances = np.zeros(K)
        for j in range(K):
            distances[j] = calc_distance(dataset[cluster_idx], center_list[j])
        cluster_list[cluster_idx] = np.argmin(distances)


def update_centers():
    for idx in range(K):
        data_sum = [dataset[cluster_idx] for cluster_idx in range(DATA_COUNT) if cluster_list[cluster_idx] == idx]
        center_list[idx] = np.round(np.mean(data_sum, axis=0), 3)


def calc_distance(data, center):  # data 와 center 간의 거리 구하기
    return np.round(sum([(data - center) ** 2 for data, center in list(zip(data, center))]), 3)


def read_file(args):
    file = open(args, 'r')
    for line in file:
        dataset.append(list(map(float, line.replace('\n', '').split('\t'))))
    file.close()


def write_file():
    with open("assignment2.txt", "w") as file:
        file.write(result())

    file.close()


def result():
    result_list = [[] for k in range(K)]
    test = ''
    for i in range(K):
        for j in range(DATA_COUNT):
            if cluster_list[j] == i:
                result_list[i].append(j)
        test += str(len(result_list[i])) + ': ' + str(result_list[i]) + '\n'
        print(len(result_list[i]), ': ', result_list[i], '\n')

    return test


if __name__ == '__main__':
    main(sys.argv[1])
