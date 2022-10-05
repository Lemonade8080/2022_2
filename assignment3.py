# 2018253061_현동윤

import math
import sys
import numpy as np
import time
from copy import deepcopy

K = 10
DATA_COUNT = 500  # GENE 개수

centroid_list = np.zeros((K, 12), dtype=float)  # 중심 [ 12 개의 디멘션 ]
cluster_list = np.zeros(DATA_COUNT)             # 군집 [ 500 개의 데이터의 군집 ]
dataset = []


def main():
    read_file()
    assignment3()
    write_file()


def assignment3():
    start_time = time.process_time_ns() * 1000
    start_time2 = time.process_time()

    clustering()

    end_time = time.process_time_ns() * 1000
    end_time2 = time.process_time()
    print('time ', end_time - start_time)
    print('time ', end_time2 - start_time2)


def clustering():  # cluster 구하기 -> centroid 구하기 -> 중심값 변화 구하기

    move = True
    init_centroid()
    a = 0
    while move:
        old_centroid_list = deepcopy(centroid_list)
        update_clusters()
        update_centroid()
        move = calc_distance(old_centroid_list, centroid_list).any()
        a += 1

    print(result())
    print(centroid_list)
    print(a)


def init_centroid():
    sum_list = [np.round(np.sum([calc_distance(data, data2) for data2 in dataset]), 3) for data in dataset]
    idx_of_min = [sum_list.index(np.sort(sum_list)[idx]) for idx in range(K)]

    for idx in range(K):
        centroid_list[idx] = dataset[idx_of_min[idx]]

    # centroid_list = [dataset[idx] for idx in idx_of_min]


def update_centroid():
    for idx in range(K):
        cluster = [dataset[cluster_idx] for cluster_idx in range(DATA_COUNT) if cluster_list[cluster_idx] == idx]
        centroid_list[idx] = min_data(cluster)


def min_data(cluster):
    sum_list = [np.round(np.sum([calc_distance(data1, data2) for data2 in cluster]), 3) for data1 in cluster]
    return cluster[np.argmin(sum_list)]


def update_clusters():
    for cluster_idx in range(DATA_COUNT):
        distances = np.zeros(K)
        for idx in range(K):
            distances[idx] = calc_distance(dataset[cluster_idx], centroid_list[idx])
        cluster_list[cluster_idx] = np.argmin(distances)


def calc_distance(data, centroid):  # data 와 centroid 간의 거리 구하기
    dist_list = [(data - centroid) ** 2 for data, centroid in list(zip(data, centroid))]  # 12개 요소 각각의 차이
    return np.round(sum(dist_list) ** 0.5, 3)
    #  return np.round(sum([(data - centroid) ** 2 for data, centroid in list(zip(data, centroid))]) ** 0.5, 3)


def read_file():
    file = open('data.txt', 'r')
    for line in file:
        dataset.append(list(map(float, line.replace('\n', '').split('\t'))))
    file.close()


def write_file():
    with open("assignment3_output.txt", "w") as file:
        file.write(result())
    file.close()


def result():  # 결과 str 출력
    test = ''
    for idx in range(K):
        data_sum = [cluster_idx for cluster_idx in range(DATA_COUNT) if cluster_list[cluster_idx] == idx]
        test += str(len(data_sum)) + ': '
        for data in data_sum:
            test += str(data) + ' '
        test += '\n'

    return test


if __name__ == '__main__':
    main()
    # main(sys.argv[1])
