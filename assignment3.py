# 2018253061_현동윤

import math
import sys
import numpy as np
import random
from copy import deepcopy

K = 10
DATA_COUNT = 500  # GENE 개수

centroid_list = np.zeros((K, 12), dtype=float)  # 중심 [ 12 개의 디멘션 ]
cluster_list = np.zeros(DATA_COUNT)             # 군집 [ 500 개의 데이터의 군집 ]
medoid_list = np.zeros((K, 12), dtype=float)
dataset = []


def main():
    read_file()
    # assignment3()
    # write_file()

    sum_all()


def assignment3():  # cluster 구하기 -> center 구하기 -> 중심값 변화 구하기

    move = True
    init_centeroid()

    while move:
        old_centroid_list = deepcopy(centroid_list)
        clustering()
        move = calc_distance(old_centroid_list, centroid_list).any()

    print(result())


def clustering():
    update_clusters()
    update_centers()


# def sum_all():
#     test = np.zeros(K)
#     init_cluster = [i // 50 for i in range(DATA_COUNT)]
#
#     for i in range(K):
#         test[i] = np.sum([calc_distance(medoid_list[i], dataset[j]) for j in range(DATA_COUNT) if init_cluster[j] == i])
#
#     print(test)


def asdf():
    temp = np.zeros((DATA_COUNT, DATA_COUNT))

    for i in range(DATA_COUNT):
        for j in range(DATA_COUNT):
            temp = calc_distance(dataset[i], dataset[j])


def init_centeroid():
    rand = random.sample(range(0, DATA_COUNT), 10)
    for idx in range(K):
        medoid_list[idx] = dataset[rand[idx]]

    # init_cluster = [i // 50 for i in range(DATA_COUNT)]
    #
    # for idx in range(K):
    #     same_cluster = [dataset[cluster_idx] for cluster_idx in range(DATA_COUNT) if init_cluster[cluster_idx] == idx]
    #     centroid_list[idx] = np.round(np.mean(same_cluster, axis=0), 3)


def update_clusters():
    for cluster_idx in range(DATA_COUNT):
        distances = np.zeros(K)
        for centroid_idx in range(K):
            distances[centroid_idx] = calc_distance(dataset[cluster_idx], centroid_list[centroid_idx])
        cluster_list[cluster_idx] = np.argmin(distances)


def update_centers():
    for idx in range(K):
        same_cluster = [dataset[cluster_idx] for cluster_idx in range(DATA_COUNT) if cluster_list[cluster_idx] == idx]
        centroid_list[idx] = np.round(np.mean(same_cluster, axis=0), 3)


def calc_distance(data, center):  # data 와 center 간의 거리 구하기
    dist_list = [(data - center) ** 2 for data, center in list(zip(data, center))]  # 12개 요소 각각의 차이
    return np.round(sum(dist_list) ** 0.5, 3)
    #  return np.round(sum([(data - center) ** 2 for data, center in list(zip(data, center))]) ** 0.5, 3)


def read_file():
    file = open('data.txt', 'r')
    for line in file:
        dataset.append(list(map(float, line.replace('\n', '').split('\t'))))
    file.close()


def write_file():
    with open("assignment2_output.txt", "w") as file:
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
