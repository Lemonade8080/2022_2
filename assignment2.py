import sys
import numpy as np
from copy import deepcopy

K = 10
DATA_COUNT = 500  # GENE 개수

centroid_list = np.zeros((K, 12), dtype=float)  # 중심 [ 12 개의 디멘션 ]
cluster_list = np.zeros(DATA_COUNT)             # 군집 [ 500 개의 데이터의 군집 ]
dataset = []


def main(args):
    read_file(args)
    assignment2()
    write_file()


def assignment2():  # cluster 구하기 -> center 구하기 -> 중심값 변화 구하기

    move = True
    init_center()

    while move:
        old_centroid_list = deepcopy(centroid_list)
        update_clusters()
        update_centers()
        move = calc_distance(old_centroid_list, centroid_list).any()


def clustering():
    update_clusters()
    update_centers()


def init_center():
    # rand = random.sample(range(0, 500), 10)
    # for idx in range(K):
    #     centroid_list[idx] = dataset[rand[idx]]

    init_cluster = [i // 50 for i in range(500)]

    for idx in range(K):
        same_cluster = [dataset[cluster_idx] for cluster_idx in range(DATA_COUNT) if init_cluster[cluster_idx] == idx]
        centroid_list[idx] = np.round(np.mean(same_cluster, axis=0), 3)


def update_clusters():
    for data_idx in range(DATA_COUNT):
        distances = [calc_distance(dataset[data_idx], centroid_list[idx]) for idx in range(K)]
        cluster_list[data_idx] = np.argmin(distances)


def update_centers():
    for idx in range(K):
        same_cluster = [dataset[cluster_idx] for cluster_idx in range(DATA_COUNT) if cluster_list[cluster_idx] == idx]
        centroid_list[idx] = np.mean(same_cluster, axis=0)


def calc_distance(data, center):  # data 와 center 간의 거리 구하기
    return np.round(sum([(data - center) ** 2 for data, center in list(zip(data, center))]) ** 0.5, 3)


def read_file(args):
    file = open(args, 'r')
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
    main(sys.argv[1])
