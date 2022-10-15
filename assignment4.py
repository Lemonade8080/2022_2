# 2018253061_현동윤
import numpy as np
import time
import sys


DATA_COUNT = 500  # GENE 개수
dataset = []
cluster = [i for i in range(DATA_COUNT)]
# cluster = [4, 5, 9, 7, 4, 5, 6, 7, 9, 9]
i_j_distance = np.zeros((DATA_COUNT, DATA_COUNT))


def main():
    read_file()
    assignment4()


def assignment4():

    # start_time = time.process_time() * 1000
    clustering1()
    # end_time = time.process_time() * 1000
    # single_time = end_time - start_time

    # start_time = time.process_time() * 1000
    # clustering2()
    # end_time = time.process_time() * 1000
    # complete_time = start_time - end_time

    # print('time : ', single_time, 'ms')
    # print('time : ', complete_time, 'ms')


def clustering2():
    init_distance()

    while True:
        cluster_set = get_cluster_set()
        cluster_len = len(cluster_set)
        
        if cluster_len == 1:
            print('하나의 Cluster 로 합쳐짐')
            break

        distance_data = []   # 거리, 인덱스1(cluster_list), 인덱스2(cluster_list)
        for i in range(cluster_len):
            for j in range(i + 1, cluster_len):
                idx1 = cluster_set[i]
                idx2 = cluster_set[j]
                distance_data.append([complete_link(idx1, idx2), i, j])

        min_dists = [dist[0] for dist in distance_data]
        data_idx = np.argmin(min_dists)

        print('Min of Distance:', min(min_dists))

        if distance_data[data_idx][0] > 20:
            print('20 초과하여 중단')
            break

        set_idx1 = distance_data[data_idx][1]
        set_idx2 = distance_data[data_idx][2]
        cluster_idx1 = cluster_set[set_idx1]
        cluster_idx2 = cluster_set[set_idx2]
        print('Complete_Index:', cluster_idx1, cluster_idx2)
        merge_cluster(cluster_idx1, cluster_idx2)


def complete_link(idx1, idx2):
    cluster_idx1 = cluster[idx1]
    cluster_idx2 = cluster[idx2]
    cluster1 = [idx for idx in range(len(cluster)) if cluster[idx] == cluster_idx1]
    cluster2 = [idx for idx in range(len(cluster)) if cluster[idx] == cluster_idx2]
    # print('cluster2: ', cluster2)
    # print('cluster1: ', cluster1)
    distance = min([i_j_distance[idx1][idx2] for idx1 in cluster1 for idx2 in cluster2])
    # distance2 = min([i_j_distance[idx1][idx2] for idx1 in cluster1 for idx2 in cluster2])
    # print('max: ', distance)
    # print('min: ', distance2)
    return distance


def get_cluster_set():
    index_list = [idx for idx in range(DATA_COUNT) if idx in cluster]
    print('Get    Cluster:', index_list)
    return index_list


def init_distance():
    for i in range(DATA_COUNT):  # 500개 간의 거리 값 설정
        i_j_distance[i] = [calc_distance(i, j) for j in range(DATA_COUNT)]


def clustering1():
    count = 0

    distance_data = []
    for i in range(DATA_COUNT):
        for j in range(i + 1, DATA_COUNT):
            distance_data.append([calc_distance(i, j), i, j])

    sorted_distances = sorted(distance_data, key=lambda x: x[0])
    merge_list = [dist[1:3] for dist in sorted_distances]

    for merge in merge_list[0:20]:
        if calc_distance(merge[0], merge[1]) > 20:
            print(merge[0], '와', merge[1], '의 거리가 20을 초과하여 중단합니다')
            break

        if cluster[merge[0]] != cluster[merge[1]]:
            merge_cluster(cluster[merge[0]], cluster[merge[1]])
            count += 1

    print('\nClustering Count: ', count, '\n')


def merge_cluster(idx1, idx2):
    for i in range(DATA_COUNT):
        if cluster[i] == idx1:
            cluster[i] = idx2
    print('Change Cluster:', idx1, 'to', idx2)
    # print('After  Cluster:', cluster, '\n')


def calc_distance(idx1, idx2):
    data1 = dataset[idx1]
    data2 = dataset[idx2]
    return np.round(sum([(data1 - data2) ** 2 for data1, data2 in list(zip(data1, data2))]) ** 0.5, 3)


def read_file():
    file = open('data.txt', 'r')
    for line in file:
        dataset.append(list(map(float, line.replace('\n', '').split('\t'))))
    file.close()


def write_file(work_time1, work_time2):
    with open("assignment4_output1.txt", "w") as file:
        file.write(get_result())
        file.write('time : ' + str(work_time1) + 'ms')
    file.close()

    with open("assignment4_output2.txt", "w") as file:
        file.write(get_result())
        file.write('time : ' + str(work_time2) + 'ms')
    file.close()


def get_result():
    test = ''

    for cluster_idx in range(DATA_COUNT):
        cluster_set = [idx for idx in range(DATA_COUNT) if cluster[idx] == cluster_idx]
        if len(cluster_set) != 0:
            test += str(len(cluster_set)) + ': '
            for idx in cluster_set:
                test += str(idx) + ' '
            test += '\n'
    print(test)
    return test


if __name__ == '__main__':
    # main(sys.argv[1])
    main()
