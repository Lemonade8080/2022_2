# 2018253061_현동윤
import numpy as np
import time
import sys

DATA_COUNT = 10  # GENE 개수
dataset = []
cluster = [i for i in range(DATA_COUNT)]


def assignment4():
    # single_link_distance()
    complete_link_distance()


def single_link_distance():
    start_time = time.process_time() * 1000
    clustering1()
    end_time = time.process_time() * 1000

    write_file1(end_time - start_time)
    print('time : ', end_time - start_time, 'ms')


def clustering1():
    distance_data = []
    for i in range(DATA_COUNT):
        for j in range(i + 1, DATA_COUNT):
            distance_data.append([calc_distance(i, j), i, j])

    sorted_data = sorted(distance_data, key=lambda x: x[0])

    count = 0
    for data in sorted_data:
        if data[0] > 20:
            print(data[1], '와', data[2], '의 거리가 20을 초과하여 중단합니다')
            break

        if cluster[data[1]] != cluster[data[2]]:
            merge_cluster([cluster[data[1]], cluster[data[2]]])

    print('\nClustering Count: ', count, '\n')


def complete_link_distance():
    start_time = time.process_time() * 1000
    clustering2()
    end_time = time.process_time() * 1000

    write_file2(end_time - start_time)
    print('time : ', end_time - start_time, 'ms')


def clustering2():
    distance_data = []
    for i in range(DATA_COUNT):
        for j in range(i + 1, DATA_COUNT):
            distance_data.append([calc_distance(i, j), i, j])

    print(distance_data)
    count = 0

    while True:
        min_idx = np.argmin([dist[0] for dist in distance_data])
        min_pos = distance_data[min_idx][1:3]
        min_dist = distance_data[min_idx][0]

        if min_dist > 20:
            print('20 초과')
            break

        merge_cluster(min_pos)
        update_data = [data for data in distance_data if min_pos[0] not in data]

        for data in update_data:
            if min_pos[1] in data:
                data = update(data, min_pos)

        distance_data = update_data

        if len(distance_data) == 0:
            print('클러스터 종료')
            break

        count += 1

    print('\nClustering Count: ', count, '\n')


def get_cluster_set():
    index_list = [idx for idx in range(DATA_COUNT) if idx in cluster]
    print('Get    Cluster:', index_list)
    return index_list


def merge_cluster(cluster_idx):
    for idx in range(DATA_COUNT):
        if cluster[idx] == cluster_idx[0]:
            cluster[idx] = cluster_idx[1]
    print('Move Cluster:', cluster_idx[0], 'to', cluster_idx[1])
    print('Now  Cluster:', cluster, '\n')


def update(data, min_pos):
    if data[1] == min_pos[1]:
        if calc_distance(data[2], min_pos[0]) > data[0]:
            data[0] = calc_distance(data[1], min_pos[0])
            print('Change Data:', data)

    elif data[2] == min_pos[1]:
        if calc_distance(data[1], min_pos[0]) > data[0]:
            data[0] = calc_distance(data[1], min_pos[0])
            print('Change Data:', data)
    return data


def calc_distance(idx1, idx2):
    data1 = dataset[idx1]
    data2 = dataset[idx2]
    return np.round(sum([(data1 - data2) ** 2 for data1, data2 in list(zip(data1, data2))]) ** 0.5, 3)


def read_file():
    file = open('data.txt', 'r')
    for line in file:
        dataset.append(list(map(float, line.replace('\n', '').split('\t'))))
    file.close()


def write_file1(work_time):
    with open("assignment4_output1.txt", "w") as file:
        file.write(get_result())
        file.write('time : ' + str(work_time) + 'ms')
    file.close()


def write_file2(work_time):
    with open("assignment4_output2.txt", "w") as file:
        file.write(get_result())
        file.write('time : ' + str(work_time) + 'ms')
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
    # read_file(sys.argv[1])
    read_file()
    assignment4()
