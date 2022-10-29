# 2018253061_현동윤
import numpy as np
import time
import sys

DATA_COUNT = 500  # GENE 개수
dataset = []
cluster = [i for i in range(DATA_COUNT)]


def assignment4():
    single_link_distance()

    for i in range(DATA_COUNT):
        cluster[i] = i

    complete_link_distance()


def single_link_distance():
    start_time = time.process_time() * 1000
    clustering1()
    end_time = time.process_time() * 1000

    write_file1(end_time - start_time)
    print('time : ', end_time - start_time, 'ms')


def clustering1():
    distances = [[i, j, calc_distance(i, j)] for i in range(DATA_COUNT) for j in range(i+1, DATA_COUNT)]

    cnt = 0
    while True:
        min_idx = np.argmin([dist[2] for dist in distances])
        min_pos = distances[min_idx][0:2]
        min_dist = distances[min_idx][2]

        if min_dist > 5:
            print(min_pos, '거리 5 초과:', min_dist)
            break

        a = [data for data in distances if min_pos[0] in data[0:2] and min_pos[1] not in data]
        b = [data for data in distances if min_pos[1] in data[0:2] and min_pos[0] not in data]
        c = [[data2[0], data2[1], data1[2]] if data1[2] < data2[2] else data2 for data1, data2 in zip(a, b)]

        print(cnt, distances[min_idx])
        cnt += 1

        distances = [data for data in distances if min_pos[0] not in data[0:2] and min_pos[1] not in data[0:2]]

        for data in c:
            distances.append(data)
        distances = sorted(distances)

        merge_cluster(min_pos)

        if len(distances) == 0:
            print('클러스터 종료')
            break


def complete_link_distance():
    start_time = time.process_time() * 1000
    clustering2()
    end_time = time.process_time() * 1000

    write_file2(end_time - start_time)
    print('time : ', end_time - start_time, 'ms')


def clustering2():
    distances = [[i, j, calc_distance(i, j)] for i in range(DATA_COUNT) for j in range(i + 1, DATA_COUNT)]

    cnt = 0
    while True:
        min_idx = np.argmin([dist[2] for dist in distances])
        min_pos = distances[min_idx][0:2]  # 거리가 최소인 Cluster a, b
        min_dist = distances[min_idx][2]   # 거리 최솟값

        if min_dist > 5:
            print(min_pos, '거리 5 초과:', min_dist)
            break

        a = [data for data in distances if min_pos[0] in data[0:2] and min_pos[1] not in data]  # a가 속한 데이터
        b = [data for data in distances if min_pos[1] in data[0:2] and min_pos[0] not in data]  # b가 속한 데이터
        c = [[data2[0], data2[1], data1[2]] if data1[2] > data2[2] else data2 for data1, data2 in zip(a, b)] # a-b

        # print(cnt, distances[min_idx])
        cnt += 1

        distances = [data for data in distances if min_pos[0] not in data[0:2] and min_pos[1] not in data[0:2]]

        for data in c:
            distances.append(data)
        distances = sorted(distances)

        merge_cluster(min_pos)

        if len(distances) == 0:
            print('클러스터 종료')
            break


def merge_cluster(cluster_idx):
    for idx in range(DATA_COUNT):
        if cluster[idx] == cluster_idx[0]:
            cluster[idx] = cluster_idx[1]
    #print('Move Cluster:', cluster_idx[0], 'to', cluster_idx[1], '\n')
    # print('Now  Cluster:', cluster, '\n')


def calc_distance(idx1, idx2):
    data1 = dataset[idx1]
    data2 = dataset[idx2]
    return np.round(sum([(data1 - data2) ** 2 for data1, data2 in list(zip(data1, data2))]) ** 0.5, 3)


def read_file(argv):
    file = open(argv, 'r')
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
    return test


if __name__ == '__main__':
    read_file(sys.argv[1])
    assignment4()
