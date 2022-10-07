# 2018253061_현동윤
import numpy as np
import time
import sys

K = 10
DATA_COUNT = 500  # GENE 개수

dataset = []
i_j_distance = np.zeros((DATA_COUNT, DATA_COUNT))


def main(args):
    read_file(args)
    assignment3()


def assignment3():
    medoid = init_medoid()

    start_time = time.process_time() * 1000
    cluster = clustering(medoid)
    end_time = time.process_time() * 1000

    work_time = end_time - start_time

    write_file(cluster, work_time)

    print(result(cluster))                         # Print Clusters
    print('time : ', work_time, 'ms')  # Print Time


def clustering(medoid, move=True):
    cluster = np.zeros(DATA_COUNT)
    count = 0
    while move:
        old_list = cluster
        cluster = update_clusters(medoid)
        medoid = update_medoids(cluster)
        count += 1
        move = (old_list != cluster).any()

    print('\nClustering Count: ', count - 1, '\n')

    return cluster


def init_medoid():  # Medoid 초기화

    for i in range(DATA_COUNT):  # 500개 간의 거리 값 설정
        i_j_distance[i] = [calc_distance(dataset[i], data) for data in dataset]

    distance = [np.sum([calc_distance(data, data2) for data2 in dataset]) for data in dataset]
    return [distance.index(np.sort(distance)[idx]) for idx in range(K)]


def update_clusters(medoid):
    cluster = np.zeros(DATA_COUNT)

    for data_idx in range(DATA_COUNT):
        distance_in_cluster = [calc_distance(dataset[data_idx], dataset[medoid[idx]]) for idx in range(K)]
        cluster[data_idx] = np.argmin(distance_in_cluster)
    return cluster


def update_medoids(cluster):
    medoid = np.zeros(K, int)

    for idx in range(K):
        test = [data_idx for data_idx in range(DATA_COUNT) if cluster[data_idx] == idx]
        medoid[idx] = get_medoid(test)
    return medoid


def get_medoid(index_list):  # Medoid 인 data_idx 를 return 한다.
    sum_in_cluster = [np.sum([i_j_distance[i][j] for j in index_list]) for i in index_list]  # size = (cluster,1)
    min_idx = np.argmin(sum_in_cluster)
    data_idx_is_medoid = index_list[min_idx]
    return data_idx_is_medoid


def calc_distance(data, centroid):  # data 와 centroid 간의 거리 구하기
    calc_dimension = [(data - centroid) ** 2 for data, centroid in list(zip(data, centroid))]
    distance = np.round(sum(calc_dimension) ** 0.5, 3)
    return distance


def read_file(args):
    file = open(args, 'r')
    for line in file:
        dataset.append(list(map(float, line.replace('\n', '').split('\t'))))
    file.close()


def write_file(cluster, work_time):
    with open("assignment3_output.txt", "w") as file:
        file.write(result(cluster))
        file.write('time : ' + str(work_time) + 'ms')
    file.close()


def result(cluster):  # 결과 str 출력
    test = ''
    for idx in range(K):
        data_sum = [data_idx for data_idx in range(DATA_COUNT) if cluster[data_idx] == idx]
        test += str(len(data_sum)) + ': '
        for data in data_sum:
            test += str(data) + ' '
        test += '\n'

    return test


if __name__ == '__main__':
    main(sys.argv[1])
