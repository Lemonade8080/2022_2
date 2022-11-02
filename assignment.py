# 2018253061_현동윤
import numpy as np
import time
import sys

DATA_COUNT = 517  # GENE 개수
dataset = []
cluster = [i for i in range(DATA_COUNT)]


def assignment5():
    count = []
    for data in dataset:
        if data[0] not in count:
            count.append(data[0])
        elif data[1] not in count:
            count.append(data[1])

    print(len(count))

def calc_distance(idx1, idx2):
    data1 = dataset[idx1]
    data2 = dataset[idx2]
    return np.round(sum([(data1 - data2) ** 2 for data1, data2 in list(zip(data1, data2))]) ** 0.5, 3)


def read_file(argv):
    file = open(argv, 'r')
    for line in file:
        dataset.append(line.replace('\n', '').split('\t'))
    file.close()


def write_file1(work_time):
    with open("assignment5_output.txt", "w") as file:
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
    read_file("data.txt")
    assignment5()
