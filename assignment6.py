# 2018253061_현동윤
import numpy as np
import time
import sys
import copy

sys.setrecursionlimit(10**6)

THRESHOLD = 0.4
cluster_list = []
init_dict = dict()
init_edge = []


def assignment6():
    edge_list = init_edges()  # 1개의 complete graph를 구함.
    recursion(edge_list)      # top-down

# ------------ get init graph  ----------------- #


def init_edges():
    graph = init_graph()
    edge_list = sorted(get_edge_list(graph, init_edge))
    return edge_list


def init_graph():
    main_graph = []  # init graph
    visit = []  # 참조한 vertex 들
    for key in init_dict.keys():
        if key not in visit:
            sub_graph = get_sub_graph(key, init_edge)
            edge = get_edge_list(sub_graph, init_edge)

            if is_density(sub_graph, edge):
                cluster_list.append(sub_graph)
            else:
                main_graph = sub_graph

            for sub in sub_graph:
                visit.append(sub)

    # density 0.4 미만인 complete graph가 1개 이기 때문에 main_graph = sub_graph 한뒤 반환
    return main_graph


def get_edge_list(graph, prev_list):
    edge_list = [e for e in prev_list if e[0] in graph and e[1] in graph]
    return edge_list

# ------------ main recursion  ----------------- #


def recursion(edge_list):
    rmv_list = get_remove_edge_list(edge_list)

    while True:
        rmv_edge = rmv_list[0]
        del rmv_list[0]
        edge_list.remove(rmv_edge)

        if not is_connected(rmv_edge, edge_list):
            sub1 = get_sub_graph(rmv_edge[0], edge_list)
            sub2 = get_sub_graph(rmv_edge[1], edge_list)
            break

    add_cluster(sub1, edge_list)
    add_cluster(sub2, edge_list)

# ------------ remove list  ----------------- #


def get_remove_edge_list(edge_list):
    edge_dict = get_dict(edge_list)

    jaccard_list = []
    for e1, e2 in edge_list:
        jaccard = calc_jaccard_idx(e1, e2, edge_dict)
        jaccard_list.append([jaccard, e1, e2])

    jaccard_list = sorted(jaccard_list, key=lambda x: (x[0]))
    rmv_list = [j[1:3] for j in jaccard_list]
    return rmv_list


def calc_jaccard_idx(e1, e2, edge_dict):
    v1 = edge_dict[e1]
    v2 = edge_dict[e2]
    return len(v1 & v2) / len(v1 | v2)


def get_dict(edge_list):
    new_dict = dict()

    for edge in edge_list:
        try:
            new_dict[edge[0]].add(edge[1])
        except KeyError:
            new_dict[edge[0]] = {edge[1]}

        try:
            new_dict[edge[1]].add(edge[0])
        except KeyError:
            new_dict[edge[1]] = {edge[0]}

    return new_dict

# ------------ check connection -------------- #


def is_connected(edge, edge_list):
    start = edge[0]
    end = edge[1]
    graph = get_dict(edge_list)
    visit = []
    try:
        return dfs_connect(graph, start, end, visit)

    except KeyError:
        return False


def dfs_connect(graph, start, end, visit):
    visit.append(start)

    if start == end:
        return True

    for vertex in graph[start]:
        if vertex not in visit:
            if dfs_connect(graph, vertex, end, visit):
                return True
    return False

# ------------ get sub graph ----------------- #


def get_sub_graph(start, edge_list):
    edge_dict = get_dict(edge_list)
    sub_graph = []
    try:
        return DFS(edge_dict, start, sub_graph)

    except KeyError:
        return sub_graph


def DFS(graph, start, visit):
    visit.append(start)

    for vertex in graph[start]:
        if vertex not in visit:
            DFS(graph, vertex, visit)

    return visit


# ---------- add or div cluster ------------- #

def add_cluster(graph, edge_list):
    prev_list = copy.deepcopy(edge_list)
    now_list = get_edge_list(graph, prev_list)
    if is_density(graph, now_list):
        # print('{0}: add {1}'.format(len(graph), graph))
        cluster_list.append(graph)
    else:
        # print('{0}: div {1}'.format(len(graph), graph))
        recursion(now_list)

# ----------- check density ----------------- #


def is_density(graph, edge_list):
    return calc_density(graph, edge_list) >= THRESHOLD


def calc_density(graph, edge_list):
    vertex_cnt = len(graph)
    edge_cnt = len(edge_list)

    if vertex_cnt == 1:
        return 1
    else:
        return 2 * edge_cnt / (vertex_cnt * (vertex_cnt - 1))

# -------------------------------------------- #


def read_file(argv):
    file = open(argv, 'r')
    for line in file:
        genes = line.strip().split('\t')
        init_edge.append(genes)
        try:
            init_dict[genes[0]].add(genes[1])
        except KeyError:
            init_dict[genes[0]] = {genes[1]}

        try:
            init_dict[genes[1]].add(genes[0])
        except KeyError:
            init_dict[genes[1]] = {genes[0]}
    file.close()


def get_result():
    result = ""
    for c in cluster_list:
        if len(c) >= 10:
            result += '{0}: {1}\n'.format(len(c), ' '.join(c))
    print(result)
    return result


def write_file():
    with open("assignment6_output.txt", "w") as file:
        file.write(get_result())
    file.close()


if __name__ == '__main__':
    # read_file(sys.argv[1])
    read_file("data.txt")
    assignment6()
    write_file()
