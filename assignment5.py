# 2018253061_현동윤
import numpy as np
import time
import sys

K = 8
gene_list = dict()
gene_pair = []
maximal_clique = []


def assignment5():
    zero()
    print_result()


def zero():
    clique = sorted(gene_pair)  # degree 2

    while True:
        new_clique = []
        for c in clique:                      # {A: D}, {B: D, F}, {C: D, G}
            inter = list(intersection(c))     # {D}

            if len(inter) > 0:
                for gene in inter:            # gene = D
                    genes = c.copy()          # genes = [A, B, C] [B, A, C]...
                    genes.append(gene)        # genes = [A, B, C, D] [B, A, C, D]...
                    genes = sorted(genes)     # genes = [A, B, C, D] [A, B, C, D]...
                    new_clique.append(genes)
            else:
                maximal_clique.append(c)      # 중복이 없으면 maximal 하다.

        clique = unique_list(new_clique)      # 리스트 중복 제거

        if len(new_clique) == 0:              # 종료 조건
            break


def unique_list(test):
    unique = []
    return [x for x in test if x not in unique and not unique.append(x)]


def intersection(clique):
    i = gene_list[clique[0]]
    for c in clique[1:]:
        i = i & gene_list[c]

        if len(i) <= 0:
            break
    return i


def read_file(argv):
    file = open(argv, 'r')
    for line in file:
        genes = line.strip().split('\t')
        gene_pair.append(genes)
        try:
            gene_list[genes[0]].add(genes[1])
        except KeyError:
            gene_list[genes[0]] = {genes[1]}

        try:
            gene_list[genes[1]].add(genes[0])
        except KeyError:
            gene_list[genes[1]] = {genes[0]}

    file.close()


def write_file():
    with open("assignment5_output.txt", "w") as file:
        file.write(print_result())
    file.close()


def print_result():
    result = ""
    for clique in maximal_clique:
        if len(clique) >= K:
            result += str(len(clique)) + ': '
            for c in clique:
                result += str(c) + ' '
            result += '\n'
    print(result)
    return result


if __name__ == '__main__':
    read_file(sys.argv[1])
    assignment5()
    write_file()
