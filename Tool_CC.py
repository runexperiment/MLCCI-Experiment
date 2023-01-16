# coding=utf-8
from __future__ import division


def getTCC_singeFault(covMatrix, in_vector):
    tcc_inner_list = []
    # 肯定有错误的语句
    fault_contain_statement_list = []
    for index in range(len(covMatrix[0])):
        flag = True
        for item in range(len(in_vector)):
            if covMatrix[item][index] == 0 and in_vector[item]==1:
                flag = False
                break
        if flag:
            fault_contain_statement_list.append(index)
    # 根据错误语句找TCC
    for index2 in range(len(in_vector)):
        if in_vector[index2] == 0 and index2 not in tcc_inner_list:
            fail = True
            for index3 in fault_contain_statement_list:
                if covMatrix[index2][index3] == 0:
                    fail = False
                    break
            if fail:
                tcc_inner_list.append(index2)
    return tcc_inner_list


def getTCC(covMatrix, in_vector):
    TCC = []
    for index1 in range(len(in_vector)):
        if in_vector[index1] == 1:
            for index2 in range(len(in_vector)):
                if in_vector[index2] == 0 and index2 not in TCC:
                    fail = True
                    for index3 in range(len(covMatrix[0])):
                        if covMatrix[index1][index3] == 1 and covMatrix[index2][index3] == 0:
                            fail = False
                            break
                    if fail:
                        TCC.append(index2)
    return TCC


if __name__ == "__main__":
    covMatrix=[[1,0,1,1],
               [1,1,1,0],
               [1,0,1,0],
               [1,1,1,1],
               [0,0,0,0]]
    in_vector=[1,1,0,0,0]
    result=getTCC_singeFault(covMatrix, in_vector)
    print(result)
    result=getTCC(covMatrix, in_vector)
    print(result)