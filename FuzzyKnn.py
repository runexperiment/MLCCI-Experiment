import os

import numpy as np
from tqdm import trange
from sklearn.neighbors import NearestNeighbors

import Tool_io


# 向量之间距离
def vec_distance(vector1, vector2):
    v1 = np.mat(vector1)
    v2 = np.mat(vector2)
    res = np.sqrt((v1 - v2) * (v1 - v2).T)
    return res


# knn算法
def fuzzy_knn(cov_dict, failIndex, trueCC, versionPath):
    # 若存在结果 直接返回
    WPcc = Tool_io.checkAndLoad(versionPath[1], "fuzzy_knn")
    if WPcc != None:
        return WPcc
    cov = cov_dict

    dict = cov_dict['dstar']
    #将字典变成numpy数组ssss
    cov = []
    for index in dict:
       cov.append(dict[index])
    cov = np.array(cov)

    # knn实现ss
    # print("fail:",len(failIndex))
    # print("truecc:",len(trueCC))
    # print("cov:",len(cov))
    # print(len(cov))
    if len(failIndex)+len(trueCC) == len(cov):
        n = len(failIndex)+len(trueCC)
    else:
        n = len(failIndex) + len(trueCC) + 1
    nbrs = NearestNeighbors(n_neighbors=n, algorithm='auto').fit(cov)
    distances, indices = nbrs.kneighbors(cov)
    # 记录测试用例cc概率值
    WPcc = {}
    for key_index in trange(len(distances)):
        if key_index not in failIndex:
            # 当前测试用例邻居（距离值）
            current_test = distances[key_index]
            # 最大距离值
            max_dis = np.max(current_test) + 1
            # 当前测试用例邻居（编号）
            id_dis = indices[key_index]
            # 模糊加权公式 分子(numerator) / 分母(denominator)
            numerator = 0
            denominator = 0
            for dis_index in range(len(current_test)):
                # 编号为0的邻居是自己
                if dis_index != 0:
                    func = 0
                    if id_dis[dis_index] in failIndex:
                        func = 1
                    numerator = numerator + (max_dis - current_test[dis_index]) * func
                    denominator = denominator + (max_dis - current_test[dis_index])
            WPcc[key_index] = numerator / denominator
    # 保留计算结果
    Tool_io.checkAndSave(versionPath[1], "fuzzy_knn", WPcc)
    return WPcc




# cc计算指标
def fuzzy_knn_metric(real_CC, failIndex, WPcc, versionPath, error_pro_ver):


    identity_result = Tool_io.checkAndLoad(versionPath[1], "cc_identity_result")
    if identity_result != None:
        return identity_result

    rank = sorted(WPcc.items(), key=lambda d: d[1], reverse=True)

    r = []
    for real in real_CC:
        r.append(real)

    # cc概率排序
    sort = Tool_io.checkAndLoad(versionPath[1], "cc_sort")
    if sort == None:
        sort = []
        for id in range(len(rank)):
            s = rank[id][0]
            if s in r:
                sort.append(id)
        Tool_io.checkAndSave(versionPath[1], "cc_sort", sort)

    WPcc_len = len(WPcc)
    # cc概率之和s
    cc_sum = 0
    for cc in real_CC:
        cc_tmp = WPcc[cc]
        cc_sum = cc_sum + cc_tmp
        WPcc.pop(cc)

    # 通过测试用例但非cc概率之和
    no_cc_sum = 0
    for no_cc in WPcc:
        no_cc_sum = no_cc_sum + WPcc[no_cc]

    # 召回率
    if len(real_CC) == 0:
        recall = 0
    else:
        recall = cc_sum / len(real_CC)

    # FPR
    FPrate = no_cc_sum / (WPcc_len - len(real_CC))

    # precision
    if cc_sum == 0 and no_cc_sum == 0:
        precision = 0
    else:
        precision = cc_sum / (cc_sum + no_cc_sum)

    # F-measure
    if precision == 0 and recall == 0:
        Fmeasure = 0
    else:
        Fmeasure = (2 * precision * recall) / (precision + recall)

    identity_result = {}
    identity_result['recall'] = recall
    identity_result['FPrate'] = FPrate
    identity_result['precision'] = precision
    identity_result['Fmeasure'] = Fmeasure

    # 保存结果
    Tool_io.checkAndSave(versionPath[1], "cc_identity_result", identity_result)

    return identity_result







