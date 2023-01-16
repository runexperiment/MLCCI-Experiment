import copy
import random
import os
import numpy as np
import sklearn
import csv
import Tool_io
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# 统计特征结果
def deal_feature(ver):
    # 获取realcc和测试用例数量,字典（4:失败测试用例数量;5:通过测试用例数量;6:cc测试用例(字典))
    statistic_res = Tool_io.checkAndLoad(ver[1], "data_Coverage_InVector_saveAgain")
    if statistic_res is not None:
        # 获取归一化后的特征
        res_nor = Tool_io.checkAndLoad(ver[1], "normalization")
        if res_nor is not None:
            res_array = np.array(res_nor['normal'])
            # 记录测试用例是否为cc
            cc_vector = np.zeros(statistic_res[4] + statistic_res[5], dtype=np.int)
        else:
            return None, None
        for cc_index in statistic_res[6]:
            cc_vector[cc_index] = 1
        for fault_index in statistic_res[7]:
            cc_vector[fault_index] = 1
        return res_array, cc_vector
    else:
        return None, None


# 决策树
def decision_tree(train_sample, train_target, ver_dict, pro_name,res_path):
    clf = tree.DecisionTreeClassifier(criterion='gini')
    clf = clf.fit(train_sample, train_target)
    recall_sum = 0
    precision_sum = 0
    FPR_sum = 0
    f1_score_sum = 0
    content = []
    for ver in ver_dict:
        X_test = ver_dict[ver]['res_array']
        y_test = ver_dict[ver]['cc_vector']
        score = clf.score(X_test, y_test)
        y_predicate = clf.predict(X_test)
        res = confusion_matrix(y_test, y_predicate)
        cmatrix = sklearn.metrics.classification_report(y_test, y_predicate)
        # 混淆矩阵参数
        TP = res[1,1]
        TN = res[0,0]
        FP = res[0,1]
        FN = res[1,0]
        #print(res)
        #print(cmatrix)
        print(pro_name, ':', score)
        # cc识别指标
        if TP+FN == 0:
            recall = 0
        else:
            recall = TP / (TP + FN)
        if TP + FP == 0:
            precision = 0
        else:
            precision = TP / (TP + FP)
        if FP + TN == 0:
            FPR = 0
        else:
            FPR = FP / (FP + TN)
        f1_score = 2*TP / (2*TP+FP+FN)
        print(recall,precision,FPR,f1_score)
        # 记录版本信息
        content.append([ver,recall,precision,FPR,f1_score])
        # 算各个指标总和
        recall_sum = recall_sum + recall
        precision_sum = precision_sum + precision
        FPR_sum = FPR_sum + FPR
        f1_score_sum = f1_score_sum + f1_score
    # 记录当前总和
    count = len(ver_dict)
    content.append(['avg',recall_sum/count, precision_sum/count, FPR_sum/count, f1_score_sum/count])
    creat_res_file(pro_name, content)
    return recall_sum/count, precision_sum/count, FPR_sum/count, f1_score_sum/count


# 随机抽取三分之一的训练集
def one_third(ver_dict):
    res = []
    for i in range(5):
        train_data_key = random.sample(list(ver_dict), int(len(ver_dict) / 3))
        res.append(train_data_key)
    return res


# 存储结果
def creat_res_file(csv_name, content):
    path = os.path.join(res_path, csv_name) + ".csv"
    header = ['', 'recall', 'precision','FPrate','Fmeasure']
    with open(path, 'a+', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for row in content:
            writer.writerow(row)
        writer.writerow(['', '', '', '', ''])


# 准备数据集和训练集
def machine_learning(root,data,error_pro_ver,res_path):
    # 获取程序路径
    res = Tool_io.create_data_folder(root, data)
    for index in res:
        # 程序名称
        pro_name = os.path.basename(index)
        # 测试用例
        ver_dict = {}
        for ver in res[index]:
            res_array, cc_vector = deal_feature(ver)
            if res_array is not None and cc_vector is not None:
                ver_one = {}
                ver_one['res_array'] = res_array
                ver_one['cc_vector'] = cc_vector
                ver_dict[os.path.basename(ver[0])] = ver_one
        # 随机选取1/3版本作为训练集，随机选择5次
        trains_data_key = one_third(ver_dict)
        r = 0
        p = 0
        fp = 0
        f1 = 0
        for train_index in trains_data_key:
            train_data = list()
            train_target = np.array([], dtype=np.intc)
            # 遍历获取训练集、训练集结果；测试集、测试集结果
            test_dict = copy.deepcopy(ver_dict)
            for ver_index in train_index:
                array = ver_dict[ver_index]['res_array']
                vector = ver_dict[ver_index]['cc_vector']
                test_dict.pop(ver_index)
                train_data.append(array)
                train_target = np.hstack([train_target, vector])
            train_sample = np.concatenate(train_data, axis=0)
            recall,pre,fpr,f1_score = decision_tree(train_sample, train_target, test_dict,pro_name, res_path)
            r = r + recall
            p = p + pre
            fp = fp + fpr
            f1 = f1 + f1_score
        creat_res_file(pro_name,[['all_avg',r/5, p/5, fp/5, f1/5]])
        print("s")
        # 执行决策树
        #decision_tree(all_sample, cc_res_vec, pro_name,res_path)


if __name__ =="__main__":

    # execution([['D:\\CC\\data_test\\data\\Math\\101b','D:\\CC\\data_test\\other\\Math\\101b'],'D:\\CC\\data_test\\error'])
    #linux path

    root = '/home/tianshuaihua/dataset'
    data = '/home/tianshuaihua/pydata'
    error_pro_ver = '/home/tianshuaihua/error'
    res_path = '/home/tianshuaihua/res'

    machine_learning(root,data,error_pro_ver,res_path)

    #test = Tool_io.checkAndLoad('/home/tianshuaihua/pydata/Chart/1b', "data_Coverage_InVector_saveAgain")

    # 获取程序路径
    # res = Tool_io.create_data_folder(root, data)
    # for index in res:
    #     pro_name = os.path.basename(index)
    #     # 特征矩阵
    #     lst = list()
    #     # 测试用例
    #     cc_res_vec = np.array([], dtype=np.intc)
    #     for ver in res[index]:
    #         res_array, cc_vector = deal_feature(ver)
    #         if res_array is not None and cc_vector is not None:
    #             lst.append(res_array)
    #             cc_res_vec = np.hstack([cc_res_vec,cc_vector])
    #     all_sample = np.concatenate(lst, axis=0)
    #     decision_tree(all_sample, cc_res_vec, pro_name)

