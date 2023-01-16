import os
import pickle

import Tool_io
from multiprocessing import Pool
import methodFuzzy
from tqdm import trange
import numpy as np
import csv
import sklearn
from sklearn.metrics import confusion_matrix


# 读取一个文件夹，返回里面所有的文件名
def get_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        return dirs
    return []


# 模糊算法模型
def fuzzy_method(param):
    verpath = param[0]
    formulas = param[1]
    # 存在结果，直接返回
    FCCI_CC = Tool_io.checkAndLoad(verpath, "FCCI_CC")
    if FCCI_CC != None:
        return FCCI_CC
    matlab_fun = methodFuzzy.initialize()
    static_fcci = Tool_io.checkAndLoad(verpath, 'data_STATIC_baseline')
    if static_fcci != None:
        SS = Tool_io.checkAndLoad(verpath, 'data_SS_all')
        CR = Tool_io.checkAndLoad(verpath, 'data_CR_all')
        SF = Tool_io.checkAndLoad(verpath, 'data_SF_all')
        FM = Tool_io.checkAndLoad(verpath, 'data_FM_all')

        res = {}
        for formula in formulas:
            if formula != 'ochiai':
                continue
            # 存储归一化值
            fir = Tool_io.checkAndLoad(verpath, "SS_"+formula)
            if fir == None:
                fir = min_max01(SS[formula])
                Tool_io.checkAndSave(verpath, "SS_"+formula, fir)

            sec = Tool_io.checkAndLoad(verpath, "CR_" + formula)
            if sec == None:
                sec = min_max01(CR[formula])
                Tool_io.checkAndSave(verpath, "CR_" + formula, sec)

            thi = Tool_io.checkAndLoad(verpath, "SF_" + formula)
            if thi == None:
                thi = min_max01(SF[formula])
                Tool_io.checkAndSave(verpath, "SF_" + formula, thi)

            fou = Tool_io.checkAndLoad(verpath, "FM_" + formula)
            if fou == None:
                fou = min_max01(FM[formula])
                Tool_io.checkAndSave(verpath, "FM_" + formula, fou)

            fif = Tool_io.checkAndLoad(verpath, "Stat_" + formula)
            if fif == None:
                fif = min_max01(static_fcci[formula])
                Tool_io.checkAndSave(verpath, "Stat_" + formula, fir)

            res_list = []
            for index in trange(len(SS[formula])):
                one = float(fir[index])
                two = float(sec[index])
                three = float(thi[index])
                four = float(fou[index])
                five = float(fif[index])
                cal = matlab_fun.methodFuzzy(one, two, three, four, five)
                res_list.append(cal)
            res[formula] = res_list
        Tool_io.checkAndSave(verpath, "FCCI_CC", res)



# FCCI CC识别
def cc_identity(data,formula):
    programs = get_folder(data)
    # 遍历程序
    for pro_name in programs:
        pro_path = os.path.join(data, pro_name)
        vers = get_folder(pro_path)
        # 遍历程序版本
        pool = Pool(processes=8)
        for ver_name in vers:
            ver_path = os.path.join(pro_path, ver_name)
            param = []
            param.append(ver_path)
            param.append(formula)
            if pro_name != 'Closure':
                continue
            pool.apply_async(fuzzy_method, (param,))
        pool.close()
        pool.join()


# 0,1归一化
def min_max01(data_dict):
    arr = []
    for k in data_dict:
        arr.append(data_dict[k])
    res = []
    for x in arr:
        if np.max(arr)-np.min(arr) != 0:
            x = float(x - np.min(arr)) / (np.max(arr) - np.min(arr))
            res.append(x)
        else:
            res.append(0)
    return res


# 计算cc概率
def cal_cc(data_path, model_path, res_path,ver_data):
    programs = get_folder(data)
    # 遍历所有程序
    for pro_name in programs:
        if pro_name != 'Chart':
            continue
        print(pro_name)
        pro_path = os.path.join(data_path, pro_name)
        vers = get_folder(pro_path)
        # 遍历程序版本
        # f = open(os.path.join(model_path, pro_name + '.td'), 'rb')
        # td = pickle.load(f)
        td = ver_data
        deal_ver_cc = {}
        for ver_name in vers:
            if ver_name not in td[pro_name]:
                continue
            print(ver_name)
            ver_path = os.path.join(pro_path, ver_name)
            ver_formula = deal_cc(ver_path, res_path)
            deal_ver_cc[ver_name] = ver_formula
        complete_data = deal_allver(deal_ver_cc)
        deal_csv(pro_name,complete_data,res_path)


# 存储结果
def deal_csv(csv_name, content,res_path):
    path = os.path.join(res_path, 'FCCI_'+csv_name) + ".csv"
    header = ['name', 'ochiai_recall', 'ochiai_precision','ochiai_FPrate','ochiai_Fmeasure']
    with open(path, 'a+', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for row in content:
            writer.writerow(row)
        writer.writerow(['', '', '', '', ''])


# 处理所有版本
def deal_allver(deal_ver_cc):
    ver_all = []
    ver_name = []
    for ver in deal_ver_cc:
        ver_list = []
        ver_name.append(ver)
        formulas = deal_ver_cc[ver]
        for formula in formulas:
            ver_list = ver_list + formulas[formula]
        ver_all.append(ver_list)
    ver_name.append('avg')
    # 处理为二维形式
    ver_n = np.array([ver_name]).T
    deal_ver_all = np.array(ver_all)
    # 求均值
    avg = np.average(deal_ver_all, axis=0)
    # 纵向拼接 均值
    num_info = np.vstack((deal_ver_all, avg))
    # 横向拼接 版本号
    complete_data = np.hstack((ver_n,num_info))
    return complete_data


# 计算cc结果
def deal_cc(ver_path,res_path):
    cc_res = Tool_io.checkAndLoad(ver_path, 'FCCI_CC')
    info = Tool_io.checkAndLoad(ver_path, 'data_Coverage_InVector_saveAgain')
    real_cc = info[6]
    fail_test = info[7]
    fail_test = sorted(fail_test)
    ver_formula = {}
    # 真实cc
    origin = np.zeros(len(info[3]))
    for re in real_cc:
        origin[re] = 1
    for re in fail_test:
        origin[re] = 1
    cc_all = {}
    # 遍历怀疑度公式
    for formula in cc_res:
        # fcci识别结果
        cc = []
        # 遍历各个测试用例
        for test in cc_res[formula]:
            if test >= 0.5:
                cc.append(1)
            else:
                cc.append(0)
        #res_origin, fcci_cc = del_fail(origin, cc, fail_test)

        save_pre = Tool_io.checkAndLoad(ver_path, "fcci_predict")
        if save_pre == None:
            Tool_io.checkAndSave(ver_path, "fcci_predict", cc)

        cc_all[formula] = cc
        res = confusion_matrix(origin, cc)
        if res.shape[0] == 1:
            res = np.array([[res[0,0],0],[0,0]])
        recall, precision, FPR, f1_score = cal_metric(res)
        ver_formula[formula] = [recall, precision, FPR, f1_score]
    return ver_formula


# 去除失败测试用例影响
def del_fail(origin, cc, fail_test):
    res_origin = [origin[i] for i in range(len(origin)) if (i not in fail_test)]
    fcci_cc = [cc[i] for i in range(len(cc)) if (i not in fail_test)]
    return res_origin, fcci_cc


# 计算cc指标
def cal_metric(res):
    TP = res[1, 1]
    TN = res[0, 0]
    FP = res[0, 1]
    FN = res[1, 0]
    #print(res)
    if TP + FN == 0:
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

    if(TP + FP + FN == 0):
        f1_score = 0
    else:
        f1_score = 2 * TP / (2 * TP + FP + FN)
    return recall, precision, FPR, f1_score


if __name__ == "__main__":

    # matlab_fun = methodFuzzy.initialize()
    # cal = matlab_fun.methodFuzzy(0.4, 0.1, 0.3, 0.2, 0.6)
    # print(cal)
    # cal = matlab_fun.methodFuzzy(0.6, 0.7, 0.5, 0.2, 0.6)
    # print(cal)
    # cal = matlab_fun.methodFuzzy(0.1, 0.1, 0.6, 0.7, 0.6)
    # print(cal)

    data = '/home/tianshuaihua/tpydata'
    model_path = '/home/tianshuaihua/model'
    res_path = '/home/tianshuaihua/fcci/res'

    vers_path = '/home/tianshuaihua/wang/fea'
    ver_data = Tool_io.checkAndLoad(vers_path, 'vers.in')

    formula = ['ochiai','dstar','op2','dstar2']
    #cc_identity(data,formula)

    cal_cc(data,model_path,res_path,ver_data)
