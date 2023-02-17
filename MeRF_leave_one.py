import shutil
from asyncio import as_completed
from concurrent.futures import ThreadPoolExecutor

import copy
import pickle
import random
import os
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
import csv
import Tool_io
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")


# 统计特征结果
def deal_feature(ver, proname, origin_path, flag):
    # 获取realcc和测试用例数量,字典（4:失败测试用例数量;5:通过测试用例数量;6:cc测试用例(字典))
    statistic_res = Tool_io.checkAndLoad(ver[1], "data_Coverage_Info")
    if statistic_res is not None:
        res_nor = None
        # 获取归一化后的特征
        if flag == 1:
            res_nor = Tool_io.checkAndLoad(ver[1], "normalization_other")
        elif flag == 2:
            res_nor = Tool_io.checkAndLoad(ver[1], "normalization2")
        elif flag == 3:
            origin_data = Tool_io.checkAndLoad(origin_path, proname+'_'+"data0")
            if os.path.basename(ver[1]) in origin_data:
                res_nor = origin_data[os.path.basename(ver[1])]
        elif flag == 4:
            origin_data = Tool_io.checkAndLoad(origin_path, proname+'_'+"data")
            if os.path.basename(ver[1]) in origin_data:
                res_nor = origin_data[os.path.basename(ver[1])]
        fault = []
        fault_loc = Tool_io.checkAndLoad(ver[0], "faultHuge.in")
        if fault_loc is not None:
            for jfile in fault_loc:
                for index in fault_loc[jfile]:
                    fault.append(index)
        if res_nor is not None and len(fault) > 0:
            if flag == 1 or flag == 2:
                res_array = np.array(res_nor['normal'])
            else:
                res_array = np.array(res_nor['sample'].T)
            # 记录测试用例是否为cc
            cc_vector = np.zeros(statistic_res[4] + statistic_res[5], dtype=np.int)
        else:
            return None, None, None
        for cc_index in statistic_res[6]:
            cc_vector[cc_index] = 1
        for fault_index in statistic_res[7]:
            cc_vector[fault_index] = 1
        return res_array, cc_vector,statistic_res[3]
    else:
        return None, None, None


def predictSingle(ver_dict_total,clf,key,ver):
    # print(key,ver)
    X_test = ver_dict_total[key][ver]['res_array']
    y_test = ver_dict_total[key][ver]['cc_vector']
    score = clf.score(X_test, y_test)
    y_predicate = clf.predict(X_test)
    res = confusion_matrix(y_test, y_predicate)
    cmatrix = sklearn.metrics.classification_report(y_test, y_predicate)
    # 混淆矩阵参数
    TP = res[1, 1]
    TN = res[0, 0]
    FP = res[0, 1]
    FN = res[1, 0]
    # print(res)
    # print(cmatrix)
    # print(key, ':', score)
    # cc识别指标
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
    f1_score = 2 * TP / (2 * TP + FP + FN)
    return recall,precision,FPR,f1_score,y_predicate


# 决策树
def decision_tree_Liuyi(programName,versionindex, ver_dict_total, trains_data_key_total,treeNum,depth,m_path, result_path):
    print(result_path)
    # print("start",programName,versionindex,treeNum,depth)
    print(f"正在训练 {programName} {versionindex} {treeNum} {depth}")
    train_data = list()
    train_target = np.array([], dtype=np.intc)
    test_dict = {}

    additionFeature = np.array([])
    additionLabel = np.array([])
    columN = -1

    # for key in ver_dict_total:
    key = programName
    test_dict[key] = {}
    test_dict_temp = copy.deepcopy(ver_dict_total[key])
    for ver_index in trains_data_key_total[key][versionindex]:
        # 遍历获取训练集、训练集结果；测试集、测试集结果
        array = ver_dict_total[key][ver_index]['res_array']
        vector = ver_dict_total[key][ver_index]['cc_vector']
        if ver_index in test_dict_temp:
            test_dict_temp.pop(ver_index)
        train_data.append(array)
        if columN == -1:
            columN = array.shape[1]
        train_target = np.hstack([train_target, vector])

    for tempKey in test_dict_temp:
        test_dict[key][tempKey] = test_dict_temp[tempKey]

        for Tindex in range(len(test_dict_temp[tempKey]['origin_vector'])):
            if test_dict_temp[tempKey]['origin_vector'][Tindex] == 1:
                additionFeature = np.append(additionFeature, test_dict_temp[tempKey]['res_array'][[Tindex]])
                additionLabel = np.append(additionLabel, 1)

    additionFeature = additionFeature.reshape(-1, columN)
    train_sample = np.concatenate(train_data, axis=0)

    train_sample = np.append(train_sample, additionFeature).reshape(-1, columN)
    train_target = np.hstack([train_target, additionLabel])

    ver_name = None
    for ver in test_dict[key]:
        ver_name = ver
    pro_dir = os.path.join(m_path, programName)
    if not os.path.exists(pro_dir):
        os.mkdir(pro_dir)
    ppath = os.path.join(pro_dir, programName + ver_name + '.mod')
    ppres = os.path.join(pro_dir, programName + ver_name + '.res')
    model = None
    if os.path.exists(ppath):
        f = open(ppath, 'rb')
        model = pickle.load(f)

    try:
        if model is None:
            clf = RandomForestClassifier(random_state=0,
                                         n_estimators=treeNum,
                                         max_depth=depth,
                                         n_jobs=18)
            clf = clf.fit(train_sample, train_target)
        else:
            clf = model

        recall_topsum = 0
        precision_topsum = 0
        FPR_topsum = 0
        f1_score_topsum = 0
        count_topsum = 0
        ver_one = ''

        printInfo={}

        for key in test_dict:
            recall_sum = 0
            precision_sum = 0
            FPR_sum = 0
            f1_score_sum = 0

            for ver in test_dict[key]:
                threadname = "thread" + str(ver)
                res_path = str(ver) + '_'
                ver_one = ver

                recall,precision,FPR,f1_score,y_predicate=predictSingle(ver_dict_total,clf,key,ver)

                recall_sum = recall_sum + recall
                precision_sum = precision_sum + precision
                FPR_sum = FPR_sum + FPR
                f1_score_sum = f1_score_sum + f1_score

            recall_topsum += recall_sum
            precision_topsum += precision_sum
            FPR_topsum += FPR_sum
            f1_score_topsum += f1_score_sum

        thismax = 0 - FPR_topsum
        print(f"训练完毕 {programName} {versionindex} {treeNum} {depth}")

        if not os.path.exists(ppath):
            with open(ppath, 'wb') as f:
                pickle.dump(clf, f)

        if not os.path.exists(ppres):
            with open(ppres, 'wb') as f:
                pickle.dump(y_predicate, f)

        content = []
        content.append([ver_one, recall_topsum, precision_topsum, FPR_topsum, f1_score_topsum])
        creat_res_file(result_path, programName, content)

        return versionindex,clf,recall_topsum,precision_topsum,FPR_topsum,f1_score_topsum,thismax

    except Exception as e:
        print("出错了",e)

# 计算均值
def cal_avg(res_path, csv_name):
    path = os.path.join(res_path, csv_name) + ".csv"
    if os.path.exists(path):
        table = pd.read_csv(path)
        recall = table['recall'].mean()
        precision = table['precision'].mean()
        FPrate = table['FPrate'].mean()
        Fmeasure = table['Fmeasure'].mean()

        data = np.atleast_2d(np.array([recall, precision, FPrate, Fmeasure]))
        avg_data = pd.DataFrame(data, columns=list('0123'), index=['avg'])
        avg_data.to_csv(path, mode='a', header=False)

# 存储结果
def creat_res_file(res_path, csv_name, content):
    path = os.path.join(res_path, csv_name) + ".csv"
    flag = 0
    if not os.path.exists(path):
        flag = 1
    with open(path, 'a+', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if flag == 1:
            writer.writerow(['', 'recall', 'precision', 'FPrate', 'Fmeasure'])
        for row in content:
            writer.writerow(row)


# 随机抽取三分之一的训练集
def one_third(ver_dict,repeatTime,liuyi=False):
    res = []
    if liuyi:
        for index in ver_dict:
            temp=[]
            for index2 in ver_dict:
                if index!=index2:
                    temp.append(index2)
            res.append(temp)
    else:
        for i in range(repeatTime):
            length = int((len(ver_dict)*3)/10)
            train_data_key = random.sample(list(ver_dict), length)
            res.append(train_data_key)
    return res


projectOrigin = {
    "Chart": 26,
    "Closure": 176,
    "Lang": 65,
    "Math": 106,
    "Mockito": 38,
    "Time": 27,
    # "Cli": 39,
    # "Codec": 18,
    # "Collections": 4,
    # "Compress": 47,
    # "Csv": 16,
    # "Gson": 18,
    # "JacksonCore": 26,
    # "JacksonDatabind": 112,
    # "JacksonXml": 6,
    # "Jsoup": 93,
    # "JxPath": 22,
}

# 准备数据集和训练集
def machine_learning(root,data,error_pro_ver,res_path,model_path,treeNum,depth,origin_path, feature_index,m_path):
    # 获取程序路径
    res = Tool_io.create_data_folder(root, data)

    ver_dict_total={}
    trains_data_key_total={}
    repeatTime=5
    maxSampleNum=0

    for index in res:
        # 程序名称
        pro_name = os.path.basename(index)
        if pro_name not in projectOrigin:
            continue
        if pro_name != 'Closure':
            continue
        # 测试用例
        ver_dict = Tool_io.checkAndLoad(model_path,pro_name+f"_dict_{feature_index}")
        if ver_dict==None:
            print(f"{model_path}/{pro_name}_dict缓存没有内容，需要重新处理数据")
            ver_dict={}
            for ver in res[index]:
                print(pro_name,ver)
                res_array, cc_vector,origin_vector = deal_feature(ver, pro_name, origin_path, feature_index)
                if res_array is not None and cc_vector is not None:
                    ver_one = {}
                    ver_one['res_array'] = res_array
                    ver_one['cc_vector'] = cc_vector
                    ver_one['origin_vector'] = origin_vector
                    ver_dict[os.path.basename(ver[0])] = ver_one
            Tool_io.checkAndSave(model_path,pro_name+f"_dict_{feature_index}",ver_dict)
        ver_dict_total[pro_name]=ver_dict
        # 随机选取30%版本作为训练集，随机选择5次
        trains_data_key = one_third(ver_dict,repeatTime,liuyi=True)
        if len(trains_data_key[0])>maxSampleNum:
            maxSampleNum=len(trains_data_key[0])
        trains_data_key_total[pro_name]=trains_data_key

    r = 0
    p = 0
    fp = 0
    f1 = 0
    # 判断当前模型是否应该存储
    recall_old = 0
    pre_old = 0

    thismax = 0
    all_len = 0

    thisProjectMaxSaveAll = Tool_io.checkAndLoad(model_path, 'maxsaveAll')  # 所有程序结果求平均值最好的情况
    thisProjectMaxSaveParameterAll = os.path.join(model_path, 'maxsavePAll')  # 所有程序结果求平均值最好的情况
    saveDetailPathAll = os.path.join(model_path, "logDetailAll.txt")
    for programName in trains_data_key_total:
        print("开始训练 ")

        this_project_model_path=os.path.join(model_path,programName)
        if not os.path.exists(this_project_model_path):
            os.mkdir(this_project_model_path)
        thisProjectMaxSave = Tool_io.checkAndLoad(this_project_model_path, 'maxsave')

        thisProjectMaxSaveParameter = os.path.join(this_project_model_path, 'maxsaveP')

        saveLogPath = os.path.join(this_project_model_path, "log.txt")
        saveDetailPath = os.path.join(this_project_model_path, "logDetail.txt")

        modeSavePath=os.path.join(this_project_model_path, "mode")

        if thisProjectMaxSave == None:
            thisProjectMaxSave = -99999

        clf_list= {}
        recall_sum = 0
        precision_sum = 0
        FPR_sum = 0
        f1_score_sum = 0
        thismax_sum=0

        args1_list = []
        args2_list = []
        args3_list = []
        args4_list = []
        args5_list = []
        args6_list = []
        args7_list = []
        args8_list = []
        for versionindex in range(len(trains_data_key_total[programName])):
            args1_list.append(programName)
            args2_list.append(versionindex)
            args3_list.append(ver_dict_total)
            args4_list.append(trains_data_key_total)
            args5_list.append(treeNum)
            args6_list.append(depth)
            args7_list.append(m_path)
            args8_list.append(res_path)

        print("process",programName)
        creat_res_file(res_path, programName, [])
        with ThreadPoolExecutor(18) as t:
            results = t.map(decision_tree_Liuyi, args1_list, args2_list, args3_list, args4_list, args5_list, args6_list,args7_list, args8_list)
        print("训练完毕，接下来写日志啥的")
        cal_avg(res_path, programName)
        for versionindex,clf,recall,precision,FPR,f1_score,thismanx in results:
            versionStr=list(ver_dict_total[programName].keys())[versionindex]
            print("treeNum",treeNum,"depth",depth,"program",programName,"version",versionStr,"precision",precision,"FPR",FPR)
            clf_list[versionStr] = clf
            recall_sum = recall_sum + recall
            precision_sum = precision_sum + precision
            FPR_sum = FPR_sum + FPR
            f1_score_sum = f1_score_sum + f1_score
            thismax_sum = thismax_sum+thismanx
            # printInfo="treeNum"+str(treeNum)+"\tdepth"+str(depth)+"\tprogram"+str(programName)+"\tversion"+str(versionStr)+"\t"+str(recall)+"\t"+str(precision)+"\t"+str(FPR)+"\t"+str(f1_score)+"\t"+str(thismanx)

            printInfo = f"{datetime.now()} treeNum{treeNum}\tdepth{depth}\tprogram{programName}\tversion{versionStr}\t{recall}\t{precision}\t{FPR}\t{f1_score}\t{thismanx}\t{feature_index}"

            with open(saveDetailPath, "a") as file:
                file.write(printInfo+"\n")

        r += recall_sum
        p += precision_sum
        fp += FPR_sum
        f1 += f1_score_sum
        thismax += thismax_sum
        all_len += len(trains_data_key_total[programName])

        recall_save = recall_sum/len(trains_data_key_total[programName])
        precision_save = precision_sum/len(trains_data_key_total[programName])
        FPR_save = FPR_sum/len(trains_data_key_total[programName])
        f1_score_save = f1_score_sum/len(trains_data_key_total[programName])
        thismax_save = thismax_sum/len(trains_data_key_total[programName])



        if thismax_save > thisProjectMaxSave:
            print("treeNum",treeNum,"depth","更好",depth,"program",programName, recall_save, precision_save , FPR_save,f1_score_save)
            with open(thisProjectMaxSaveParameter, "a") as file:
                file.write("treeNum"+str(treeNum)+"\tdepth"+str(depth) + "\n") # 记录更好的 树的数量和深度
            with open(os.path.join(this_project_model_path, 'maxsave'), 'wb') as f:
                pickle.dump(thismax_save, f) # 更新thismax_save
            if os.path.exists(modeSavePath):
                shutil.rmtree(modeSavePath)
            os.mkdir(modeSavePath)
            for key in clf_list:
                tempPath=os.path.join(modeSavePath,key)
                with open(tempPath, 'wb') as f:
                    pickle.dump(clf_list[key], f)

            printInfo="treeNum"+str(treeNum)+"\tdepth"+str(depth)+"\t更好\t"+str(recall_save)+"\t"+str(precision_save)+"\t"+str(FPR_save)+"\t"+str(f1_score_save)+"\t"+str(thismax_save)
            with open(saveLogPath, "a") as file:
                file.write(printInfo+"\n")
        else:
            print("treeNum", treeNum, "depth", "不好", depth, "program", programName, recall_save, precision_save,
                  FPR_save, f1_score_save)

            printInfo="treeNum"+str(treeNum)+"\tdepth"+str(depth)+"\t不好\t"+str(recall_save)+"\t"+str(precision_save)+"\t"+str(FPR_save)+"\t"+str(f1_score_save)+"\t"+str(thismax_save)
            with open(saveLogPath, "a") as file:
                file.write(printInfo+"\n")

    # 将所有情况的结果求平均值，然后将最优的结果保存到 maxsaveAll
    thismax /= all_len
    r /= all_len
    p /= all_len
    fp /= all_len
    f1 /= all_len

    printInfo = f"{datetime.now()}\ttreeNum:{treeNum}\tdepth:{depth}\t{r}\t{p}\t{fp}\t{f1}\t{thismax}\t{feature_index}"

    with open(saveDetailPathAll, "a") as file:
        file.write(printInfo + "\n")
    if thisProjectMaxSaveAll == None:
        thisProjectMaxSaveAll = -99999

    if thismax > thisProjectMaxSaveAll:
        with open(thisProjectMaxSaveParameterAll, "a") as file:
            file.write(f"{datetime.now()}\ttreeNum:{treeNum}\tdepth:{depth}\n")  # 记录更好的 树的数量和深度
        with open(os.path.join(model_path, 'maxsaveAll'), 'wb') as f:
            pickle.dump(thismax, f)  # 更新thismax_save


if __name__ =="__main__":
    # execution([['D:\\CC\\data_test\\data\\Math\\101b','D:\\CC\\data_test\\other\\Math\\101b'],'D:\\CC\\data_test\\error'])
    #linux path

    # root = '/home/wuyonghao/CCIdentifyFile/dataset'
    # data = '/home/wuyonghao/CCIdentifyFile/pydata'
    # error_pro_ver = '/home/wuyonghao/CCIdentifyFile/error'
    # res_path = '/home/wuyonghao/CCIdentifyFile/res'
    # model_path = '/home/wuyonghao/CCIdentifyFile/model_liuyi'

    root = '/home/tianshuaihua/base_dataset'
    data = '/home/tianshuaihua/tpydata'
    error_pro_ver = '/home/tianshuaihua/error'
    res_path = '/home/tianshuaihua/res3'
    model_path = '/home/tianshuaihua/model_last3'
    m_path = '/home/tianshuaihua/model3'
    origin_path = '/home/tianshuaihua/origin'


    if not os.path.exists(model_path):
        os.mkdir(model_path)

    machine_learning(root, data, error_pro_ver, res_path, model_path, 85, 85, origin_path, 1, m_path)

    # # /usr/bin/python3 MeRF_留一法.py
    # for treeNum in range(65, 86, 10): # 65 115 10
    # # for treeNum in range(95, 116, 10):  # 65 115 10
    #     for depth in range(40,90,10): # 40 90 10
    #         # for min_split in range(2,110,10):
    #         for feature_index in range(1, 5):
    #             print("treeNum", treeNum, "depth", depth, "feature ", feature_index)
    #             machine_learning(root,data,error_pro_ver,res_path,model_path,treeNum,depth,origin_path, feature_index, m_path)

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

