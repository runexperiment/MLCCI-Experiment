import shutil
from asyncio import as_completed
from concurrent.futures import ThreadPoolExecutor

import copy
import pickle
import random
import os
import numpy as np
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
def deal_feature(ver):
    # 获取realcc和测试用例数量,字典（4:失败测试用例数量;5:通过测试用例数量;6:cc测试用例(字典))
    statistic_res = Tool_io.checkAndLoad(ver[1], "data_Coverage_InVector_saveAgain")
    if statistic_res is not None:
        statistic_res[1] = 0
        Tool_io.checkAndSave(ver[1], "data_Coverage_InVector_saveAgain", statistic_res, enforce=True)
        # 获取归一化后的特征
        res_nor = Tool_io.checkAndLoad(ver[1], "normalization")
        fault = []
        fault_loc = Tool_io.checkAndLoad(ver[0], "faultHuge.in")
        if fault_loc is not None:
            for jfile in fault_loc:
                for index in fault_loc[jfile]:
                    fault.append(index)
        if res_nor is not None and len(fault) > 0:
            res_array = np.array(res_nor['normal'])
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
    return recall,precision,FPR,f1_score


# 决策树
def decision_tree_Liuyi(programName,versionindex, ver_dict_total, trains_data_key_total,treeNum,depth):
    # print("start",programName,versionindex,treeNum,depth)
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

    try:
        clf = RandomForestClassifier(random_state=0,
                                     n_estimators=treeNum,
                                     max_depth=depth,
                                     n_jobs=18)
        clf = clf.fit(train_sample, train_target)

        recall_topsum = 0
        precision_topsum = 0
        FPR_topsum = 0
        f1_score_topsum = 0
        count_topsum = 0

        printInfo={}

        for key in test_dict:
            recall_sum = 0
            precision_sum = 0
            FPR_sum = 0
            f1_score_sum = 0

            for ver in test_dict[key]:
                threadname = "thread" + str(ver)
                res_path = str(ver) + '_'

                recall,precision,FPR,f1_score=predictSingle(ver_dict_total,clf,key,ver)

                recall_sum = recall_sum + recall
                precision_sum = precision_sum + precision
                FPR_sum = FPR_sum + FPR
                f1_score_sum = f1_score_sum + f1_score

            recall_topsum += recall_sum
            precision_topsum += precision_sum
            FPR_topsum += FPR_sum
            f1_score_topsum += f1_score_sum

        thismax = 0 - FPR_topsum
        return versionindex,clf,recall_topsum,precision_topsum,FPR_topsum,f1_score_topsum,thismax

    except Exception as e:
        print("出错了",e)


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


# 存储结果
def creat_res_file(csv_name, res_path, content):
    path = os.path.join(res_path, csv_name) + ".csv"
    header = ['', 'recall', 'precision','FPrate','Fmeasure']
    with open(path, 'a+', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for row in content:
            writer.writerow(row)
        writer.writerow(['', '', '', '', ''])

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
def machine_learning(root,data,error_pro_ver,res_path,model_path,treeNum,depth):
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
        # 测试用例
        ver_dict = Tool_io.checkAndLoad(model_path,pro_name+"_dict")
        if ver_dict==None:
            ver_dict={}
            for ver in res[index]:
                print(pro_name,ver)
                res_array, cc_vector,origin_vector = deal_feature(ver)
                if res_array is not None and cc_vector is not None:
                    ver_one = {}
                    ver_one['res_array'] = res_array
                    ver_one['cc_vector'] = cc_vector
                    ver_one['origin_vector'] = origin_vector
                    ver_dict[os.path.basename(ver[0])] = ver_one
            Tool_io.checkAndSave(model_path,pro_name+"_dict",ver_dict)
        ver_dict_total[pro_name]=ver_dict
        # 随机选取30%版本作为训练集，随机选择5次
        trains_data_key = one_third(ver_dict,repeatTime,liuyi=True)
        if len(trains_data_key[0])>maxSampleNum:
            maxSampleNum=len(trains_data_key[0])
        trains_data_key_total[pro_name]=trains_data_key

    for key in trains_data_key_total:
        for index in range(len(trains_data_key_total[key])):
            while len(trains_data_key_total[key][index])<maxSampleNum:
                temppI=random.randint(0, len(trains_data_key_total[key][index])-1)
                trains_data_key_total[key][index].append(trains_data_key_total[key][index][temppI])

    r = 0
    p = 0
    fp = 0
    f1 = 0
    # 判断当前模型是否应该存储
    recall_old = 0
    pre_old = 0
    for programName in trains_data_key_total:

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
        for versionindex in range(len(trains_data_key_total[programName])):
            args1_list.append(programName)
            args2_list.append(versionindex)
            args3_list.append(ver_dict_total)
            args4_list.append(trains_data_key_total)
            args5_list.append(treeNum)
            args6_list.append(depth)

        print("process",programName)
        with ThreadPoolExecutor(18) as t:
            results = t.map(decision_tree_Liuyi, args1_list, args2_list, args3_list, args4_list, args5_list, args6_list)

        for versionindex,clf,recall,precision,FPR,f1_score,thismanx in results:
            versionStr=list(ver_dict_total[programName].keys())[versionindex]
            print("treeNum",treeNum,"depth",depth,"program",programName,"version",versionStr,"precision",precision,"FPR",FPR)
            clf_list[versionStr]=clf
            recall_sum = recall_sum + recall
            precision_sum = precision_sum + precision
            FPR_sum = FPR_sum + FPR
            f1_score_sum = f1_score_sum + f1_score
            thismax_sum=thismax_sum+thismanx
            printInfo="treeNum"+str(treeNum)+"\tdepth"+str(depth)+"\tprogram"+str(programName)+"\tversion"+str(versionStr)+"\t"+str(recall)+"\t"+str(precision)+"\t"+str(FPR)+"\t"+str(f1_score)+"\t"+str(thismanx)
            with open(saveDetailPath, "a") as file:
                file.write(printInfo+"\n")

        recall_save = recall_sum/len(trains_data_key_total[programName])
        precision_save = precision_sum/len(trains_data_key_total[programName])
        FPR_save = FPR_sum/len(trains_data_key_total[programName])
        f1_score_save = f1_score_sum/len(trains_data_key_total[programName])
        thismax_save = thismax_sum/len(trains_data_key_total[programName])

        if thismax_save > thisProjectMaxSave:
            print("treeNum",treeNum,"depth","更好",depth,"program",programName, recall_save, precision_save , FPR_save,f1_score_save)
            with open(thisProjectMaxSaveParameter, "a") as file:
                file.write("treeNum"+str(treeNum)+"\tdepth"+str(depth) + "\n")
            with open(os.path.join(this_project_model_path, 'maxsave'), 'wb') as f:
                pickle.dump(thismax_save, f)
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
    res_path = '/home/tianshuaihua/res'
    model_path = '/home/tianshuaihua/model_liuyi'
    m_path = '/home/tianshuaihua/model'
    origin_path = '/home/tianshuaihua/origin'

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    for treeNum in range(65, 115, 10):
        for depth in range(40,90,10):
            # for min_split in range(2,110,10):
            print("treeNum",treeNum,"depth",depth)
            machine_learning(root,data,error_pro_ver,res_path,model_path,treeNum,depth)

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

