import math
import os

# 读取一个文件夹，返回里面所有的文件名
import pickle
import sys

import Tool_io
from tqdm import trange

# 获取文件工具类
def get_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        return files
    return []


# 处理cc计算结果
def deal_cc():
    print("s")


# 计算预测值
def cal_predict(root,data,model_path,p_name,res_path,originPath):
    # 加载模型
    f = open(os.path.join(model_path, 'All.mod'), 'rb')
    model = pickle.load(f)
    # 加载要预测的数据
    f = open(os.path.join(model_path, 'All.td'), 'rb')
    td = pickle.load(f)
    # 程序路径
    pn_pydata = os.path.join(data,p_name)
    pn_dataset = os.path.join(root,p_name)
    origin_dataset = os.path.join(originPath,p_name)
    for ver in td[p_name]:
        print(p_name,ver)

        sus_path = os.path.join(os.path.join(data,p_name),ver)

        sus_value = Tool_io.checkAndLoad(sus_path, "sus_value_function")
        if sus_value != None:
            print("finished", "skip")
            continue

        vn_pydata = os.path.join(pn_pydata,ver)
        res = Tool_io.checkAndLoad(vn_pydata, "data_Coverage_InVector_saveAgain")
        origin_dataset_version=os.path.join(origin_dataset,ver)
        originalCoverage = Tool_io.checkAndLoad(origin_dataset_version, "CoverageMatrix_Function.in")
        if originalCoverage==None:
            print("lake of original coverage","skip")
            continue
        # 失败测试用例1
        fault_index = res[7]
        X_test = td[p_name][ver]['res_array']
        Y_test = model.predict(X_test)
        # 将失败测试用例结果改为2，cc为1
        for val in fault_index:
            Y_test[val] = 2
        # 获取真实错误位置
        # fault = []
        # vn_dataset = os.path.join(pn_dataset,ver)
        # fault_loc = Tool_io.checkAndLoad(vn_dataset, "faultHuge.in")
        # for jfile in fault_loc:
        #     for index in fault_loc[jfile]:
        #         fault.append(index)
        # 计算怀疑度


        cal_sus(len(originalCoverage), len(originalCoverage[0]), originalCoverage, Y_test, sus_path)
    #     print("s")
    # print("s")

project = {
    "Chart": 26,
    "Closure": 176,
    "Lang": 65,
    "Math": 106,
    "Mockito": 38,
    "Time": 27,
    "Cli": 39,
    "Codec": 18,
    "Collections": 4,
    "Compress": 47,
    "Csv": 16,
    "Gson": 18,
    "JacksonCore": 26,
    "JacksonDatabind": 112,
    "JacksonXml": 6,
    "Jsoup": 93,
    "JxPath": 22,
}

# 获取要预测的程序信息
def get_info(root,data,error_pro_ver,res_path,model_path,originPath):
    # files = get_files(model_path)
    # pro = []
    # for file in files:
    #     name = file.split(".")[0]
    #     pro.append(name)
    # program = list(set(pro))
    for p_name in project:
        cal_predict(root,data,model_path,p_name,res_path,originPath)
    # print(program)


# 计算怀疑度
def cal_sus(case_num,statement_num,covMatrix,inVector,sus_path):

    #失败的测试用例
    tf = 0
    #成功的测试用例
    tp = 0
    #cc测试用例
    cc = 0
    for case_index in range(case_num):
        if inVector[case_index] == 1:
            cc += 1
        elif inVector[case_index] == 0:
            tp += 1
        else:
            tf += 1

    #执行语句e且通过的测试用例数量
    aef = {}
    #执行语句e且未通过该的测试用例数量
    aep = {}
    #未执行语句e且未通过该的测试用例数量
    anf = {}
    #执行语句e且通过该的测试用例数量
    anp = {}
    # 执行了语句e的cc测试用例
    ce = {}
    # 未执行语句e的cc测试用例
    cn = {}
    for statement_index in trange(statement_num):
        aef[statement_index] = 0
        aep[statement_index] = 0
        anf[statement_index] = 0
        anp[statement_index] = 0
        ce[statement_index] = 0
        cn[statement_index] = 0
        for case_index in range(case_num):
            current_int = covMatrix[case_index][statement_index]
            if current_int == 1:
                if inVector[case_index] == 1:
                    ce[statement_index] += 1
                    aep[statement_index] += 1
                elif inVector[case_index] == 0:
                    aep[statement_index] += 1
                else:
                    aef[statement_index] += 1
            else:
                if inVector[case_index] == 1:
                    cn[statement_index] += 1
                    anp[statement_index] += 1
                elif inVector[case_index] == 0:
                    anp[statement_index] += 1
                else:
                    anf[statement_index] += 1
    formulaSus = SBFL_location_cases(aef, aep, anf, anp, covMatrix, tf, tp, ce, cn)
    Tool_io.checkAndSave(sus_path, "sus_value_function", formulaSus)
    return formulaSus


def SBFL_location_cases(aefList,aepList,anfList,anpList,covMatrix,tf,tp,ceList,cnList):
    statement_num = len(covMatrix[0])
    sus_oc_zero = {}
    sus_oc_one = {}
    sus_oc_two = {}
    sus_oc_three = {}
    sus_ds_zero = {}
    sus_ds_one = {}
    sus_ds_two = {}
    sus_ds_three = {}
    for statement_index in range(statement_num):
            aef = aefList[statement_index]
            aep = aepList[statement_index]
            anf = anfList[statement_index]
            anp = anpList[statement_index]
            ce = ceList[statement_index]
            cn = cnList[statement_index]

            # 原始公式
            oc_zero = cal_ochiai(tf, tp, aef, aep, anf, anp)
            sus_oc_zero[statement_index] = oc_zero

            ds_zero = cal_dstar(tf, tp, aef, aep, anf, anp, 3)
            sus_ds_zero[statement_index] = ds_zero


            # 清洗策略
            aepp = aep - ce

            oc_one = cal_ochiai(tf, tp, aef, aepp, anf, anp)
            sus_oc_one[statement_index] = oc_one

            ds_one = cal_dstar(tf, tp, aef, aepp, anf, anp, 3)
            sus_ds_one[statement_index] = ds_one

            # 重标策略
            aepp2 = aep - ce
            aeff2 = aef + ce

            oc_two = cal_ochiai(tf, tp, aeff2, aepp2, anf, anp)
            sus_oc_two[statement_index] = oc_two

            ds_two = cal_dstar(tf, tp, aeff2, aepp2, anf, anp, 3)
            sus_ds_two[statement_index] = ds_two

            # 交换策略
            aepp3 = aep - ce
            anpp3 = anp - cn
            aeff3 = aef + ce
            anff3 = anf + cn

            oc_three = cal_ochiai(tf, tp, aeff3, aepp3, anff3, anpp3)
            sus_oc_three[statement_index] = oc_three

            ds_three = cal_dstar(tf, tp, aeff3, aepp3, anff3, anpp3, 3)
            sus_ds_three[statement_index] = ds_three

    sus_oc_zero = sorted(sus_oc_zero.items(), key=lambda d: d[1], reverse=True)
    sus_oc_one = sorted(sus_oc_one.items(), key=lambda d: d[1], reverse=True)
    sus_oc_two = sorted(sus_oc_two.items(), key=lambda d: d[1], reverse=True)
    sus_oc_three = sorted(sus_oc_three.items(), key=lambda d: d[1], reverse=True)
    sus_ds_zero = sorted(sus_ds_zero.items(), key=lambda d: d[1], reverse=True)
    sus_ds_one = sorted(sus_ds_one.items(), key=lambda d: d[1], reverse=True)
    sus_ds_two = sorted(sus_ds_two.items(), key=lambda d: d[1], reverse=True)
    sus_ds_three = sorted(sus_ds_three.items(), key=lambda d: d[1], reverse=True)

    formulaSus = {'ochiai':sus_oc_zero,"ochiai_c": sus_oc_one,'ochiai_r':sus_oc_two,'ochiai_e':sus_oc_three,
                  'ds':sus_ds_zero,"ds_c":sus_ds_one,'ds_r':sus_ds_two,'ds_e':sus_ds_three}

    return formulaSus


# ochiai
def cal_ochiai(tf, tp, aef, aep, anf, anp):
    a = aef + aep
    b = math.sqrt(tf * a)
    # ochiai值
    e = 0
    if b != 0:
        e = aef / b
    return e


# jaccard
def cal_jaccard(tf, tp, aef, aep, anf, anp):
    a = tf+aep
    if a == 0:
        return 0
    b = aef/a
    return b


# dstar
def cal_dstar(tf, tp, aef, aep, anf, anp, index):
    a = aep + (tf - aef)
    if a == 0:
        if aef>0:
            return sys.maxsize
        else:
            return 0
    b = math.pow(aef, index)
    c = b / a
    return c


# gp13
def cal_gp13(tf, tp, aef, aep, anf, anp):
    a = 2 * aep + aef
    d = 0
    if a != 0:
        c = aef / a
        # gp13公式
        d = aef + c
    return d


# op2
def cal_op2(tf, tp, aef, aep, anf, anp):
    a = aep / (tp + 1)
    b = aef - a
    return b


if __name__ =="__main__":

    originPath="/home/wuyonghao/Defeats4JFile/outputClean"
    root = '/home/wuyonghao/CCIdentifyFile/dataset'
    data = '/home/wuyonghao/CCIdentifyFile/pydata'
    error_pro_ver = '/home/wuyonghao/CCIdentifyFile/error'
    res_path = '/home/wuyonghao/CCIdentifyFile/res'
    model_path = '/home/wuyonghao/CCIdentifyFile/model'

    get_info(root,data,error_pro_ver,res_path,model_path,originPath)