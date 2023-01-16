import math
import os
import sys
from tqdm import trange
# 读取一个文件夹，返回里面所有的文件名
import Tool_io


def get_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        return dirs
    return []


def base_info(root, data, error_pro_ver, res_path,originPath):
    all = Tool_io.create_data_folder(root, data)
    for pros in all:
        origin_dataset = os.path.join(originPath, os.path.basename(pros))
        for ver in all[pros]:
            print(ver)
            WPCC = Tool_io.checkAndLoad(ver[1], "fuzzy_knn")
            if WPCC is not None:
                # res = Tool_io.checkAndLoad(ver[1], "data_Coverage_InVector_saveAgain")
                origin_dataset_version = os.path.join(origin_dataset, os.path.basename(ver[0]))
                originalCoverage = Tool_io.checkAndLoad(origin_dataset_version, "CoverageMatrix_Function.in")
                vector = Tool_io.checkAndLoad(ver[0], "inVector.in")
                cal_sus(len(originalCoverage), len(originalCoverage[0]), originalCoverage, vector, ver[1],WPCC)
                print("s")


# 计算怀疑度
def cal_sus(case_num,statement_num,covMatrix,inVector,sus_path,WPCC):

    sus_value = Tool_io.checkAndLoad(sus_path, "sus_value_function")
    if sus_value != None:
        return sus_value

    # 失败的测试用例
    tf = 0
    # 成功的测试用例
    tp = 0
    for case_index in range(case_num):
        if inVector[case_index] == 1:
            tf += 1
        else:
            tp += 1
    # 执行语句e且通过的测试用例数量
    aef = {}
    # 执行语句e且未通过该的测试用例数量
    aep = {}
    # 未执行语句e且未通过该的测试用例数量
    anf = {}
    # 执行语句e且通过该的测试用例数量
    anp = {}
    # 执行语句e的cc概率
    cc = {}
    for statement_index in trange(statement_num):
        aef[statement_index] = 0
        aep[statement_index] = 0
        anf[statement_index] = 0
        anp[statement_index] = 0
        cc[statement_index] = 0
        for case_index in range(case_num):
            current_int = covMatrix[case_index][statement_index]
            if current_int == 1:
                if WPCC.__contains__(statement_index):
                    cc[statement_index] += WPCC[statement_index]
                else:
                    cc[statement_index] += 0
                if inVector[case_index] == 1:
                    aef[statement_index] += 1
                else:
                    aep[statement_index] += 1
            else:
                if inVector[case_index] == 1:
                    anf[statement_index] += 1
                else:
                    anp[statement_index] += 1
    formulaSus = SBFL_location(aef, aep, anf, anp, covMatrix, tf, tp, cc,WPCC)
    Tool_io.checkAndSave(sus_path, "sus_value_function", formulaSus)
    return formulaSus

def SBFL_location(aefList,aepList,anfList,anpList,covMatrix,tf,tp,ccList,WPCC):
    statement_num = len(covMatrix[0])
    sus_ds_one = {}
    sus_ds_two = {}
    sus_ds_three = {}
    for statement_index in range(statement_num):
            aef = aefList[statement_index]
            aep = aepList[statement_index]
            anf = anfList[statement_index]
            anp = anpList[statement_index]
            if ccList.__contains__(statement_index):
                cc = ccList[statement_index]
            else:
                cc = 0

            # 清洗策略
            ds_one = cal_dstar_c(tf, tp, aef, aep, anf, anp, 3,cc)
            sus_ds_one[statement_index] = ds_one

            # 重标策略
            ds_two = cal_dstar_r(tf, tp, aef, aep, anf, anp, 3,cc)
            sus_ds_two[statement_index] = ds_two

            # 交换策略
            cc_sum = 0
            for dict in WPCC:
                cc_sum = cc_sum + WPCC[dict]
            ds_three = cal_dstar_e(tf, tp, aef, aep, anf, anp, 3,cc,cc_sum)
            sus_ds_three[statement_index] = ds_three

    sus_ds_one = sorted(sus_ds_one.items(), key=lambda d: d[1], reverse=True)
    sus_ds_two = sorted(sus_ds_two.items(), key=lambda d: d[1], reverse=True)
    try:
        sus_ds_three = sorted(sus_ds_three.items(), key=lambda d: d[1], reverse=True)
    except:
        return {"ds_c":sus_ds_one,'ds_r':sus_ds_two,'ds_error':sus_ds_three}

    formulaSus = {"ds_c":sus_ds_one,'ds_r':sus_ds_two,'ds_e':sus_ds_three}

    return formulaSus


# dstar
def cal_dstar_c(tf, tp, aef, aep, anf, anp, index,cc):
    a = aep - cc + (tf - aef)
    if a == 0:
        if aef>0:
            return sys.maxsize
        else:
            return 0, 0
    b = math.pow(aef, index)
    c = b / a
    return c


# dstar
def cal_dstar_r(tf, tp, aef, aep, anf, anp, index,cc):
    a = aep + (tf - aef) - cc
    if a == 0:
        if aef>0:
            return sys.maxsize
        else:
            return 0, 0
    b = math.pow(aef+cc, index)
    c = b / a
    return c

# dstar
def cal_dstar_e(tf, tp, aef, aep, anf, anp, index,cc,cc_sum):
    a = aep + (tf - aef) +cc_sum -2 * cc
    if a == 0:
        if aef>0:
            return sys.maxsize
        else:
            return 0, 0
    b = math.pow(aef+cc, index)
    c = b / a
    return c


if __name__ =="__main__":
    originPath = "/home/wuyonghao/Defeats4JFile/outputClean"
    root = '/home/wuyonghao/CCIdentifyFile/base_dataset'
    data = '/home/wuyonghao/CCIdentifyFile/base_pydata'
    error_pro_ver = '/home/wuyonghao/CCIdentifyFile/base_error'
    res_path = '/home/wuyonghao/CCIdentifyFile/base_res'

    base_info(root, data, error_pro_ver, res_path,originPath)