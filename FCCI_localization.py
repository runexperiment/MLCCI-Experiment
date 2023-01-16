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


# 计算预测值
def cal_predict(root,data,model_path,p_name,res_path,ver_data):
    # 加载模型
    # f = open(os.path.join(model_path, p_name+'.mod'), 'rb')
    # model = pickle.load(f)
    # 加载要预测的数据
    # f = open(os.path.join(model_path, p_name + '.td'), 'rb')
    # td = pickle.load(f)
    print(p_name)
    td = ver_data
    # 程序路径
    pn_pydata = os.path.join(data,p_name)
    for ver in td:
        print(ver)
        vn_pydata = os.path.join(pn_pydata,ver)
        res = Tool_io.checkAndLoad(vn_pydata, "data_Coverage_InVector_saveAgain")
        # 失败测试用例1
        fault_index = res[7]
        Y_test = Tool_io.checkAndLoad(vn_pydata, "fcci_predict")
        # 将失败测试用例结果改为2，cc为1
        for val in fault_index:
            Y_test[val] = 2
        # 计算怀疑度
        sus_path = os.path.join(os.path.join(data, p_name), ver)

        cal_sus(len(res[1]), len(res[1][0]), res[1], Y_test, sus_path)

    print("s")


# 获取要预测的程序信息
def get_info(root,data,error_pro_ver,res_path,model_path,ver_data):
    files = get_files(model_path)
    pro = []
    for file in files:
        name = file.split(".")[0]
        pro.append(name)
    program = list(set(pro))
    for p_name in program:
        if p_name == 'Math' or p_name == 'Chart' or p_name == 'Time' or p_name == 'Lang' or p_name == 'Closure' or p_name == 'Mockito':
            cal_predict(root,data,model_path,p_name,res_path,ver_data[p_name])
    print(program)


# 计算怀疑度
def cal_sus(case_num,statement_num,covMatrix,inVector,sus_path):

    sus_value = Tool_io.checkAndLoad(sus_path, "fcci_sus_value")
    if sus_value != None:
       return sus_value

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
    Tool_io.checkAndSave(sus_path, "fcci_sus_value", formulaSus)
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
    sus_op_zero = {}
    sus_op_one = {}
    sus_op_two = {}
    sus_op_three = {}

    sus_ds2_zero = {}
    sus_ds2_one = {}
    sus_ds2_two = {}
    sus_ds2_three = {}

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

            op2_zero = cal_op2(tf, tp, aef, aep, anf, anp)
            sus_op_zero[statement_index] = op2_zero

            ds2_zero = cal_dstar(tf, tp, aef, aep, anf, anp, 2)
            sus_ds2_zero[statement_index] = ds2_zero


            # 清洗策略
            aepp = aep - ce

            oc_one = cal_ochiai(tf, tp, aef, aepp, anf, anp)
            sus_oc_one[statement_index] = oc_one

            ds_one = cal_dstar(tf, tp, aef, aepp, anf, anp, 3)
            sus_ds_one[statement_index] = ds_one

            op_one = cal_op2(tf, tp, aef, aepp, anf, anp)
            sus_op_one[statement_index] = op_one

            ds2_one = cal_dstar(tf, tp, aef, aepp, anf, anp, 2)
            sus_ds2_one[statement_index] = ds2_one


            # 重标策略
            aepp2 = aep - ce
            aeff2 = aef + ce
            tf2 =  anf + aeff2

            oc_two = cal_ochiai(tf2, tp, aeff2, aepp2, anf, anp)
            sus_oc_two[statement_index] = oc_two

            ds_two = cal_dstar(tf2, tp, aeff2, aepp2, anf, anp, 3)
            sus_ds_two[statement_index] = ds_two

            op_two = cal_op2(tf2, tp, aeff2, aepp2, anf, anp)
            sus_op_two[statement_index] = op_two

            ds2_two = cal_dstar(tf2, tp, aeff2, aepp2, anf, anp, 2)
            sus_ds2_two[statement_index] = ds2_two

            # 交换策略
            aepp3 = aep - ce
            anpp3 = anp - cn
            aeff3 = aef + ce
            anff3 = anf + cn

            tf3 = aeff3 + anff3

            oc_three = cal_ochiai(tf3, tp, aeff3, aepp3, anff3, anpp3)
            sus_oc_three[statement_index] = oc_three

            ds_three = cal_dstar(tf3, tp, aeff3, aepp3, anff3, anpp3, 3)
            sus_ds_three[statement_index] = ds_three

            op_three = cal_op2(tf3, tp, aeff3, aepp3, anff3, anpp3)
            sus_op_three[statement_index] = op_three

            ds2_three = cal_dstar(tf3, tp, aeff3, aepp3, anff3, anpp3, 2)
            sus_ds2_three[statement_index] = ds2_three


    sus_oc_zero = sorted(sus_oc_zero.items(), key=lambda d: d[1], reverse=True)
    sus_oc_one = sorted(sus_oc_one.items(), key=lambda d: d[1], reverse=True)
    sus_oc_two = sorted(sus_oc_two.items(), key=lambda d: d[1], reverse=True)
    sus_oc_three = sorted(sus_oc_three.items(), key=lambda d: d[1], reverse=True)
    sus_ds_zero = sorted(sus_ds_zero.items(), key=lambda d: d[1], reverse=True)
    sus_ds_one = sorted(sus_ds_one.items(), key=lambda d: d[1], reverse=True)
    sus_ds_two = sorted(sus_ds_two.items(), key=lambda d: d[1], reverse=True)
    sus_ds_three = sorted(sus_ds_three.items(), key=lambda d: d[1], reverse=True)

    sus_op_zero = sorted(sus_op_zero.items(), key=lambda d: d[1], reverse=True)
    sus_op_one = sorted(sus_op_one.items(), key=lambda d: d[1], reverse=True)
    sus_op_two = sorted(sus_op_two.items(), key=lambda d: d[1], reverse=True)
    sus_op_three = sorted(sus_op_three.items(), key=lambda d: d[1], reverse=True)

    sus_ds2_zero = sorted(sus_ds2_zero.items(), key=lambda d: d[1], reverse=True)
    sus_ds2_one = sorted(sus_ds2_one.items(), key=lambda d: d[1], reverse=True)
    sus_ds2_two = sorted(sus_ds2_two.items(), key=lambda d: d[1], reverse=True)
    sus_ds2_three = sorted(sus_ds2_three.items(), key=lambda d: d[1], reverse=True)

    formulaSus = {'ochiai':sus_oc_zero,"ochiai_c": sus_oc_one,'ochiai_r':sus_oc_two,'ochiai_e':sus_oc_three,
                  'ds':sus_ds_zero,"ds_c":sus_ds_one,'ds_r':sus_ds_two,'ds_e':sus_ds_three,
                  'op':sus_op_zero,'op_c':sus_op_one,'op_r':sus_op_two,'op_e':sus_op_three,
                  'ds2':sus_ds2_zero,"ds2_c":sus_ds2_one,'ds2_r':sus_ds2_two,'ds2_e':sus_ds2_three,}

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

    root = '/home/tianshuaihua/base_dataset'
    data = '/home/tianshuaihua/tpydata'
    error_pro_ver = '/home/tianshuaihua/error'
    res_path = '/home/tianshuaihua/fcci/res'
    model_path = '/home/tianshuaihua/model'

    vers_path = '/home/tianshuaihua/wang/fea'
    ver_data = Tool_io.checkAndLoad(vers_path, 'vers.in')

    get_info(root,data,error_pro_ver,res_path,model_path,ver_data)

