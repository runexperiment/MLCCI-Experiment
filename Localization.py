import copy
import math
import os

# 读取一个文件夹，返回里面所有的文件名
import pickle
import sys

import Tool_io
from tqdm import trange
from multiprocessing import Pool


# 获取文件工具类
def get_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        return files
    return []


# 处理cc计算结果
def deal_cc():
    print("s")


# 计算预测值
def cal_predict(root,data,model_path,p_name,res_path):

    files_origin = get_files(model_path)
    result = []
    for str_file in files_origin:
        if str_file.find(".res") != -1:
            result.append(str_file)

    # # 加载模型
    # f = open(os.path.join(model_path, p_name+'.mod'), 'rb')
    # model = pickle.load(f)
    # # 加载要预测的数据
    # f = open(os.path.join(model_path, p_name + '.td'), 'rb')
    # td = pickle.load(f)
    # 程序路径
    pn_pydata = os.path.join(data,p_name)
    pp_name = copy.deepcopy(p_name)

    pool = Pool(processes=8)
    for ver_name in result:
        tmp = ver_name.split('.')[0]
        ver_n = tmp.replace(p_name, '')
        param = []
        param.append(pn_pydata)
        param.append(ver_n)
        param.append(pp_name)
        param.append(model_path)
        param.append(ver_name)
        # process_start(param)
        pool.apply_async(process_start, (param,))
    pool.close()
    pool.join()

    # for ver in td:
    #     vn_pydata = os.path.join(pn_pydata,ver)
    #     res = Tool_io.checkAndLoad(vn_pydata, "data_Coverage_InVector_saveAgain")
    #     # 失败测试用例1
    #     fault_index = res[7]
    #     X_test = td[ver]['res_array']
    #     Y_test = model.predict(X_test)
    #     # 将失败测试用例结果改为2，cc为1
    #     for val in fault_index:
    #         Y_test[val] = 2
    #     # 获取真实错误位置
    #     # fault = []
    #     # vn_dataset = os.path.join(pn_dataset,ver)
    #     # fault_loc = Tool_io.checkAndLoad(vn_dataset, "faultHuge.in")
    #     # for jfile in fault_loc:
    #     #     for index in fault_loc[jfile]:sss
    #     #         fault.append(index)
    #     # 计算怀疑度
    #     sus_path = os.path.join(os.path.join(data,p_name),ver)
    #
    #     cal_sus(len(res[1]), len(res[1][0]), res[1], Y_test, sus_path)
    #     print("s")
    # print("s")


def process_start(param):

    pn_pydata = param[0]
    ver = param[1]
    p_name = param[2]
    model_path = param[3]
    ver_name = param[4]
    Y_test = Tool_io.checkAndLoad(model_path, ver_name)

    vn_pydata = os.path.join(pn_pydata,ver)
    res = Tool_io.checkAndLoad(vn_pydata, "data_Coverage_InVector_saveAgain")
    # 失败测试用例1
    fault_index = res[7]
    # 将失败测试用例结果改为2，cc为1
    for val in fault_index:
        Y_test[val] = 2
    # 计算怀疑度
    sus_path = os.path.join(os.path.join(data,p_name),ver)

    cal_sus(len(res[1]), len(res[1][0]), res[1], Y_test, sus_path)


# 获取要预测的程序信息
def get_info(root,data,error_pro_ver,res_path,model_path):
    # files = get_files(model_path)
    # pro = []
    # for file in files:
    #     name = file.split(".")[0]
    #     pro.append(name)
    # program = list(set(pro))
    program = ['Chart', 'Time', 'Mockito', 'Lang', 'Math', 'Closure']
    for p_name in program:
        #if p_name == 'Chart' or p_name == 'Time' or p_name == 'Closure' or p_name == 'Lang' or p_name == 'Math' or p_name == 'Mockito':
        if p_name == 'Closure':
            print(p_name)
            m_path = os.path.join(model_path, p_name)
            cal_predict(root, data, m_path, p_name, res_path)
    print(program)


# 计算怀疑度
def cal_sus(case_num,statement_num,covMatrix,inVector,sus_path):

    sus_value = Tool_io.checkAndLoad(sus_path, "sus_value3")
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
    Tool_io.checkAndSave(sus_path, "sus_value3", formulaSus)
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
    sus_ja_zero = {}
    sus_ja_one = {}
    sus_ja_two = {}
    sus_ja_three = {}
    sus_gp_zero = {}
    sus_gp_one = {}
    sus_gp_two = {}
    sus_gp_three = {}
    sus_op_zero = {}
    sus_op_one = {}
    sus_op_two = {}
    sus_op_three = {}
    sus_tu_zero = {}
    sus_tu_one = {}
    sus_tu_two = {}
    sus_tu_three = {}
    sus_ru_zero = {}
    sus_ru_one = {}
    sus_ru_two = {}
    sus_ru_three = {}
    sus_cal_crosstab_zero = {}
    sus_cal_crosstab_one = {}
    sus_cal_crosstab_two = {}
    sus_cal_crosstab_three = {}
    sus_na_zero = {}
    sus_na_one = {}
    sus_na_two = {}
    sus_na_three = {}
    sus_bin_zero = {}
    sus_bin_one = {}
    sus_bin_two = {}
    sus_bin_three = {}

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

            ja_zero = cal_jaccard(tf, tp, aef, aep, anf, anp)
            sus_ja_zero[statement_index] = ja_zero

            gp_zero = cal_gp13(tf, tp, aef, aep, anf, anp)
            sus_gp_zero[statement_index] = gp_zero

            op_zero = cal_op2(tf, tp, aef, aep, anf, anp)
            sus_op_zero[statement_index] = op_zero

            tu_zero = cal_turantula(tf, tp, aef, aep, anf, anp)
            sus_tu_zero[statement_index] = tu_zero

            ru_zero = cal_russell(tf, tp, aef, aep, anf, anp)
            sus_ru_zero[statement_index] = ru_zero

            cal_crosstab_zero = cal_crosstab(tf, tp, aef, aep, anf, anp)
            sus_cal_crosstab_zero[statement_index] = cal_crosstab_zero

            na_zero = cal_naish1(tf, tp, aef, aep, anf, anp)
            sus_na_zero[statement_index] = na_zero

            bin_zero = cal_binary(tf, tp, aef, aep, anf, anp)
            sus_bin_zero[statement_index] = bin_zero


            # 清洗策略
            aepp = aep - ce

            oc_one = cal_ochiai(tf, tp, aef, aepp, anf, anp)
            sus_oc_one[statement_index] = oc_one

            ds_one = cal_dstar(tf, tp, aef, aepp, anf, anp, 3)
            sus_ds_one[statement_index] = ds_one

            ja_one = cal_jaccard(tf, tp, aef, aepp, anf, anp)
            sus_ja_one[statement_index] = ja_one

            gp_one = cal_gp13(tf, tp, aef, aepp, anf, anp)
            sus_gp_one[statement_index] = gp_one

            op_one = cal_op2(tf, tp, aef, aepp, anf, anp)
            sus_op_one[statement_index] = op_one

            tu_one = cal_turantula(tf, tp, aef, aepp, anf, anp)
            sus_tu_one[statement_index] = tu_one

            ru_one = cal_russell(tf, tp, aef, aepp, anf, anp)
            sus_ru_one[statement_index] = ru_one

            cal_crosstab_one = cal_crosstab(tf, tp, aef, aepp, anf, anp)
            sus_cal_crosstab_one[statement_index] = cal_crosstab_one

            na_one = cal_naish1(tf, tp, aef, aepp, anf, anp)
            sus_na_one[statement_index] = na_one

            bin_one = cal_binary(tf, tp, aef, aepp, anf, anp)
            sus_bin_one[statement_index] = bin_one

            # 重标策略
            aepp2 = aep - ce
            aeff2 = aef + ce
            tf2 =  anf + aeff2

            oc_two = cal_ochiai(tf2, tp, aeff2, aepp2, anf, anp)
            sus_oc_two[statement_index] = oc_two

            ds_two = cal_dstar(tf2, tp, aeff2, aepp2, anf, anp, 3)
            sus_ds_two[statement_index] = ds_two

            ja_two = cal_jaccard(tf2, tp, aeff2, aepp2, anf, anp)
            sus_ja_two[statement_index] = ja_two

            gp_two = cal_gp13(tf2, tp, aeff2, aepp2, anf, anp)
            sus_gp_two[statement_index] = gp_two

            op_two = cal_op2(tf2, tp, aeff2, aepp2, anf, anp)
            sus_op_two[statement_index] = op_two

            tu_two = cal_turantula(tf2, tp, aeff2, aepp2, anf, anp)
            sus_tu_two[statement_index] = tu_two

            ru_two = cal_russell(tf2, tp, aeff2, aepp2, anf, anp)
            sus_ru_two[statement_index] = ru_two

            cal_crosstab_two = cal_crosstab(tf2, tp, aeff2, aepp2, anf, anp)
            sus_cal_crosstab_two[statement_index] = cal_crosstab_two

            na_two = cal_naish1(tf2, tp, aeff2, aepp2, anf, anp)
            sus_na_two[statement_index] = na_two

            bin_two = cal_binary(tf2, tp, aeff2, aepp2, anf, anp)
            sus_bin_two[statement_index] = bin_two

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

            ja_three = cal_jaccard(tf3, tp, aeff3, aepp3, anff3, anpp3)
            sus_ja_three[statement_index] = ja_three

            gp_three = cal_gp13(tf3, tp, aeff3, aepp3, anff3, anpp3)
            sus_gp_three[statement_index] = gp_three

            op_three = cal_op2(tf3, tp, aeff3, aepp3, anff3, anpp3)
            sus_op_three[statement_index] = op_three

            tu_three = cal_turantula(tf3, tp, aeff3, aepp3, anff3, anpp3)
            sus_tu_three[statement_index] = tu_three

            ru_three = cal_russell(tf3, tp, aeff3, aepp3, anff3, anpp3)
            sus_ru_three[statement_index] = ru_three

            cal_crosstab_three = cal_crosstab(tf3, tp, aeff3, aepp3, anff3, anpp3)
            sus_cal_crosstab_three[statement_index] = cal_crosstab_three

            na_three = cal_naish1(tf3, tp, aeff3, aepp3, anff3, anpp3)
            sus_na_three[statement_index] = na_three

            bin_three = cal_binary(tf3, tp, aeff3, aepp3, anff3, anpp3)
            sus_bin_three[statement_index] = bin_three

    sus_oc_zero = sorted(sus_oc_zero.items(), key=lambda d: d[1], reverse=True)
    sus_oc_one = sorted(sus_oc_one.items(), key=lambda d: d[1], reverse=True)
    sus_oc_two = sorted(sus_oc_two.items(), key=lambda d: d[1], reverse=True)
    sus_oc_three = sorted(sus_oc_three.items(), key=lambda d: d[1], reverse=True)

    sus_ds_zero = sorted(sus_ds_zero.items(), key=lambda d: d[1], reverse=True)
    sus_ds_one = sorted(sus_ds_one.items(), key=lambda d: d[1], reverse=True)
    sus_ds_two = sorted(sus_ds_two.items(), key=lambda d: d[1], reverse=True)
    sus_ds_three = sorted(sus_ds_three.items(), key=lambda d: d[1], reverse=True)

    sus_ja_zero = sorted(sus_ja_zero.items(), key=lambda d: d[1], reverse=True)
    sus_ja_one = sorted(sus_ja_one.items(), key=lambda d: d[1], reverse=True)
    sus_ja_two = sorted(sus_ja_two.items(), key=lambda d: d[1], reverse=True)
    sus_ja_three = sorted(sus_ja_three.items(), key=lambda d: d[1], reverse=True)

    sus_gp_zero = sorted(sus_gp_zero.items(), key=lambda d: d[1], reverse=True)
    sus_gp_one = sorted(sus_gp_one.items(), key=lambda d: d[1], reverse=True)
    sus_gp_two = sorted(sus_gp_two.items(), key=lambda d: d[1], reverse=True)
    sus_gp_three = sorted(sus_gp_three.items(), key=lambda d: d[1], reverse=True)

    sus_op_zero = sorted(sus_op_zero.items(), key=lambda d: d[1], reverse=True)
    sus_op_one = sorted(sus_op_one.items(), key=lambda d: d[1], reverse=True)
    sus_op_two = sorted(sus_op_two.items(), key=lambda d: d[1], reverse=True)
    sus_op_three = sorted(sus_op_three.items(), key=lambda d: d[1], reverse=True)

    sus_tu_zero = sorted(sus_tu_zero.items(), key=lambda d: d[1], reverse=True)
    sus_tu_one = sorted(sus_tu_one.items(), key=lambda d: d[1], reverse=True)
    sus_tu_two = sorted(sus_tu_two.items(), key=lambda d: d[1], reverse=True)
    sus_tu_three = sorted(sus_tu_three.items(), key=lambda d: d[1], reverse=True)

    sus_ru_zero = sorted(sus_ru_zero.items(), key=lambda d: d[1], reverse=True)
    sus_ru_one = sorted(sus_ru_one.items(), key=lambda d: d[1], reverse=True)
    sus_ru_two = sorted(sus_ru_two.items(), key=lambda d: d[1], reverse=True)
    sus_ru_three = sorted(sus_ru_three.items(), key=lambda d: d[1], reverse=True)

    sus_cal_crosstab_zero = sorted(sus_cal_crosstab_zero.items(), key=lambda d: d[1], reverse=True)
    sus_cal_crosstab_one = sorted(sus_cal_crosstab_one.items(), key=lambda d: d[1], reverse=True)
    sus_cal_crosstab_two = sorted(sus_cal_crosstab_two.items(), key=lambda d: d[1], reverse=True)
    sus_cal_crosstab_three = sorted(sus_cal_crosstab_three.items(), key=lambda d: d[1], reverse=True)

    sus_na_zero = sorted(sus_na_zero.items(), key=lambda d: d[1], reverse=True)
    sus_na_one = sorted(sus_na_one.items(), key=lambda d: d[1], reverse=True)
    sus_na_two = sorted(sus_na_two.items(), key=lambda d: d[1], reverse=True)
    sus_na_three = sorted(sus_na_three.items(), key=lambda d: d[1], reverse=True)

    sus_bin_zero = sorted(sus_bin_zero.items(), key=lambda d: d[1], reverse=True)
    sus_bin_one = sorted(sus_bin_one.items(), key=lambda d: d[1], reverse=True)
    sus_bin_two = sorted(sus_bin_two.items(), key=lambda d: d[1], reverse=True)
    sus_bin_three = sorted(sus_bin_three.items(), key=lambda d: d[1], reverse=True)

    formulaSus = {'ochiai':sus_oc_zero,"ochiai_c": sus_oc_one,'ochiai_r':sus_oc_two,'ochiai_e':sus_oc_three,
                  'ds':sus_ds_zero,"ds_c":sus_ds_one,'ds_r':sus_ds_two,'ds_e':sus_ds_three,
                  'ja':sus_ja_zero,"ja_c":sus_ja_one,'ja_r':sus_ja_two,'ja_e':sus_ja_three,
                  'op':sus_op_zero,"op_c":sus_op_one,'op_r':sus_op_two,'op_e':sus_op_three,
                  'gp':sus_gp_zero,"gp_c":sus_gp_one,'gp_r':sus_gp_two,'gp_e':sus_gp_three,
                  'tu':sus_tu_zero,"tu_c":sus_tu_one,'tu_r':sus_tu_two,'tu_e':sus_tu_three,
                  'ru':sus_ru_zero,"ru_c":sus_ru_one,'ru_r':sus_ru_two,'ru_e':sus_ru_three,
                  'cross':sus_cal_crosstab_zero,"cross_c":sus_cal_crosstab_one,'cross_r':sus_cal_crosstab_two,'cross_e':sus_cal_crosstab_three,
                  'bin':sus_bin_zero,"bin_c":sus_bin_one,'bin_r':sus_bin_two,'bin_e':sus_bin_three,
                  'na':sus_na_zero,"na_c":sus_na_one,'na_r':sus_na_two,'na_e':sus_na_three}

    return formulaSus

def cal_naish1(tf, tp, aef, aep, anf, anp):
    if anf > 0:
        return -1
    else:
        return anp


def cal_binary(tf, tp, aef, aep, anf, anp):
    if anf > 0:
        return 0
    else:
        return 1


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


# turantula
def cal_turantula(tf, tp, aef, aep, anf, anp):
    if aef == 0:
        return 0
    if tf == 0 or tp == 0:
        return 0
    a = aef / tf
    b = aep / tp
    c = a / (a + b)
    return c


# russell
def cal_russell(tf, tp, aef, aep, anf, anp):
    a = aep + anp + aef + anf
    c = 0
    if a != 0:
        c = aef / a
    return c


def cal_crosstab(Nf, Ns, Ncfw, Ncsw, Nufw, Nusw):
    N = Nf + Ns
    Sw = 0
    Ncw = Ncfw + Ncsw
    Nuw = Nufw + Nusw
    try:
        Ecfw = Ncw * (Nf / N)
        Ecsw = Ncw * (Ns / N)
        Eufw = Nuw * (Nf / N)
        Eusw = Nuw * (Ns / N)
        X2w = pow(Ncfw - Ecfw, 2) / Ecfw + pow(Ncsw - Ecsw, 2) / Ecsw + pow(Nufw - Eufw, 2) / Eufw + pow(
            Nusw - Eusw, 2) / Eusw
        yw = (Ncfw / Nf) / (Ncsw / Ns)
        if yw > 1:
            Sw = X2w
        elif yw < 1:
            Sw = -X2w
    except:
        Sw = 0
    return Sw


if __name__ =="__main__":

    root = '/home/tianshuaihua/base_dataset'
    data = '/home/tianshuaihua/tpydata'
    error_pro_ver = '/home/tianshuaihua/error'
    res_path = '/home/tianshuaihua/res3'
    model_path = '/home/tianshuaihua/model3'

    get_info(root,data,error_pro_ver,res_path,model_path)




