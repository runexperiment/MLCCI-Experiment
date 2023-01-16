# coding=utf-8
from __future__ import division

import copy
import os
import sys
import math
import numpy as np
# anf, anp
import sys
import csv
from sklearn.neighbors import NearestNeighbors

from tqdm import trange

import Tool_io


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

def cal_dstar(tf, tp, aef, aep, anf, anp, index):
    a = aep + (tf - aef)
    if a == 0:
        if aef>0:
            return sys.maxsize, sys.maxsize
        else:
            return 0, 0
    b = math.pow(aef, index)
    c = b / a
    e = 1 / a
    return c, e

def cal_ochiai(tf, tp, aef, aep, anf, anp):
    a = aef + aep
    b = math.sqrt(tf * a)
    # 子公式值（分母）
    c = math.sqrt(tf)
    d = math.sqrt(a)

    # ochiai值
    e = 0
    # 子公式 1 值
    f = 0
    # 子公式 2 值
    g = 0

    if b != 0:
        e = aef / b
    if c != 0:
        f = 1 / c
    if d != 0:
        g = 1 / d

    return e, f, g

def cal_jaccard(tf, tp, aef, aep, anf, anp):
    a=tf+aep
    if a==0:
        return 0,0
    b=aef/a
    # 子公式值 分子为1
    c = 1/a
    return b,c

def cal_gp13(tf, tp, aef, aep, anf, anp):
    a = 2 * aep + aef
    b = 0
    c = 0
    d = 0
    if a != 0:
        # gp13子公式
        b = 1 / a
        c = aef / a
        # gp13公式
        d = aef + c
    return d, b, c


def cal_naish2(tf, tp, aef, aep, anf, anp):
    a = aep + anf + 1
    b = 0
    c = 0
    d = 0
    if a != 0:
        # naish2子公式
        b = 1 / a
        c = aep / a
        # naish2公式
        d = aef - c
    return d, b, c

def cal_russell(tf, tp, aef, aep, anf, anp):
    a = aep + anp + aef + anf
    b = 0
    c = 0
    if a != 0:
        # russell子公式
        b = 1 / a
        # russell公式
        c = aef / a
    return c, b


def cal_binary(tf, tp, aef, aep, anf, anp):
    if anf > 0:
        return 0
    else:
        return 1


def cal_naish1(tf, tp, aef, aep, anf, anp):
    if anf > 0:
        return -1
    else:
        return anp


def cal_turantula(tf, tp, aef, aep, anf, anp):
    if aef == 0:
        return 0, 0
    if tf == 0 or tp == 0:
        return 0, 0
    a = aef / tf
    b = aep / tp
    c = a / (a + b)
    d = 1 / (a + b)
    return c, d

def cal_op2(tf, tp, aef, aep, anf, anp):
    a = aep / (tp + 1)
    b = aef - a
    c = 1 / (tp + 1)
    return b, a, c


# anf, anp
def cal_crosstab_cc(tf, tp, fail_s, pass_s, cc_pro_sum, cc_s):
    a = cal_crosstab(tf, tp - cc_s, fail_s, pass_s - cc_s, tf - fail_s, tp - pass_s)
    b = cal_crosstab(tf + cc_s, tp - cc_s, fail_s + cc_s, pass_s - cc_s, tf - fail_s, tp - pass_s)
    c = cal_crosstab(tf + cc_pro_sum, tp - cc_pro_sum, fail_s + cc_s, pass_s - cc_s, tf - fail_s, tp - pass_s)
    return a, b, c

def cal_ochiai_cc(tf, tp, fail_s, pass_s, cc_pro_sum, cc_s):
    a = cal_ochiai(tf, tp - cc_s, fail_s, pass_s - cc_s, None, None)
    b = cal_ochiai(tf + cc_s, tp - cc_s, fail_s +
                   cc_s, pass_s - cc_s, None, None)
    c = cal_ochiai(tf + cc_pro_sum, tp - cc_pro_sum,
                   fail_s + cc_s, pass_s - cc_s, None, None)
    return a, b, c

def cal_jaccard_cc(tf, tp, fail_s, pass_s, cc_pro_sum, cc_s):
    a = cal_jaccard(tf, tp - cc_s, fail_s, pass_s - cc_s, None, None)
    b = cal_jaccard(tf + cc_s, tp - cc_s, fail_s +
                       cc_s, pass_s - cc_s, None, None)
    c = cal_jaccard(tf + cc_pro_sum, tp - cc_pro_sum,
                       fail_s + cc_s, pass_s - cc_s, None, None)
    return a, b, c

def cal_dstar_cc(tf, tp, fail_s, pass_s, cc_pro_sum, cc_s):
    a = cal_dstar(tf, tp - cc_s, fail_s, pass_s - cc_s, None, None, 3)
    b = cal_dstar(tf + cc_s, tp - cc_s, fail_s +
                  cc_s, pass_s - cc_s, None, None, 3)
    c = cal_dstar(tf + cc_pro_sum, tp - cc_pro_sum, fail_s +
                  cc_s, pass_s - cc_s, None, None, 3)
    return a, b, c

def cal_turantula_cc(tf, tp, fail_s, pass_s, cc_pro_sum, cc_s):
    a = cal_turantula(tf, tp - cc_s, fail_s, pass_s - cc_s, None, None)
    b = cal_turantula(tf + cc_s, tp - cc_s, fail_s +
                      cc_s, pass_s - cc_s, None, None)
    c = cal_turantula(tf + cc_pro_sum, tp - cc_pro_sum,
                      fail_s + cc_s, pass_s - cc_s, None, None)
    return a, b, c

def cal_op2_cc(tf, tp, fail_s, pass_s, cc_pro_sum, cc_s):
    a = cal_op2(tf, tp - cc_s, fail_s, pass_s - cc_s, None, None)
    b = cal_op2(tf + cc_s, tp - cc_s, fail_s + cc_s, pass_s - cc_s, None, None)
    c = cal_op2(tf + cc_pro_sum, tp - cc_pro_sum,
                fail_s + cc_s, pass_s - cc_s, None, None)
    return a, b, c


def SBFL_location(covMatrix_int,in_vector,use_list=None):
    statement_num = len(covMatrix_int[0])
    case_num = len(covMatrix_int)

    sus_oc = {}
    sus_tu = {}
    sus_op = {}
    sus_ds = {}
    sus_cr = {}
    sus_ja = {}
    if len(covMatrix_int) == 0:
        for i in range(statement_num):
            sus_oc[i] = 0
            sus_tu[i] = 0
            sus_op[i] = 0
            sus_ds[i] = 0
            sus_cr[i] = 0
            sus_ja[i] = 0
        return sus_oc, sus_tu, sus_op, sus_ds, sus_cr, sus_ja

    tf = 0
    tp = 0
    for case_index in range(case_num):
        if use_list == None or len(use_list) == 0 or case_index in use_list:
            if in_vector[case_index] == 1:
                tf += 1
            else:
                tp += 1

    for statement_index in trange(statement_num):
        initial_int = covMatrix_int[0][statement_index]
        if initial_int == 2:
            sus_oc[statement_index] = -1
            sus_tu[statement_index] = -1
            sus_op[statement_index] = -1
            sus_ds[statement_index] = -1
            sus_cr[statement_index] = -1
            sus_ja[statement_index] = -1
        else:
            aef = 0
            aep = 0
            anf = 0
            anp = 0
            for case_index in range(case_num):
                if use_list == None or len(use_list) == 0 or case_index in use_list:
                    current_int = covMatrix_int[case_index][statement_index]
                    if current_int == 1:
                        if in_vector[case_index] == 1:
                            aef += 1
                        else:
                            aep += 1
                    else:
                        if in_vector[case_index] == 1:
                            anf += 1
                        else:
                            anp += 1

            tempp = cal_ochiai(tf, tp, aef, aep, anf, anp)
            sus_oc[statement_index] = tempp
            tempp = cal_turantula(tf, tp, aef, aep, anf, anp)
            sus_tu[statement_index] = tempp
            tempp = cal_op2(tf, tp, aef, aep, anf, anp)
            sus_op[statement_index] = tempp
            tempp = cal_dstar(tf, tp, aef, aep, anf, anp, 3)
            sus_ds[statement_index] = tempp
            tempp = cal_crosstab(tf, tp, aef, aep, anf, anp)
            sus_cr[statement_index] = tempp
            tempp = cal_jaccard(tf, tp, aef, aep, anf, anp)
            sus_ja[statement_index] = tempp
    return sus_oc, sus_tu, sus_op, sus_ds, sus_cr, sus_ja


# def SBFL_location_single_test_case(aefList,aepList,anfList,anpList,covMatrix,case_index,tf,tp):
#     statement_num = len(covMatrix[0])
#
#     sus_oc = {}
#     sus_tu = {}
#     sus_op = {}
#     sus_ds = {}
#     sus_cr = {}
#     sus_ja = {}
#
#     for statement_index in range(statement_num):
#             aef = aefList[statement_index]
#             aep = aepList[statement_index]
#             anf = anfList[statement_index]
#             anp = anpList[statement_index]
#             if covMatrix[case_index][statement_index] == 1:
#                 aep += 1
#             else:
#                 anp += 1
#
#             tempp = cal_ochiai(tf, tp, aef, aep, anf, anp)
#             sus_oc[statement_index] = tempp
#
#             tempp = cal_turantula(tf, tp, aef, aep, anf, anp)
#             sus_tu[statement_index] = tempp
#
#             tempp = cal_op2(tf, tp, aef, aep, anf, anp)
#             sus_op[statement_index] = tempp
#
#             tempp = cal_dstar(tf, tp, aef, aep, anf, anp, 3)
#             sus_ds[statement_index] = tempp
#
#             tempp = cal_crosstab(tf, tp, aef, aep, anf, anp)
#             sus_cr[statement_index] = tempp
#
#             tempp = cal_jaccard(tf, tp, aef, aep, anf, anp)
#             sus_ja[statement_index] = tempp
#
#     sus_ds = sorted(sus_ds.items(), key=lambda d: d[1], reverse=True)
#     sus_oc = sorted(sus_oc.items(), key=lambda d: d[1], reverse=True)
#     sus_tu = sorted(sus_tu.items(), key=lambda d: d[1], reverse=True)
#     sus_op = sorted(sus_op.items(), key=lambda d: d[1], reverse=True)
#     sus_cr = sorted(sus_cr.items(), key=lambda d: d[1], reverse=True)
#     sus_ja = sorted(sus_ja.items(), key=lambda d: d[1], reverse=True)
#     # formulaSus = {"dstar": sus_ds, "ochiai": sus_oc, "turantula": sus_tu, "op2": sus_op, "crosstab": sus_cr,"jaccard": sus_ja}
#     formulaSus = {"dstar": sus_ds, "ochiai": sus_oc,"jaccard": sus_ja}
#     return formulaSus

def SBFL_location_all_test_case(aefList,aepList,anfList,anpList,covMatrix,tf,tp):
    statement_num = len(covMatrix[0])

    sus_oc = {}
    sus_oc_subOne = {}
    sus_oc_subTwo = {}
    sus_tu = {}
    sus_tu_subOne = {}
    sus_op = {}
    sus_op_subOne = {}
    sus_op_subTwo = {}
    sus_ds = {}
    sus_ds_subOne = {}
    sus_cr = {}
    sus_ja = {}
    sus_ja_subOne= {}
    sus_gp = {}
    sus_gp_subOne = {}
    sus_gp_subTwo = {}
    # sus_na2 = {}
    # sus_na2_subOne = {}
    # sus_na2_subTwo = {}
    sus_rar = {}
    sus_rar_subOne = {}
    sus_bin = {}
    sus_na1 = {}

    for statement_index in range(statement_num):
            aef = aefList[statement_index]
            aep = aepList[statement_index]
            anf = anfList[statement_index]
            anp = anpList[statement_index]

            tempp,oc_one,oc_two = cal_ochiai(tf, tp, aef, aep, anf, anp)
            sus_oc[statement_index] = tempp
            sus_oc_subOne[statement_index] = oc_one
            sus_oc_subTwo[statement_index] = oc_two

            tempp,turantula_one = cal_turantula(tf, tp, aef, aep, anf, anp)
            sus_tu[statement_index] = tempp
            sus_tu_subOne[statement_index] = turantula_one

            tempp = cal_binary(tf, tp, aef, aep, anf, anp)
            sus_bin[statement_index] = tempp

            tempp = cal_naish1(tf, tp, aef, aep, anf, anp)
            sus_na1[statement_index] = tempp

            tempp,rar_one = cal_russell(tf, tp, aef, aep, anf, anp)
            sus_rar[statement_index] = tempp
            sus_rar_subOne[statement_index] = rar_one

            # tempp,na2_one,na2_two = cal_naish2(tf, tp, aef, aep, anf, anp)
            # sus_na2[statement_index] = tempp
            # sus_na2_subOne[statement_index] = na2_one
            # sus_na2_subTwo[statement_index] = na2_two

            tempp,gp_one,gp_two = cal_gp13(tf, tp, aef, aep, anf, anp)
            sus_gp[statement_index] = tempp
            sus_gp_subOne[statement_index] = gp_one
            sus_gp_subTwo[statement_index] = gp_two

            tempp,op_one,op_two = cal_op2(tf, tp, aef, aep, anf, anp)
            sus_op[statement_index] = tempp
            sus_op_subOne[statement_index] = op_one
            sus_op_subTwo[statement_index] = op_two

            tempp, dstar_one = cal_dstar(tf, tp, aef, aep, anf, anp, 3)
            sus_ds[statement_index] = tempp
            sus_ds_subOne[statement_index] = dstar_one

            tempp = cal_crosstab(tf, tp, aef, aep, anf, anp)
            sus_cr[statement_index] = tempp

            tempp,ja_one = cal_jaccard(tf, tp, aef, aep, anf, anp)
            sus_ja[statement_index] = tempp
            # jaccard 子公式值 分母为1
            sus_ja_subOne[statement_index] = ja_one

    ds = copy.deepcopy(sus_ds)
    ds1 = copy.deepcopy(sus_ds_subOne)
    ochiai = copy.deepcopy(sus_oc)
    ochiai1 = copy.deepcopy(sus_oc_subOne)
    ochiai2 = copy.deepcopy(sus_oc_subTwo)
    tu = copy.deepcopy(sus_tu)
    tu1 = copy.deepcopy(sus_tu_subOne)
    op = copy.deepcopy(sus_op)
    op1 = copy.deepcopy(sus_op_subOne)
    op2 = copy.deepcopy(sus_op_subTwo)
    cr = copy.deepcopy(sus_cr)
    ja = copy.deepcopy(sus_ja)
    ja1 = copy.deepcopy(sus_ja_subOne)
    gp = copy.deepcopy(sus_gp)
    gp1 = copy.deepcopy(sus_gp_subOne)
    gp2 = copy.deepcopy(sus_gp_subTwo)
    # na2 = copy.deepcopy(sus_na2)
    # na21 = copy.deepcopy(sus_na2_subOne)
    # na22 = copy.deepcopy(sus_na2_subTwo)
    rar = copy.deepcopy(sus_rar)
    rar1 = copy.deepcopy(sus_rar_subOne)
    bin = copy.deepcopy(sus_bin)
    na1 = copy.deepcopy(sus_na1)

    sus_ds = sorted(sus_ds.items(), key=lambda d: d[1], reverse=True)
    sus_ds_subOne = sorted(sus_ds_subOne.items(), key=lambda d: d[1], reverse=True)
    sus_oc = sorted(sus_oc.items(), key=lambda d: d[1], reverse=True)
    sus_oc_subOne = sorted(sus_oc_subOne.items(), key=lambda d: d[1], reverse=True)
    sus_oc_subTwo = sorted(sus_oc_subTwo.items(), key=lambda d: d[1], reverse=True)
    sus_tu = sorted(sus_tu.items(), key=lambda d: d[1], reverse=True)
    sus_tu_subOne = sorted(sus_tu_subOne.items(), key=lambda d: d[1], reverse=True)
    sus_op = sorted(sus_op.items(), key=lambda d: d[1], reverse=True)
    sus_op_subOne = sorted(sus_op_subOne.items(), key=lambda d: d[1], reverse=True)
    sus_op_subTwo = sorted(sus_op_subTwo.items(), key=lambda d: d[1], reverse=True)
    sus_cr = sorted(sus_cr.items(), key=lambda d: d[1], reverse=True)
    sus_ja = sorted(sus_ja.items(), key=lambda d: d[1], reverse=True)
    sus_ja_subOne = sorted(sus_ja_subOne.items(), key=lambda d: d[1], reverse=True)
    sus_gp = sorted(sus_gp.items(), key=lambda d: d[1], reverse=True)
    sus_gp_subOne = sorted(sus_gp_subOne.items(), key=lambda d: d[1], reverse=True)
    sus_gp_subTwo = sorted(sus_gp_subTwo.items(), key=lambda d: d[1], reverse=True)
    # sus_na2 = sorted(sus_na2.items(), key=lambda d: d[1], reverse=True)
    # sus_na2_subOne = sorted(sus_na2_subOne.items(), key=lambda d: d[1], reverse=True)
    # sus_na2_subTwo = sorted(sus_na2_subTwo.items(), key=lambda d: d[1], reverse=True)
    sus_rar = sorted(sus_rar.items(), key=lambda d: d[1], reverse=True)
    sus_rar_subOne = sorted(sus_rar_subOne.items(), key=lambda d: d[1], reverse=True)
    sus_bin = sorted(sus_bin.items(), key=lambda d: d[1], reverse=True)
    sus_na1 = sorted(sus_na1.items(), key=lambda d: d[1], reverse=True)

    # formulaSus = {"dstar": sus_ds, "ochiai": sus_oc, "turantula": sus_tu, "op2": sus_op, "crosstab": sus_cr,"jaccard": sus_ja}

    formulaSus = {"dstar": sus_ds, 'dstar_sub_one':sus_ds_subOne, "turantula": sus_tu, 'turantula_sub_one':sus_tu_subOne,"op2": sus_op,'op21':sus_op_subOne,'op22':sus_op_subTwo, "crosstab": sus_cr,
                  "ochiai": sus_oc,"ochiai_sub_one":sus_oc_subOne,"ochiai_sub_two":sus_oc_subTwo,
                  "jaccard": sus_ja,"jaccard_sub_one":sus_ja_subOne,
                  "gp13": sus_gp,"gp13_sub_one": sus_gp_subOne,"gp13_sub_two": sus_gp_subTwo,
                  "russell": sus_rar,"russell_sub_one":sus_rar_subOne,
                  "naish1": sus_na1, "binary": sus_bin}
    notSort = {
        "dstar": ds, 'dstar1': ds1, "turantula": tu,'turantula1':tu1, "op2": op,'op21':op1,'op22':op2, "crosstab": cr,
        "ochiai": ochiai, "ochiai_sub_one": ochiai1, "ochiai_sub_two": ochiai2,
        "jaccard": ja, "jaccard_sub_one": ja1,
        "gp13": gp, "gp13_sub_one": gp1, "gp13_sub_two": gp2,
        "russell": rar, "russell_sub_one": rar1,
        "naish1": na1, "binary": bin
    }
    return formulaSus,notSort


def SBFL_location_CC(covMatrix_int, in_vector,cc_pro,ccSum,StatementNum,use_list=None):
    sus_oc = {}
    sus_tu = {}
    sus_op = {}
    sus_ds = {}
    sus_cr = {}
    sus_ja = {}

    if len(covMatrix_int) == 0:
        for i in range(StatementNum):
            sus_oc[i] = 0
            sus_tu[i] = 0
            sus_op[i] = 0
            sus_ds[i] = 0
            sus_cr[i] = 0
            sus_ja[i] = 0
        return sus_oc, sus_tu, sus_op, sus_ds, sus_cr, sus_ja
    statement_num = len(covMatrix_int[0])
    case_num = len(covMatrix_int)



    tf = 0
    tp = 0
    for case_index in range(case_num):
        if use_list == None or len(use_list) == 0 or case_index in use_list:
            if in_vector[case_index] == 1:
                tf += 1
            else:
                tp += 1

    for statement_index in range(statement_num):
        initial_int = covMatrix_int[0][statement_index]
        if initial_int == 2:
            sus_oc[statement_index] = -1
            sus_tu[statement_index] = -1
            sus_op[statement_index] = -1
            sus_ds[statement_index] = -1
            sus_cr[statement_index] = -1
            sus_ja[statement_index] = -1
        else:
            aef = 0
            aep = 0
            anf = 0
            anp = 0

            cc_es = 0
            cc_ns = 0
            for case_index in range(case_num):
                if use_list == None or len(use_list) == 0 or case_index in use_list:
                    current_int = covMatrix_int[case_index][statement_index]
                    if current_int == 1:
                        if in_vector[case_index] == 1:
                            aef += 1
                        else:
                            aep += 1
                            cc_es += cc_pro[case_index]
                    else:
                        if in_vector[case_index] == 1:
                            anf += 1
                        else:
                            anp += 1
                            cc_ns+=cc_pro[case_index]

            tempp = cal_ochiai_cc(tf, tp, aef, aep,ccSum,cc_es)
            sus_oc[statement_index] = tempp
            tempp = cal_turantula_cc(tf, tp, aef, aep,ccSum,cc_es)
            sus_tu[statement_index] = tempp
            tempp = cal_op2_cc(tf, tp, aef, aep,ccSum,cc_es)
            sus_op[statement_index] = tempp
            tempp = cal_dstar_cc(tf, tp, aef, aep, ccSum,cc_es)
            sus_ds[statement_index] = tempp
            tempp = cal_crosstab_cc(tf, tp, aef, aep,ccSum,cc_es)
            sus_cr[statement_index] = tempp
            tempp = cal_jaccard_cc(tf, tp, aef, aep,ccSum,cc_es)
            sus_ja[statement_index] = tempp
    return sus_oc, sus_tu, sus_op, sus_ds, sus_cr, sus_ja


def statement_sus(formulas, case_num,statement_num,covMatrix,inVector,versionPath):
    newformulas = {}
    sus_score = Tool_io.checkAndLoad(versionPath[1], "sus_score")
    if sus_score != None:
        sus_score, newformulas, flag = Tool_io.add_del_formula(sus_score,formulas,versionPath[1],'sus_score')
        if flag == 0:
            return sus_score

    #失败的测试用例
    tf = 0
    #成功的测试用例
    tp = 0
    for case_index in range(case_num):
        if inVector[case_index] == 1:
            tf += 1
        else:
            tp += 1
    #执行语句e且通过的测试用例数量
    aef = {}
    #执行语句e且未通过该的测试用例数量
    aep = {}
    #未执行语句e且未通过该的测试用例数量
    anf = {}
    #执行语句e且通过该的测试用例数量
    anp = {}
    for statement_index in trange(statement_num):
        aef[statement_index] = 0
        aep[statement_index] = 0
        anf[statement_index] = 0
        anp[statement_index] = 0
        for case_index in range(case_num):
            current_int = covMatrix[case_index][statement_index]
            if current_int == 1:
                if inVector[case_index] == 1:
                    aef[statement_index] += 1
                else:
                    aep[statement_index] += 1
            else:
                if inVector[case_index] == 1:
                    anf[statement_index] += 1
                else:
                    anp[statement_index] += 1
    #计算怀疑分数（ochiai）
    if sus_score==None:
        newformulas = formulas
    formulaSus = SBFL_location_Test_Cases(newformulas, len(covMatrix[0]), aef, aep, anf, anp, tf, tp)

    # 保存结果
    if sus_score is not None:
        for s_index in formulaSus:
            sus_score[s_index] = formulaSus[s_index]
    else:
        sus_score = formulaSus

    Tool_io.checkAndSave(versionPath[1], "sus_score", sus_score)

    return sus_score


# 从csv文件中读取怀疑度公式
def deal_suspicion_formula():
    path = os.path.join(os.path.join(os.getcwd(), 'config'), 'formula.csv')
    formulas = {}
    with open(path, 'r', newline='') as csvfile:
        content = csv.reader(csvfile)
        for row in content:
            # 不读取表头
            if row[0] == 'name':
                continue
            formulas[row[0]] = row[1]
    return formulas


def cal_formula_suspicion(name, formula, aef, aep, anf, anp, tf, tp, index):
    # 计算怀疑度值
    try:
        if name == 'crosstab':
            res = cal_crosstab(tf, tp, aef, aep, anf, anp)
        else:
            res = eval(formula)
    except ZeroDivisionError:
        if name.__contains__('dstar') and aef > 0:
            return sys.maxsize
        else:
            return 0
    else:
        return res


# 计算所有测试用例怀疑度值
def SBFL_location_Test_Cases(formulas, fun_num, aefList, aepList, anfList, anpList, tf, tp):
    formula_data = {}
    # 为不同怀疑度公式创建空字典
    for name in formulas:
        formula_data[name] = {}
    index = 3
    # 遍历语句
    for fun_index in range(fun_num):
        aef = aefList[fun_index]
        aep = aepList[fun_index]
        anf = anfList[fun_index]
        anp = anpList[fun_index]
        for formula_index in formulas:
            sus_value = cal_formula_suspicion(formula_index, formulas[formula_index], aef, aep, anf, anp, tf, tp, index)
            formula_data[formula_index][fun_index] = sus_value

    return formula_data


# 归一化特征
def normalization(SS_or,CR_or,SF_or,FM_or,versionPath):

    SS =copy.deepcopy(SS_or)
    CR =copy.deepcopy(CR_or)
    SF =copy.deepcopy(SF_or)
    FM =copy.deepcopy(FM_or)

    SS.pop('dstar2')
    CR.pop('dstar2')
    SF.pop('dstar2')
    FM.pop('dstar2')

    res_nor = Tool_io.checkAndLoad(versionPath[1], "normalization_other")
    if res_nor != None:
        return res_nor

    all = [SS, CR, SF, FM]
    test_id = []
    for key in SS:
        for k in SS[key]:
            test_id.append(k)
        break

    lst = list()
    for feature in all:
        cov = []
        for key in feature:
            row = []
            for k in feature[key]:
                row.append(feature[key][k])
            cov.append(row)
        tmp = np.array(cov)
        lst.append(tmp)

    arrn = np.concatenate(lst, axis=0)
    nor_res = []
    for row in range(np.shape(arrn)[0]):
        row_val = arrn[row, :]
        fe = []
        for x in row_val:
            if row_val.std() != 0:
                x = float(x - row_val.mean()) / row_val.std()
                #x = float(x - np.min(col_val)) / (np.max(col_val) - np.min(col_val))
                fe.append(x)
            else:
                fe.append(0)
        nor_res.append(fe)
    normalization = np.array(nor_res).T
    # 保留计算结果
    res_nor = {}
    res_nor['normal'] = normalization
    Tool_io.checkAndSave(versionPath[1], "normalization_other", res_nor)
    return res_nor


def normal_info(array):
    fe = []
    arr = np.array(array)
    for x in arr:
        if arr.std() != 0:
            x = float(x - arr.mean()) / arr.std()
            fe.append(x)
        else:
            fe.append(0)
    return fe



# knn算法
def knn(X, realCC, failIndex, failN, trueCC):
    # 找出cc测试用例编号
    k_number = len(trueCC)+failN
    # cc = []
    # for x in realCC:
    #     cc.append(x)
    cc_dict = {}
    nbrs = NearestNeighbors(n_neighbors=k_number+1, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    for row_index in range(len(indices)):
        if row_index not in failIndex:
            for col_index in indices[row_index]:
                n = 0
                if col_index in failIndex:
                    n = n+1
            cc_dict[row_index] = n/k_number

    cc_dict = sorted(cc_dict.items(), key=lambda d: d[1], reverse=True)
    #print("s")

# knn算法
def fuzzy_knn(X, failIndex, failN):
    dict = X['dstar']
    cov = []
    for k,v in dict.items():
        cov.append(v)
    cov = np.array(cov)
    # 找出cc测试用例编号
    k_number = failN
    # cc = []
    # for x in realCC:
    #     cc.append(x)
    cc_dict = {}
    nbrs = NearestNeighbors(n_neighbors=k_number+1, algorithm='ball_tree').fit(cov)
    distances, indices = nbrs.kneighbors(cov)
    for row_index in range(len(indices)):
        if row_index not in failIndex:
            for col_index in indices[row_index]:
                n = 0
                if col_index in failIndex:
                    n = n+1
            cc_dict[row_index] = n/k_number

    cc_dict = sorted(cc_dict.items(), key=lambda d: d[1], reverse=True)
    #print("s")


if __name__ == "__main__":

    s = [3,4,5,3,6]
    res = normal_info(s)
    print(res)
    #a=cal_turantula(256,104,4,3,0,0)
    #b=cal_turantula(256,103,4,2,0,0)
    #print(a)
    #print(b)
    #normalization()
    # S = {
    #     'DS':{
    #         '0':1.1,
    #         '1':2.3,
    #         '3':3.8,
    #     },
    #     'OC':{
    #         '0':1.6,
    #         '1':2.8,
    #         '3': 4.8,
    #     }
    # }
    #
    # S2 = {
    #     'DS': {
    #         '0': 1.9,
    #         '1': 2.3,
    #         '3': 3.8,
    #     },
    #     'OC': {
    #         '0': 3.5,
    #         '1': 2.8,
    #         '3': 4.8,
    #     }
    # }
    # all = [S,S2]
    # test_id = []
    # for key,val in S.items():
    #     for k,v in val.items():
    #         test_id.append(k)
    #     break
    #
    # lst = list()
    # for feature in all:
    #     cov = []
    #     for key, val in feature.items():
    #         row = []
    #         for k, v in val.items():
    #             row.append(v)
    #         cov.append(row)
    #     tmp = np.array(cov)
    #     lst.append(tmp)
    #
    # arrn = np.concatenate(lst, axis=0)
    #print("s")