# coding=utf-8
from __future__ import division
import copy
import hashlib
import multiprocessing
import os
import pickle
from functools import partial
import csv

import numpy as np
from tqdm import trange

# 获取所有java文件
import Tool_CC


def getJavaFile(sourcePath):
    FileList = []
    for root, d_names, f_names in os.walk(sourcePath):
        for fileName in f_names:
            if fileName.lower().endswith(".java"):
                originPath = os.path.join(root, fileName)
                filePath = originPath.replace(sourcePath, "")
                FileList.append(filePath)
    FileList.sort()
    return FileList


def md5sum(filename):
    if os.path.exists(filename):
        with open(filename, 'r', encoding="utf-8", errors="ignore") as f:
            strings = f.read()
            # print(strings)
            if "print" in filename:
                if "doesn't exists" in strings:
                    if "print_tokens" in strings:
                        # print(filename + "一定出错")
                        return "print tokens not exists"
                    elif "garbage/nothing" in strings:
                        # print(filename + "找到不存在的测试用例")
                        return "nothing"

        with open(filename, 'rb') as f:
            d = hashlib.md5()
            for buf in iter(partial(f.read, 128), b''):
                d.update(buf)
            f.close()
        return d.hexdigest()
    else:
        return None


def get_case_num(root_path):
    case_num = {}
    # 首先获得所有项目的测试用例数量
    project_files = os.listdir(root_path)
    for project_file in project_files:
        project_name = project_file
        project_path = root_path + "/" + project_name
        scriptPath = project_path + "/gettraces.sh"
        if not os.path.exists(scriptPath):
            continue
        Case_num = getTestNum(scriptPath)
        case_num[project_name] = Case_num
    case_num = sorted(case_num.items(), key=lambda d: d[1], reverse=False)
    return case_num


def getTestNum(scriptPath):
    f = open(scriptPath, 'r')
    script_lines = f.readlines()
    f.close()
    for i in range(0, script_lines.__len__())[::-1]:
        temp_line = script_lines[i]
        if "$target/" in temp_line and "cp" in temp_line:
            Case_num = 1 + int(temp_line.split('$target/')[1].split('.txt')[0])
            return Case_num
    return 0


def get_right_md5(project_path):
    scriptPath = project_path + "/gettraces.sh"
    Case_num = getTestNum(scriptPath)

    Out_Result = []
    for Rcov in range(Case_num):
        temp_Out_Path = outputPath = project_path + \
                                     "/outputs/v0" + "/t" + str(Rcov + 1)
        tempMD5 = md5sum(temp_Out_Path)
        Out_Result.append(tempMD5)
    return Out_Result


def getLines(origin_fault_source_path):
    fh = open(origin_fault_source_path)
    string = fh.readlines()
    fh.close()
    return string


def checkDele(root, param):
    dump_path = os.path.join(root, param)
    if os.path.exists(dump_path):
        os.remove(dump_path)


def checkAndSave(root, param, content, enforce=False):
    dump_path = os.path.join(root, param)
    if (not os.path.exists(dump_path)) or enforce:
        f = open(dump_path, 'wb')
        pickle.dump(content, f)
        f.close()


def checkAndLoad(root, param):
    dump_path = os.path.join(root, param)
    if os.path.exists(dump_path):
        f = open(dump_path, 'rb')
        content = pickle.load(f)
        f.close()
        return content
    return None

def create_duplicate_code(root,param,target,delestat):
    code_path = os.path.join(root[0], param)
    target_path = os.path.join(root[1], target)
    if os.path.exists(target_path):
        return None
    if os.path.exists(code_path):
        k=0
        with open(code_path, 'r',encoding='utf-8') as f:
            for statment in f.readlines():
                if delestat[k] == False:
                    with open(target_path, 'a',encoding='utf-8') as file:
                        file.write(statment)
                k += 1
    return None

def create_duplicate_static(root,param,target,delestat):
    code_path = os.path.join(root, param)
    target_path = os.path.join(root, target)
    if os.path.exists(target_path):
        return None
    if os.path.exists(code_path):
        con = []
        with open(code_path, 'r',encoding='utf-8') as f:
            res = f.read()
            res = res.replace('[','').replace(']','')
            list_res = res.split(',')
            for list_index in range(len(list_res)):
                if delestat[list_index] == False:
                    tmp = float(list_res[list_index])
                    con.append(tmp)
        with open(target_path, 'a', encoding='utf-8') as file:
            file.write(str(con))
    return None

# 统计测试用例
def analysisTest(testPath):
    allPass = 0
    allFiled = 0
    failIndex = []
    with open(os.path.join(testPath,'inVector.in'), "rb") as tests:
        inVector = pickle.load(tests)
        if len(inVector) == 0:
            return None, None, None, None, None
        for t_index in range(len(inVector)):
            if inVector[t_index] == 0:
                allPass = allPass +1
            else:
                allFiled = allFiled + 1
                failIndex.append(t_index)

    return failIndex, inVector, len(inVector), allPass, allFiled


def getFailedTest(testPath):
    with open(testPath + "/failing_tests", "r") as failing_tests:
        failingLines = failing_tests.readlines()
    with open(testPath + "/all_tests", "r") as all_tests:
        allLines = all_tests.readlines()

    failTest = []
    failIndex = []
    for line in failingLines:
        if line.startswith("--- "):
            line = line[3:]
            line = line.strip()
            strs = line.split("::")
            failTest.append(strs[1] + "(" + strs[0] + ")")

    inVector = []
    for index in range(len(allLines)):
        line = allLines[index].strip()
        if line in failTest:
            failIndex.append(index)
            inVector.append(1)
        else:
            inVector.append(0)

    return failIndex, inVector, len(allLines), len(allLines) - len(failTest), len(failTest)

def saveAgainCheck(covMatrix,statementIndex):
    AllZeroFlag = True
    for testCaseIndex in range(len(covMatrix)):
        if covMatrix[testCaseIndex][statementIndex] == 1:
            AllZeroFlag = False
            break
    #print(statementIndex)
    return AllZeroFlag

def readFile(versionPath, error_pro_ver, saveSpace=False, saveAgain=True):
    if saveAgain:
        savedata = checkAndLoad(versionPath[1], "data_Coverage_InVector_saveAgain")
    elif saveSpace:
        savedata = checkAndLoad(versionPath[1], "data_Coverage_InVector_saveSpace")
    else:
        savedata = checkAndLoad(versionPath[1], "data_Coverage_InVector")
    if savedata != None:
        covMatrix = savedata[1]
        fault = savedata[2]
        in_vector = savedata[3]
        failN = savedata[4]
        passN = savedata[5]
        realCC = savedata[6]
        failIndex = savedata[7]
        trueCC = savedata[8]
        return covMatrix, fault, in_vector, failN, passN, realCC, failIndex, trueCC

    covMatrix_path = os.path.join(versionPath[0], "CoverageMatrix.in")
    hugeCode_path = os.path.join(versionPath[0], "hugeCode.txt")
    inVector_path = os.path.join(versionPath[0], "inVector.in")

    #判断版本文件是否缺失
    if not os.path.exists(covMatrix_path) or not os.path.exists(hugeCode_path) or not os.path.exists(inVector_path):
        f_error = open(os.path.join(error_pro_ver, 'error_loc.txt'), 'a+')
        f_error.writelines(versionPath[0] + " 缺少文件\r\n")
        f_error.close()
        return None, None, None, None, None, None, None, None

    fault_position = checkAndLoad(versionPath[0], "faultHuge.in")
    fault = []
    originfault=[]
    for javaFile in fault_position:
        for key in fault_position[javaFile]:
            temp = key - 1
            fault.append(temp)
            originfault.append(temp)

    if saveSpace:
        with open(hugeCode_path) as f:
            hugeCode=f.readlines()

    with open(covMatrix_path, 'rb') as f:
        tempCovMatrix = pickle.load(f)
        # 判断覆盖矩阵是否为空
        if len(tempCovMatrix) == 0:
            f_error = open(os.path.join(error_pro_ver, 'error_loc.txt'), 'a+')
            f_error.writelines(versionPath[0]+" 覆盖信息内容为空\r\n")
            f_error.close()
            return None, None, None, None, None, None, None, None

    covMatrix = []
    originRead=True

    if saveAgain==True:
        delStatementIndex = checkAndLoad(versionPath[1], "data_saveAgain_del_statement_index")
        if delStatementIndex!=None:
            originRead=False

    #查看是否存在测试用例未执行程序
    # covMatrix2=[]
    # for i in trange(len(tempCovMatrix)):
    #     tempStatementlist = tempCovMatrix[i].strip().split(' ')
    #     tempcovMatrix = []
    #     for j in range(len(tempStatementlist)):
    #         tempStatement = tempStatementlist[j]
    #         if tempStatement != "":
    #             tempcovMatrix.append(int(tempStatement))
    #     covMatrix2.append(tempcovMatrix)
    # flag = 0
    # for i in range(len(covMatrix2)):
    #     for j in range(len(covMatrix2[i])):
    #         if (covMatrix2[i][j] == 1):
    #             flag = 1
    #             break
    #     if (flag == 0):
    #         print("not exist 1", i)
    #     flag = 0


    if originRead:
        for i in trange(len(tempCovMatrix)):
            tempStatementlist = tempCovMatrix[i]
            tempcovMatrix = []
            for j in range(len(tempStatementlist)):
                if saveSpace:
                    if hugeCode[j].startswith("package ") or hugeCode[j].startswith("import "):
                        if i==0:
                            for faultIndex in range(len(fault)):
                                if j < originfault[faultIndex]:
                                    fault[faultIndex]-=1
                        continue
                tempStatement = tempStatementlist[j]
                if tempStatement != "":
                    tempcovMatrix.append(int(tempStatement))
            covMatrix.append(tempcovMatrix)

    if saveAgain:
        delStatementIndex = checkAndLoad(versionPath[1], "data_saveAgain_del_statement_index")
        if delStatementIndex==None:
            delStatementIndex=[]
            for statementIndex in range(len(covMatrix[0])):
                tempresult=saveAgainCheck(covMatrix,statementIndex)
                delStatementIndex.append(tempresult)

            # pool = multiprocessing.Pool(processes=8)
            # for statementIndex in range(len(covMatrix[0])):
            #     tempIndex=pool.apply_async(saveAgainCheck,(covMatrix,statementIndex,))
            #     delStatementIndex.append(tempIndex)
            # for index in range(len(delStatementIndex)):
            #     delStatementIndex[index]=(delStatementIndex[index]).get()
            checkAndSave(versionPath[1], "data_saveAgain_del_statement_index", delStatementIndex)

        if os.path.exists(os.path.join(versionPath[0], "hugeCode.txt")):
            create_duplicate_code(versionPath,"hugeCode.txt","hugeCodeCopy.txt",delStatementIndex)
            #create_duplicate_static(versionPath,"static_fea.txt","static_fea_copy.txt",delStatementIndex)

        tempCount=0
        for item in delStatementIndex:
            if item==False:
                tempCount+=1
        #print(tempCount)

        for statementIndex in range(len(delStatementIndex)):
            if delStatementIndex[statementIndex]==True:
                for faultIndex in range(len(fault)):
                    if statementIndex < originfault[faultIndex]:
                        fault[faultIndex] -= 1
        covMatrix=[]
        for i in trange(len(tempCovMatrix)):
            tempStatementlist = tempCovMatrix[i]
            tempcovMatrix = []
            for j in range(len(tempStatementlist)):
                if delStatementIndex[j]==False:
                    tempStatement = tempStatementlist[j]
                    if tempStatement != "":
                        tempcovMatrix.append(int(tempStatement))
            covMatrix.append(tempcovMatrix)

    #failIndex, inVector, allN, passN, failN = getFailedTest(versionPath)
    failIndex, inVector, allN, passN, failN = analysisTest(versionPath[0])
    # 判断测试用例结果向量是否为空
    if failIndex==None or inVector==None or allN==None or passN==None or failN==None:
        f_error = open(os.path.join(error_pro_ver, 'error_loc.txt'), 'a+')
        f_error.writelines(versionPath[0] + " 测试用例结果向量为空\r\n")
        f_error.close()
        return None, None, None, None, None, None, None, None


    realCC = {}
    for index in range(len(inVector)):
        for faultLine in fault:
            if covMatrix[index][faultLine] == 1:
                if inVector[index] == 0:
                    if index in realCC:
                        realCC[index].append(faultLine)
                    else:
                        realCC[index] = []
                        realCC[index].append(faultLine)
                #     print(index,"是CC")
                # else:
                #     print(index,"是执行了错误语句的fail测试用例")

    #trueCC = []
    # # 失败测试用例
    # fail_cov = []
    # for fi in failIndex:
    #     fail_cov.append(covMatrix[fi])
    # fail_cov = np.array(fail_cov)
    # # stf集合 失败测试用例共同执行过的语句
    # stf = []
    # for col in range(np.shape(fail_cov)[1]):
    #     col_val = fail_cov[:, col]
    #     same_flag = 0
    #     for x in col_val:
    #         if x != 1:
    #             same_flag = 1
    #     if same_flag == 0:
    #         stf.append(col)
    # # 计算
    # for row_cov in range(len(covMatrix)):
    #     if row_cov not in failIndex:
    #         correct_flag = 0
    #         for stf_index in stf:
    #             if covMatrix[row_cov][stf_index] != 1:
    #                 correct_flag = 1
    #         if correct_flag == 0:
    #             trueCC.append(row_cov)
    #trueCC = Tool_CC.getTCC(covMatrix,inVector)
    trueCC = Tool_CC.getTCC(covMatrix,inVector)

    savedata = {}
    savedata[1] = covMatrix
    savedata[2] = fault
    savedata[3] = inVector
    savedata[4] = failN
    savedata[5] = passN
    savedata[6] = realCC
    savedata[7] = failIndex
    savedata[8] = trueCC

    # 判断失败测试用例覆盖信息是否全为0
    res = test_error(covMatrix,inVector)
    if res == 0:
        f_error = open(os.path.join(error_pro_ver, 'error_loc.txt'), 'a+')
        f_error.writelines(versionPath[0]+" 失败测试用例覆盖信息全为0\r\n")
        f_error.close()
        return None, None, None, None, None, None, None, None
    # 判断是否存在失败测试用例
    if len(failIndex) == 0 :
        f_error = open(os.path.join(error_pro_ver, 'error_loc.txt'), 'a+')
        f_error.writelines(versionPath[0] + " 无失败测试用例\r\n")
        f_error.close()
        return None, None, None, None, None, None, None, None

    if saveAgain:
        checkAndSave(versionPath[1], "data_Coverage_InVector_saveAgain", savedata)
    elif saveSpace:
        checkAndSave(versionPath[1], "data_Coverage_InVector_saveSpace", savedata)
    else:
        checkAndSave(versionPath[1], "data_Coverage_InVector", savedata)
    return covMatrix, fault, inVector, failN, passN, realCC, failIndex, trueCC


# 失败测试用例覆盖信息是否全为0
def test_error(covMatrix, failIndex):
    res = 1
    for f_index in failIndex:
        flag = 0
        row = covMatrix[f_index]
        for index in row:
            if index == 1:
                flag = 1
                break
        res = res & flag
    return res


# 读取一个文件夹，返回里面所有的文件名
def get_folder(folder_path):
    dirs=os.listdir(folder_path)
    return dirs


# 创建数据文件夹ss
def create_data_folder(root, data, originPath=""):
    res = {}
    all_files = []
    dirs = get_folder(root)
    dirs.sort()
    for dir in dirs:
        # 数据集程序路径
        dump_path = os.path.join(root, dir)
        # 数据集程序对应计算结果存储路径
        data_path = os.path.join(data, dir)
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        sub_dirs = get_folder(dump_path)
        for sub_dir in sub_dirs:
            # 数据集程序不同版本子路径
            version_path = os.path.join(dump_path, sub_dir)
            # 数据集程序不同版本计算结果存放路径
            res_path = os.path.join(data_path, sub_dir)

            originDataPath=os.path.join(originPath,dir,sub_dir)
            if not os.path.exists(res_path):
                os.mkdir(res_path)
            tmp = []
            tmp.append(version_path)
            tmp.append(res_path)
            tmp.append(originDataPath)
            all_files.append(tmp)
        # 某个程序的全部版本路径
        res[dump_path] = all_files
        # 注意清空上一个程序包含的不同版本代码
        all_files = []
    return res


# 统计计算结果
def cal_res(data_path, pro_name, res_path, keys):
    # 获取程序所有版本数据路径
    pro_data_path = os.path.join(data_path, pro_name)
    # 获取程序所有每个版本数据路径
    # pro_vers_path = get_folder(pro_data_path)
    pro_vers_path = keys
    # cc识别结果路径
    pro_vers = []
    for ver in pro_vers_path:
        pro_ver = os.path.join(pro_data_path, ver)
        # cc识别结果路径
        pro_vers.append(pro_ver)
    # 统计结果
    recall = 0
    FPrate = 0
    precision = 0
    Fmeasure = 0
    data = []
    for index_path in pro_vers:
        if not os.path.exists(os.path.join(index_path,'cc_identity_result')):
            continue
        f = open(os.path.join(index_path,'cc_identity_result'), 'rb')
        identity_result = pickle.load(f)
        # 各类指标求和ss
        recall = recall + identity_result['recall']
        FPrate = FPrate + identity_result['FPrate']
        precision = precision + identity_result['precision']
        Fmeasure = Fmeasure + identity_result['Fmeasure']
        # 存储每一个版本的结果
        row = []
        row.append(os.path.basename(index_path))
        row.append(identity_result['recall'])
        row.append(identity_result['FPrate'])
        row.append(identity_result['precision'])
        row.append(identity_result['Fmeasure'])
        data.append(row)
        f.close()
    avg_recall = recall / len(pro_vers)
    avg_FPrate = FPrate / len(pro_vers)
    avg_precision = precision / len(pro_vers)
    avg_Fmeasure = Fmeasure / len(pro_vers)
    row = []
    row.append('avg')
    row.append(avg_recall)
    row.append(avg_FPrate)
    row.append(avg_precision)
    row.append(avg_Fmeasure)
    data.append(row)
    # 存储结果
    csv_res(pro_name, data, res_path)


# 将结果存入csv文件中s
def csv_res(pro_name, data, res_path):
    path = os.path.join(res_path, pro_name)+".csv"
    header = ['', 'recall', 'FPrate', 'precision', 'Fmeasure']
    with open(path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for row in data:
            writer.writerow(row)


# 增加/删除怀疑度公式 a:怀疑度公式字典，b:已用怀疑度公式算好的特征
def compare_sus(a, b):
    x = list(a.keys())
    y = list(b.keys())
    for index in y:
        if index not in x:
            b.pop(index)
    for index in x:
        if index in y:
            a.pop(index)
    return a, b


# 怀疑度公式增减
def add_del_formula(features,formulaSus_origin,path,name):
    # 原始特征维度
    origin_len = len(features)
    # 怀疑度公式类型是否有增减
    formulaSus_copy = copy.deepcopy(formulaSus_origin)
    formulaSus,features = compare_sus(formulaSus_copy, features)
    # 怀疑度公式与现有特征用到的怀疑度公式类型是否有减少
    now_len = len(features)
    if now_len < origin_len:
        checkDele(path, name)
        if len(formulaSus) == 0:
            checkAndSave(path, name, features)
            return features, formulaSus, 0
    if len(formulaSus) > 0:
        checkDele(path, name)
    else:
        # 怀疑度公式与现有SS特征用到的怀疑度公式类型没有增加，原结果不变或删减后可直接返回
        return features, formulaSus, 0
    return features,formulaSus, 1


def getSrcPath(formatCodePath, programName, versionInt):
    if not formatCodePath.lower().endswith(".java"):
        fileline=os.path.join(formatCodePath,"src")+"/"
    else:
        fileline=formatCodePath
    key=programName
    id=versionInt
    if key == "Chart":
        fileline = fileline.replace("src/", "source/")
    elif key == "Time":
        fileline = fileline.replace("src/", "src/main/java/")
    elif key == "Lang":
        if id >= 36:
            fileline = fileline.replace("src/", "src/java/")
        else:
            fileline = fileline.replace("src/", "src/main/java/")
    elif key == "Math":
        if id >= 85:
            fileline = fileline.replace("src/", "src/java/")
        else:
            fileline = fileline.replace("src/", "src/main/java/")
    elif key == "Cli":
        if id >= 30:
            fileline = fileline.replace("src/", "src/main/java/")
        else:
            fileline = fileline.replace("src/", "src/java/")
    elif key == "Codec":
        if id >= 11:
            fileline = fileline.replace("src/", "src/main/java/")
        else:
            fileline = fileline.replace("src/", "src/java/")
    elif key == "Compress":
        fileline = fileline.replace("src/", "src/main/java/")
    elif key == "Csv":
        fileline = fileline.replace("src/", "src/main/java/")
    elif key == "Gson":
        fileline = fileline.replace("src/", "gson/src/main/java/")
    elif key == "JacksonCore":
        fileline = fileline.replace("src/", "src/main/java/")
    elif key == "JacksonDatabind":
        fileline = fileline.replace("src/", "src/main/java/")
    elif key == "JacksonXml":
        fileline = fileline.replace("src/", "src/main/java/")
    elif key == "Jsoup":
        fileline = fileline.replace("src/", "src/main/java/")
    elif key == "JxPath":
        fileline = fileline.replace("src/", "src/java/")
    return fileline

if __name__ == "__main__":
    get_case_num("/mnt/f/FLN/filter/")
