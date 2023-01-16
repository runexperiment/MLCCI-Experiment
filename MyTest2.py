import pickle
from multiprocessing import Pool
import os, time

from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


import Tool_io


def func(n):
    print(n)

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


# 创建结果文件
def creat_res_file(csv_name, row=None):
    print(row)
    print("s")


# 获取数据集
def deal_program(data_path, pro_name):

    # 遍历不同程序
    pro_path = os.path.join(data_path, pro_name)
    ver_names = Tool_io.get_folder(pro_path)
    # 遍历程序不同版本
    pro_dataset = []
    for ver_name in ver_names:
        print(ver_name)
        ver_path = os.path.join(pro_path, ver_name)
        # 获取程序版本信息
        coverage_info = Tool_io.checkAndLoad(ver_path, 'data_Coverage_InVector_saveAgain')
        if coverage_info is not None:
            # 获取归一化后的特征
            res_nor = Tool_io.checkAndLoad(ver_path, 'data_FD_all')
            if res_nor is not None and len(coverage_info[2]) > 0:
                pro_dataset.append(ver_name)
    if len(pro_dataset) > 0:
        return  pro_dataset


if __name__ == '__main__':

    path = '/home/tianshuaihua/base_pydata'
    folder = Tool_io.get_folder(path)
    res = '/home/tianshuaihua/base_res'
    model_path = '/home/tianshuaihua/model'


    pro_dict = {}
    # pro_dict[1] = '23'
    # Tool_io.checkAndSave('/home/tianshuaihua/wang/fea', 'vers', pro_dict)
    # ss= Tool_io.checkAndLoad('/home/tianshuaihua/wang/fea', 'vers')

    for index in folder:
        if index == 'Chart' or index == 'Lang' or index == 'Math' or index == 'Mockito' or index == 'Time' or index == 'Closure':
        # if index == 'Chart':
            # f = open(os.path.join(model_path, index + '.td'), 'rb')
            # td = pickle.load(f)
            # keys = list(td.keys())
            print(index)
            ver_list = deal_program(path, index)
            pro_dict[index] = ver_list
            Tool_io.cal_res(path,index,res,ver_list)
            print("s")
    Tool_io.checkAndSave('/home/tianshuaihua/wang/fea','vers.in',pro_dict)




    # X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.7)
    # score = clf.score(X_test, y_test)
    # y_predicate = clf.predict(X_test)
    # res = confusion_matrix(y_test, y_predicate)
    # cmatrix = sklearn.metrics.classification_report(y_test, y_predicate)


    # wine = load_wine()
    # X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)
    # clf = tree.DecisionTreeClassifier(criterion='entropy')
    # clf = clf.fit(X_train, y_train)
    # score = clf.score(X_test, y_test)
    # # 0.9259259259259259
    # print(score)
    # print("s")

    # pool_size = 8
    # p = Pool(pool_size)
    # for i in range(5):
    #     p.apply(func=func, args=(5,))  # 异步

    # print(1 & 3)
    # a = [1,0,0,1,0,1,1,0,1,1]
    # b = [0,0,0,1,0,0,0,0,0,0]
    # c = [1,0,0,1,1,0,1,0,1,1]
    #
    # d = [1,0,0,1,0,0,1,1,1,0]
    # e = [0,1,0,1,1,0,1,1,1,1]
    # f = [0,0,0,0,0,0,0,0,0,0]
    # g = [0,0,1,1,1,0,1,1,1,0]
    #
    # cov = []
    # cov.append(a)
    # cov.append(b)
    # cov.append(c)
    # cov.append(d)
    # cov.append(e)
    # cov.append(f)
    # cov.append(g)
    #
    # res = test_error(cov,[1,1,1,1,1,0,1])
    #
    # print("s")
    # formulaSus, notSort = Tool_localization.statement_sus(case_num, statement_num, covMatrix, inVector, versionPath)

    # root = 'D:\\CC\\data_test\\data'
    # dirs = Tool_io.get_folder(root)
    # folder = []
    # for dir in dirs:
    #     dump_path = os.path.join(root, dir)
    #     folder.append(dump_path)
    #
    # all_files = []
    # for path in folder:
    #     files = Tool_io.get_folder(path)
    #     for file in files:
    #         version_path = os.path.join(path, file)
    #         all_files.append(version_path)
    #
    #



# import copy
# import linecache
# import os.path
# import sys
# import time
#
# import fuzzyCal
#
# from tqdm import trange
#
# import FuzzyKnn
# import Tool_io
# import Tool_localization
#
# # _formulaList={"crosstab", "dstar", "jaccard", "ochiai", "op2", "turantula"}
# from Sttaic_code_features import all_operators
#
# _formulaList={"dstar", "jaccard", "ochiai"}
# _CCstrategyList={"clean", "exchange", "relabel"}
#
#
# def getSuspiciousnessFDistance(versionPath,covMatrix,inVector,formulaSus,use_list=None,baseline=True):
#     print("模糊加权距离")
#     if baseline:
#         FD_All = Tool_io.checkAndLoad(versionPath, "data_FD_all")
#         if FD_All != None:
#             return FD_All
#     FD_All = {}
#     if baseline:
#         for algorithm_k, algorithm_v in formulaSus.items():
#             if algorithm_k == 'dstar' or algorithm_k == 'ochiai' or algorithm_k == 'jaccard':
#             #if algorithm_k == 'dstar' or algorithm_k == 'dstar' or algorithm_k == 'dstar':
#                 FD = {}
#                 for case_index in trange(len(covMatrix)):
#                     test = copy.deepcopy(covMatrix[case_index])
#                     for key, value in formulaSus[algorithm_k]:
#                         test[key] *= value
#                     FD[case_index] = test
#                 FD_All[algorithm_k] = FD
#         Tool_io.checkAndSave(versionPath, "data_FD_all", FD_All)
#         print("FD end")
#     return FD_All
#
#
# def getSuspiciousnessScoreFactorSimple(versionPath,covMatrix,inVector,formulaSus,use_list=None,baseline=True):
#     print("SS start")
#     if baseline:
#         SS_ALL = Tool_io.checkAndLoad(versionPath, "data_SS_all")
#         if SS_ALL != None:
#             return SS_ALL
#
#     SS_ALL = {}
#     if baseline:
#         for algorithm_k, algorithm_v in formulaSus.items():
#             SS = {}
#             for case_index in trange(len(covMatrix)):
#                 if use_list == None or len(use_list) == 0 or case_index in use_list:
#                 #if inVector[case_index] == 0:
#                     Sh_Num=0
#                     Sh_Sum=0
#                     Sl_Num=0
#                     Sl_Sum=0
#                     for key,value in formulaSus[algorithm_k]:
#                         if covMatrix[case_index][key]==1:
#                             if value>=0.5:
#                                 Sh_Num+=1
#                                 Sh_Sum+=value
#                             else:
#                                 Sl_Num+=1
#                                 Sl_Sum+=value
#                     if Sh_Num>0:
#                         SS[case_index]=Sh_Sum/Sh_Num
#                     else:
#                         if Sl_Num == 0:
#                             SS[case_index] = 0
#                         else:
#                             SS[case_index] = Sl_Sum / Sl_Num
#             SS_ALL[algorithm_k] = SS
#         Tool_io.checkAndSave(versionPath, "data_SS_all", SS_ALL)
#         print("SS end")
#     return SS_ALL
#
# def getCoverageRatioFactor(versionPath, covMatrix, inVector,formulaSus,baseline=True):
#     print("CR start")
#     #若存在CR结果，则直接返回
#     if baseline:
#         CR_ALL = Tool_io.checkAndLoad(versionPath, "data_CR_all")
#         if CR_ALL != None:
#             return CR_ALL
#     CR_ALL = {}
#     for algorithm_k, algorithm_v in formulaSus.items():
#         CR = {}
#         for case_index in trange(len(covMatrix)):
#             #if inVector[case_index] == 0:
#             S_num = 0
#             Sp_num = 0
#             for key, value in formulaSus[algorithm_k]:
#                 #怀疑分数在0.5-1之间的语句数量
#                 if(value>=0.5 and value<=1):
#                     S_num += 1
#                     # 计算怀疑分数在0.5-1之间的并且被执行的语句数量
#                     if covMatrix[case_index][key] == 1:
#                         Sp_num += 1
#             #计算CR值
#             if S_num == 0:
#                 CR[case_index] = 0
#             else:
#                 CR[case_index] = Sp_num / S_num
#         CR_ALL[algorithm_k] = CR
#     # 将计算结果保存至文件
#     Tool_io.checkAndSave(versionPath, "data_CR_all", CR_ALL)
#     return CR_ALL
#
#
# def getSimilarityFactor(versionPath,covMatrix,inVector,formulaSus,baseline=True):
#     print("SF start")
#     # 若存在SF结果，则直接返回
#     if baseline:
#         SF_ALL = Tool_io.checkAndLoad(versionPath, "data_SF_all")
#         if SF_ALL != None:
#             return SF_ALL
#     SF_ALL = {}
#     for algorithm_k, algorithm_v in formulaSus.items():
#         SF = {}
#         Ef = []
#     #计算失败测试用例的加权向量
#         for case_failed in range(len(covMatrix)):
#             if inVector[case_failed] == 1:
#                 fi = copy.deepcopy(covMatrix[case_failed])
#                 for key, value in formulaSus[algorithm_k]:
#                     fi[key] *= value
#                 Ef.append(fi)
#         for case_index in trange(len(covMatrix)):
#             #if inVector[case_index] == 0:
#             #计算Ep的加权向量
#             Ep = copy.deepcopy(covMatrix[case_index])
#             for key, value in formulaSus[algorithm_k]:
#                 Ep[key] *= value
#             #Efi和Ep距离初始值为最大值，也就是程序语句数
#             distance = len(covMatrix[case_index])
#             # 计算Efi和Ep之间的最短距离
#             for fail_index in range(len(Ef)):
#                 sum = 0
#                 for index in range(len(Ef[fail_index])):
#                     sum = sum+(Ef[fail_index][index]-Ep[index])**2
#                 if sum**0.5 < distance:
#                     distance = sum**0.5
#             #计算SF值
#             if distance == 0:
#                 SF[case_index] = 0
#             else:
#                 SF[case_index] = 1 / distance
#         SF_ALL[algorithm_k] = SF
#     #将计算结果保存至文件
#     Tool_io.checkAndSave(versionPath, "data_SF_all", SF_ALL)
#     return SF_ALL
#
#
# def getFaultMaskingFactor(versionPath,covMatrix,inVector,notSort,target,baseline=True):
#     print("FM start")
#     # 若存在FM结果，则直接返回
#     if baseline:
#         FM_ALL = Tool_io.checkAndLoad(versionPath, "data_FM_all")
#         if FM_ALL != None:
#             return FM_ALL
#     #impact factor
#     operators={
#         '>':0.79,
#         '<':0.79,
#         '=':0.79,
#         '>=':0.79,
#         '<=':0.79,
#         '+':0.08,
#         '-':0.08,
#         '*':0.08,
#         '/':0.08,
#         '%':0.08,
#     }
#     FM_ALL = {}
#     for algorithm_k, algorithm_v in notSort.items():
#         FM = {}
#         for case_index in trange(len(covMatrix)):
#             #if inVector[case_index] == 0:
#             for stat_index in range(len(covMatrix[case_index])):
#                 if covMatrix[case_index][stat_index] == 1:
#                     #查找怀疑度大于0.5的第一条被执行过的语句
#                     if(algorithm_v[stat_index])>0.5:
#                         #查找怀疑度大于0.5的第一条被执行过的语句 的下一条语句
#                         list_index = stat_index+1
#                         #遍历之后的语句
#                         sum_op = 0
#                         while list_index < len(covMatrix[case_index]):
#                             if covMatrix[case_index][list_index] == 1:
#                                 text = linecache.getline(target,list_index+1)
#                                 #取 operators 最大值
#                                 impact_factor  = 0
#                                 for k,v in operators.items():
#                                  if text.find(k):
#                                     if v > impact_factor:
#                                         impact_factor = v
#                                 #语句impact_factor求和
#                                 sum_op += impact_factor
#                             list_index += 1
#                         FM[case_index] = sum_op
#                         break
#                     else:
#                         FM[case_index] = 0
#                 else:
#                     FM[case_index] = 0
#             FM_ALL[algorithm_k] = FM
#     Tool_io.checkAndSave(versionPath, "data_FM_all", FM_ALL)
#     return FM_ALL
#
#
# def getStatementFactor(versionPath,covMatrix,inVector,notSort,target,baseline=True):
#     print("Statement start")
#     # 若存在结果，则直接返回
#     if baseline:
#         ST_ALL = Tool_io.checkAndLoad(versionPath, "data_statement_all")
#         if ST_ALL != None:
#             return ST_ALL
#     ST_ALL = {}
#     for case_index in trange(len(covMatrix)):
#         #if inVector[case_index] == 0:
#         num_operators = 0
#         num_operands = 0
#         for stat_index in range(len(covMatrix[case_index])):
#             if covMatrix[case_index][stat_index] == 1:
#                 text = linecache.getline(target, stat_index+1)
#                 op, op_key = all_operators()
#                 operators = op_key
#                 operands = {}
#                 if (not text.startswith("//")):
#                     text_tmp = ' ' + text.split('//')[0]
#                     for key in operators.keys():
#                         if key not in op:
#                             count = text_tmp.count(' ' + key + ' ')
#                             operators[key] = operators[key] + count
#                             if count > 0:
#                                 text_tmp = text_tmp.replace(' ' + key + ' ', ' ')
#                         else:
#                             operators[key] = operators[key] + text_tmp.count(key)
#                             text_tmp = text_tmp.replace(key, ' ')
#                     for key in text_tmp.split():
#                         if key in operands:
#                             operands[key] = operands[key] + 1
#                         else:
#                             if key != '':
#                                 operands[key] = 1
#                 for k, v in operators.items():
#                     num_operators += v
#                 # 计算操作数数量和操作数种类数量
#                 for m, n in operands.items():
#                     num_operands += n
#         tmp = {}
#         tmp['num_operators'] = num_operators
#         tmp['num_operands'] = num_operands
#         tmp['loc_len'] = sys.getsizeof(text)
#         ST_ALL[case_index] = tmp
#     Tool_io.checkAndSave(versionPath, "data_statement_all", ST_ALL)
#     return ST_ALL
#
#
# # 计算静态特征
# def getStaticFeature(versionPath,covMatrix,inVector,ochiai,target,baseline=True):
#     # 若存在结果，则直接返回
#     if baseline:
#         STATIC = Tool_io.checkAndLoad(versionPath, "data_STATIC_baseline")
#         if STATIC != None:
#             return STATIC
#     STATIC = {}
#
#     with open(target,'r',encoding='utf-8') as f:
#         res = f.read()
#         res = res.replace('[','').replace(']','')
#         con = res.split(',')
#
#     for case_index in trange(len(covMatrix)):
#         if inVector[case_index] == 0:
#             S_num = 0
#             Sp_num = 0
#             SFPL = 0
#             for key, value in formulaSus["ochiai"]:
#                 #怀疑分数在0.5-1之间的语句数量
#                 if(value>=0.5 and value<=1):
#                     S_num += 1
#                     # 计算怀疑分数在0.5-1之间的并且被执行的语句数量
#                     if covMatrix[case_index][key] == 1:
#                         SFPL += float(con[key])
#             #计算CR值
#             STATIC[case_index] = SFPL / S_num
#     Tool_io.checkAndSave(versionPath, "data_STATIC_baseline", STATIC)
#     return STATIC
#
#
# # 执行程序
# def execution(files):
#     for index in files:
#         versionPath = "D:\\CC\\data_test\\Chart\\1b"
#         covMatrix, fault, inVector, failN, passN, realCC, failIndex, trueCC = Tool_io.readFile(versionPath)
#         statement_num = len(covMatrix[0])
#         case_num = len(covMatrix)
#         formulaSus, notSort = Tool_localization.statement_sus(case_num, statement_num, covMatrix, inVector, versionPath)
#         sus_score = Tool_localization.statement_sus(case_num, statement_num, covMatrix, inVector, versionPath)
#
#         # 模糊加权距离
#         # FDistance = getSuspiciousnessFDistance(versionPath, covMatrix, inVector, formulaSus)
#         # WPcc = FuzzyKnn.fuzzy_knn(FDistance, failIndex, versionPath,'ochiai')
#         # res = FuzzyKnn.fuzzy_knn_metric(realCC, WPcc)
#
#         # 计算SS
#         SS = getSuspiciousnessScoreFactorSimple(versionPath, covMatrix, inVector, sus_score[0])
#         # 计算CR
#         CR = getCoverageRatioFactor(versionPath, covMatrix, inVector, sus_score[0])
#         # 计算SF
#         SF = getSimilarityFactor(versionPath, covMatrix, inVector, sus_score[0])
#         # 计算FM
#         target = os.path.join(versionPath, "hugeCodeCopy.txt")
#         FM = getFaultMaskingFactor(versionPath, covMatrix, inVector, sus_score[1], target)
#
#         res_nor = Tool_localization.normalization(SS, CR, SF, FM, versionPath)
#         deal_data = FuzzyKnn.fuzzy_knn(res_nor['normal'], failIndex, versionPath)
#         res = FuzzyKnn.fuzzy_knn_metric(realCC, deal_data, versionPath)
#
#
# if __name__ =="__main__":
#     versionPath="D:\\CC\\data_test\\Chart\\1b"
#     covMatrix, fault, inVector, failN, passN,realCC,failIndex, trueCC=Tool_io.readFile(versionPath)
#     #print(len(covMatrix))
#     #print(len(covMatrix[0]))
#     statement_num = len(covMatrix[0])
#     case_num = len(covMatrix)
#     formulaSus,notSort = Tool_localization.statement_sus(case_num, statement_num, covMatrix,inVector,versionPath)
#     sus_score = Tool_localization.statement_sus(case_num, statement_num, covMatrix,inVector,versionPath)
#
#     # for key in realCC:
#     #     inVector[key]=1
#
#     # sus_oc, sus_tu, sus_op, sus_ds, sus_cr, sus_j=Tool_localization.SBFL_location(covMatrix,inVector)
#     # formulaSusList = {"dstar":sus_ds, "ochiai":sus_oc, "turantula":sus_tu, "op2":sus_op, "crosstab":sus_cr, "jaccard":sus_j}
#     # for key in _formulaList:
#     #     cost, exam, fault_location=Tool_metric.getEXAM(formulaSusList[key],fault)
#     #     print(cost)
#
#     #模糊加权距离
#     # FDistance = getSuspiciousnessFDistance(versionPath, covMatrix, inVector, formulaSus)
#     # WPcc = FuzzyKnn.fuzzy_knn(FDistance, failIndex, versionPath,'ochiai')
#     #res = FuzzyKnn.fuzzy_knn_metric(realCC, WPcc)
#
#     #计算SS
#     SS = getSuspiciousnessScoreFactorSimple(versionPath,covMatrix,inVector,sus_score[0])
#     #计算CR
#     CR = getCoverageRatioFactor(versionPath,covMatrix,inVector,sus_score[0])
#     #计算SF
#     SF = getSimilarityFactor(versionPath,covMatrix,inVector,sus_score[0])
#     #计算FM
#     target=os.path.join(versionPath, "hugeCodeCopy.txt")
#     FM = getFaultMaskingFactor(versionPath,covMatrix,inVector,sus_score[1],target)
#
#
#     res_nor = Tool_localization.normalization(SS,CR,SF,FM,versionPath)
#     deal_data = FuzzyKnn.fuzzy_knn(res_nor['normal'], failIndex, versionPath)
#     res = FuzzyKnn.fuzzy_knn_metric(realCC, deal_data, versionPath)
#
#     # 回归静态特征
#     #target = os.path.join(versionPath, "hugeCodeCopy.txt")
#     #STATMENT = getStatementFactor(versionPath, covMatrix, inVector, notSort, target)
#
#
#     # 计算STATAIC
#     #targetStatic = os.path.join(versionPath, "static_fea_copy.txt")
#     #STATIC_FEA = getStaticFeature(versionPath, covMatrix, inVector, ochiai, targetStatic)
#
#
#     # for k,v in SF.items():
#     #    if v>1:
#     #        print((k,v))
#
#     # print(time.time())
#     # matlab_fun = fuzzyCal.initialize()
#     # cal = matlab_fun.fuzzyCal(float(0.0), float(0.0), float(0.618522873486732), float(0.0), float(0.0))
#     # print(cal)
#
#
#     # result = []
#     # matlab_fun = fuzzyCal.initialize()
#     # for index in trange(len(SS)):
#     #     if SS.__contains__(index):
#     #         one = float(SS[index])
#     #         two = float(CR[index])
#     #         three = float(SF[index])
#     #         four = float(FM[index])
#     #         five = float(STATIC_FEA[index])
#     #         cal = matlab_fun.fuzzyCal(one,two,three,four,five)
#     #         result.append(cal)
#     #
#     # with open('D:\\CC\\data_test\\result.txt','a',encoding='utf-8') as f:
#     #     for i in result:
#     #         f.write(str(i)+'\n')
#
#
#     #engine = matlab.engine.start_matlab()
#     #result=engine.Cal(2,3)
#     #result = engine.fuzzyCal(0.6,0.4,0.7,0.8,0.5)
#     #print(result)
#
#
#
#
#     # print(time.time())
#     # matlab_fun = fuzzyCal.initialize()
#     # cal = matlab_fun.fuzzyCal(0.4, 0.6, 0.7, 0.8, 0.6)
#     # print(cal)
#
#
