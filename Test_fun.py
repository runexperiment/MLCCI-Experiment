# coding=utf-8
import os

import Tool_io
import featureExtract
import Tool_localization
import math
if __name__ == "__main__":
    content = Tool_io.checkAndLoad('/home/tianshuaihua/pydata/Time/1b','data_Coverage_InVector_saveAgain')
    vector = content[3]
    cc = content[6]
    for index in cc:
        vector[index] = 1

    formula= Tool_localization.deal_suspicion_formula()

    # covMatrix = [
    #     [1, 0, 1, 1, 0],
    #     [0, 1, 1, 1, 0],
    #     [1, 1, 0, 0, 1],
    #     [0, 0, 1, 1, 1],
    # ]
    covMatrix = content[1]

    # inVector = [0, 1, 0, 1]

    statement_num = len(covMatrix[0])
    case_num = len(covMatrix)

    versionpath = ['d:','/home/tianshuaihua/wang']
    s = Tool_localization.statement_sus(formula, case_num, statement_num, covMatrix, vector, versionpath)
    for sus in s:
        tmp = sorted(s[sus].items(), key=lambda d: d[1], reverse=True)
        s[sus] = tmp

    print("s")
    # versionPath = "D:\\CC\\1_back"
    # # 四个测试用例 0 2 3通过 1未通过
    # inVector = [0, 1, 0, 1]
    # covMatrix = [
    #     [1, 0, 1, 1, 0],
    #     [0, 1, 1, 1, 0],
    #     [1, 1, 0, 0, 1],
    #     [0, 0, 1, 1, 1],
    # ]
    # statement_num = len(covMatrix[0])
    # case_num = len(covMatrix)
    # formulaSus, ochiai = Tool_localization.statement_sus(case_num, statement_num, covMatrix, inVector)
    # print(formulaSus)
    # CR = featureExtract.getCoverageRatioFactor(versionPath, covMatrix, inVector, formulaSus)
    # print(CR)
    #
    # SF = featureExtract.getSimilarityFactor(versionPath, covMatrix, inVector, formulaSus)
    # print(SF)
    #
    # target = os.path.join(versionPath, "hugeTest.txt")
    # FM = featureExtract.getFaultMaskingFactor(versionPath, covMatrix, inVector, ochiai, target)
    # print(FM)