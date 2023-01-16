import linecache
import math
import os

import sklearn
from scipy.spatial.distance import pdist, squareform
#import lizard
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix
from tqdm import trange
import javalang
import pickle
import tqdm


# 创建一个希伯特矩阵(高度病态，任何一个元素的点发生变动，整个矩阵的行列式的值和逆矩阵都会发生巨大变化)sss
# 这里的加法运算类似于矩阵相乘
import Tool_io
from FuzzyKnn import fuzzy_knn_metric
from Tool_io import create_data_folder


def one():
    X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
    y = np.ones(10)

    # 计算路径
    n_alphas = 200
    alphas = np.logspace(-10, -2, n_alphas)
    clf = linear_model.Ridge(fit_intercept=False)
    print(clf)

    coefs = []
    for a in alphas:
        clf.set_params(alpha=a)
        clf.fit(X, y)
        coefs.append(clf.coef_)
    # 图形展示
    # 设置刻度
    ax = plt.gca()
    # 设置刻度的映射
    ax.plot(alphas, coefs)
    # 设置x轴的刻度显示方式
    ax.set_xscale('log')
    # 翻转x轴
    ax.set_xlim(ax.get_xlim()[::-1])
    # 设置x、y标签以及标题
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Ridge coefficients as a function of the regularization')
    # 使得坐标轴最大值和最小值与数据保持一致
    plt.axis('tight')
    plt.show()


def two():
    import numpy as np
    n_samples, n_features = 10, 5
    np.random.seed(0)  # seed( ) 用于指定随机数生成时所用算法开始的整数值
    y = np.random.randn(n_samples)  # randn函数返回一个或一组样本，具有标准正态分布。
    X = np.random.randn(n_samples, n_features)
    clf = Ridge(alpha=1.0)
    # Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,normalize=False, random_state=None, solver='auto', tol=0.001
    clf.fit(X, y)
    pre = clf.predict(X)
    print(clf.coef_)
    print(clf.intercept_)
    print(pre)


ClassDeclareList = {}


# 统计类
def analysisList(tree):
    global ClassDeclareList
    global FunctionDeclareList
    if type(tree) == list:
        for item in tree:
            if item != None:
                analysisList(item)
    else:
        if ("InterfaceDeclaration" in str(type(tree)) or "ClassDeclaration" in str(type(tree))):
            ClassName = tree.name
            ClassDeclareList[ClassName] = tree
        if hasattr(tree, 'body'):
            childList = tree.body
            if "BlockStatement" not in str(type(childList)):
                analysisList(childList)
            else:
                childList = childList.children
                analysisList(childList)


# 统计函数变量数量
def calVariable():
    global ClassDeclareList
    file_var = {}
    for k, v in ClassDeclareList.items():
        class_var = {}
        for method in v.body:
            if 'FieldDeclaration' in str(type(method)):
                class_var['var'] = len(method.declarators)
            if "ConstructorDeclaration" in str(type(method)) or 'MethodDeclaration' in str(type(method)):
                var_count = 0
                if method.body != None:
                    for variable in method.body:
                        if 'LocalVariableDeclaration' in str(type(variable)):
                            var_count = var_count + 1
                class_var[method.name] = var_count
        if not class_var.__contains__('var'):
            class_var['var'] = 0
        file_var[k] = class_var
    # 统计文件内全部 全局变量和局部变量
    all_var = 0
    all_local_var = 0
    for key, value in file_var.items():
        for k, v in value.items():
            if k == 'var':
                all_var = all_var + v
            else:
                all_local_var = all_local_var + v
    return file_var, all_var, all_local_var


# 计算程序静态特征 按函数划分
# def Cal_code():
#     s = 'D:\\CC\\data_test\\Chart\\1b\\source\\org\\jfree\\chart\\axis\\CyclicNumberAxis.java'
#     d = 'D:\\CC\\data_test\\calculate\\src\\main\\java\\org\\jfree\\chart\\renderer\\xy\\XYBarRenderer.java'
#     res = lizard.analyze_file(s)
#     for w in res.function_list:
#         tmp = w.__dict__
#         print(tmp['long_name'])
#     # print(res.CCN)
#     # print(res.average_cyclomatic_complexity)
#     # print(res.nloc)
#     print(res.function_list)


# def cal_code(file_path):
#     res = lizard.analyze_file(file_path)
#     return res.function_list


# 读取一个文件夹，返回里面所有的文件名
def get_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        return dirs
    return []


from multiprocessing import Pool
import time
from time import sleep


def f(i):
    print(i[0])
    print(i[1])
    sleep(2)

# 读取一个文件夹，返回里面所有的文件名
def get_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        return files
    return []

def min_max_deal(data_dict):
    arr = []
    for k in data_dict:
        arr.append(data_dict[k])
    res = []
    for x in arr:
        if np.max(arr)-np.min(arr) != 0:
            x = float(x - np.min(arr)) / (np.max(arr) - np.min(arr))
            res.append(x)
        else:
            res.append(0)
    return res

def del_fail(origin, cc, fail_test):
    res_origin = [origin[i] for i in range(len(origin)) if (i not in fail_test)]
    res_cc = [cc[i] for i in range(len(cc)) if (i not in fail_test)]
    return res_origin, res_cc


if __name__ == "__main__":


    data = '/home/tianshuaihua/tpydata/Chart/1b'
    data2 = '/home/tianshuaihua/base_dataset/Chart/1b'
    s = Tool_io.checkAndLoad(data, 'data_Coverage_InVector_saveAgain')
    s2 = Tool_io.checkAndLoad(data2, 'faultHuge.in')

    print("s")




    data = '/home/tianshuaihua/tpydata'
    #data = '/home/tianshuaihua/base_dataset'
    dirs = get_folder(data)
    for dir in dirs:
        temp = os.path.join(data,dir)
        vers = get_folder(temp)
        all = 0
        count = 0
        num = 0
        loc = 0
        tests = 0
        sel = ['28b', '123b', '12b', '99b', '31b', '42b', '51b', '26b', '86b', '69b', '94b', '93b', '6b', '117b', '19b', '68b', '82b', '167b', '66b', '147b', '104b', '2b', '25b', '74b', '14b', '67b', '80b', '139b', '176b', '122b', '38b', '85b', '1b', '96b', '106b', '98b', '8b', '149b', '65b', '7b', '75b', '148b', '112b', '18b', '156b', '45b', '43b', '135b', '174b', '131b', '55b', '73b', '87b', '83b', '161b', '150b', '57b', '153b', '4b', '10b', '5b', '59b', '78b', '95b', '145b', '16b', '79b', '52b', '168b', '58b', '29b', '154b', '172b', '53b', '138b', '44b', '162b', '9b', '166b', '159b', '60b', '40b', '129b', '118b', '126b', '49b', '105b', '125b', '21b', '158b', '71b', '137b', '92b', '20b', '109b', '64b', '70b', '39b', '81b', '120b', '175b', '134b', '170b', '50b', '132b', '119b', '169b', '144b', '124b', '3b', '72b', '102b', '100b', '163b', '114b', '91b', '130b', '46b', '36b', '152b', '27b', '133b', '151b', '111b', '37b', '22b', '62b', '101b', '115b', '89b', '48b', '140b', '47b', '103b', '121b', '143b', '77b', '63b', '157b', '88b', '113b', '160b', '24b', '128b', '76b', '41b', '164b', '136b', '97b', '23b', '30b', '32b', '54b', '15b', '107b', '110b', '127b', '56b', '146b', '171b', '61b', '90b', '33b', '84b', '155b', '165b', '173b', '142b', '13b', '108b', '11b', '116b', '34b', '141b', '17b', '35b']

        if dir != 'Closure':
            continue
        li = []
        for ver in vers:
            if ver in sel:
                fol = os.path.join(temp,ver)
                file = os.path.join(fol, 'data_Coverage_InVector_saveAgain')
                if os.path.exists(file):
                    cov_res = Tool_io.checkAndLoad(fol, "data_Coverage_InVector_saveAgain")
                    if len(cov_res[7]) == 0 and len(cov_res[2] == 0):
                        print(ver)
                else:
                    li.append(ver)

        #     fol = os.path.join(temp, ver)
        #     fault = []
        #     fault_loc = Tool_io.checkAndLoad(fol, "faultHuge.in")
        #     if fault_loc is not None:
        #         for jfile in fault_loc:
        #             for index in fault_loc[jfile]:
        #                 fault.append(index)
        #     if len(fault)<=0:
        #         li.append(ver)
        print("end")
        print(li)

            # if os.path.exists(os.path.join(fol, 'data_Coverage_InVector_saveAgain')):
            #     res = Tool_io.checkAndLoad(fol, "data_Coverage_InVector_saveAgain")
            #     tests = tests + res[4] + res[5]
            #     num = num + 1

            # if os.path.exists(os.path.join(fol,'hugeCode.txt')):
            #     loc += len(open(os.path.join(fol,'hugeCode.txt')).readlines())
            #     all = all + 1

            # file = os.path.join(fol,'normalization')
            # if os.path.exists(file):
            #    count = count + 1

        #print(temp,count)
        #print(temp,all,loc/all)
        # if num!=0:
        #     print(temp,num,tests/num)

        # Tool_io.cal_res(data, pro_name, res_path)
    # a = [[0, 0, 0, 0, 1], [0, 1, 0, 0, 1]]
    # s = np.array(a)
    # for qw in s:
    #     print(qw)

    # op = {}
    # op['ochiai'] = [1, 2, 3, 4]
    # op['op2'] = [1, 2, 9, 4]
    # op['dstar'] = [5, 2, 3, 4]
    #
    # for index in op:
    #     print(index)
    # a = [0, 0, 0, 0, 1]
    # b = [0, 1, 0, 0, 1]
    # s = []
    # s.append(3)
    # s = s + b
    #
    # q = cmatrix = sklearn.metrics.classification_report(a, b)
    # res = confusion_matrix(a, b)
    # TP = res[1, 1]
    # TN = res[0, 0]
    # FP = res[0, 1]
    # FN = res[1, 0]
    # print(q)
    # print(res)
    # if TP + FN == 0:
    #     recall = 0
    # else:
    #     recall = TP / (TP + FN)
    # if TP + FP == 0:
    #     precision = 0
    # else:
    #     precision = TP / (TP + FP)
    # if FP + TN == 0:
    #     FPR = 0
    # else:
    #     FPR = FP / (FP + TN)
    # f1_score = 2 * TP / (2 * TP + FP + FN)
    # print(recall, precision, FPR, f1_score)
    # print("s")
    # a = {1: 'a', 2: 'b', 5: 'c', 6: 'd'}
    # b = {1: 'a', 2: 'b', 5: 'c', 6: 'd'}
    # # b = {1: 'm', 2: 'n', 3:'p',4:'q'}
    #
    # x = list(a.keys())
    # y = list(b.keys())
    #
    # for index in y:
    #     if index not in x:
    #         b.pop(index)
    # for index in x:
    #     if index in y:
    #         a.pop(index)
    # print(a, b)
    # print(x, y)
    # t1 = [[1,3,6,5],[4,5,6,7]]
    # t2 = [[2,5,45,74],[2,5,6,9]]
    # tmp = []
    # q = [1, 3, 6, 5]
    # w = [1, 3, 6, 5]
    # tmp = np.hstack((tmp, q))
    # tmp = np.hstack((tmp, w))
    # print(tmp)
    # path = '/home/tianshuaihua/pydata/Mockito/20b'
    # WPcc = Tool_io.checkAndLoad(path, "fuzzy_knn")
    # a = Tool_io.checkAndLoad(path, "data_Coverage_InVector_saveAgain")
    # versionPath = ['/home/tianshuaihua/test','/home/tianshuaihua/test']
    # fuzzy_knn_metric(a[6], WPcc, versionPath, error_pro_ver)
    # print("s")

    # loc = os.path.join('D:\CC\data_test\data', 'error.in')
    # if not os.path.exists(loc):
    #     os.makedirs(loc)
    # f_error = open(loc, 'rb+')
    # content = pickle.load(f)
    # pickle.dump('D:\CC\data_test' + " 缺少文件\r\n", f_error)
    # f_error.close()


    # ss = get_folder('D:\\CC\\data_test\\data')
    # res = create_data_folder('D:\\CC\\data_test\\data','D:\\CC\\data_test\\other')
    # print("s")

    # a = []
    # b = 'dsafa'
    # c ='dfsaasfew'
    # d = []
    # d.append(b)
    # d.append(c)
    # a.append(d)
    # a.append('qwerqwe')
    # print('main start')
    # for j in range(2):
    #     pool = Pool(processes=3)
    #     for i in range(3):
    #         if i > 30:
    #             print(j)
    #             pool.apply_async(f, (a,))
    #         else:
    #             continue
    #     pool.close()
    #     pool.join()
    #     sleep(5)
    #     print("s")
    # print('main running')
    # print('main end')
    #
    # f_error = open(os.path.join('D:\CC\data_test', 'result.txt'), 'a+')
    # f_error.writelines(" 缺少文件\r\n")
    # f_error.close()

    import csv

    # path = os.path.join('D:\CC\data_test', "sss") + '.csv'
    # print(path)
    # data也可以是dict
    # row = ['', 'recall', 'FPrate', 'precision', 'Fmeasure']
    # row2 = ['', 0.56484, 4444.2665, 0.0, 3.256]
    # with open('D:\CC\data_test\one.csv', 'w', newline='',encoding='utf-8') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(row)
    #     writer.writerow(row2)


    # with open(os.path.join('/home/tianshuaihua/dataset/Chart/8b', 'inVector.in'), "rb") as tests:
    #     allPass = 0
    #     allFiled = 0
    #     inVector = pickle.load(tests)
    #     for t_index in range(len(inVector)):
    #         if inVector[t_index] == 0:
    #             allPass = allPass + 1
    #         else:
    #             allFiled = allFiled + 1
    # print("s")

    # index = 6
    # if index < 15 or index > 0:
    #     print("s")
    #
    # # if not os.path.exists('D:\CC\data_test\error\Chart\\1b') or not os.path.exists('D:\CC\data_test\error\Chart\\98b'):
    # #     print("s")
    # a = []
    # f = open(os.path.join('D:\\CC\\data_test', 'n.txt'), 'wb')
    # pickle.dump(a, f)
    # f.close()
    #
    # f = open(os.path.join('D:\\CC\\data_test', 'n.txt'), 'rb')
    # error_con = pickle.load(f)
    # print(len(error_con))
    #
    #
    #
    # root = 'D:\\CC\\data_test\\data\\Chart\\1b'
    # formatCodePath = root
    # versionInt = int(os.path.basename(root.replace('b', '')))
    # key = os.path.basename(os.path.dirname(root))
    #
    # print(formatCodePath,versionInt,key)

    # arr = np.array([[1, 2, 3],
    #                 [4, 5, 6],
    #                 [7, 8, 9]])
    # for index in range(len(arr)):
    #     index_ = arr[index]
    #     print(np.max(index_)+1)
    # # 取出第一行
    # arr[0, :]
    # # 取出第一列
    # arr[:, 0]

    # for dir in dirs:
    #     dump_path = os.path.join(root, dir)
    #     folder.append(dump_path)
    #     pro_path = os.path.join(data, dir)
    #     data_folder.append(pro_path)
    #     if not os.path.exists(pro_path):
    #         os.mkdir(pro_path)
    #
    # all_files = []
    # data_all_files = []
    # for path in folder:
    #     files = Tool_io.get_folder(path)
    #     for file in files:
    #         version_path = os.path.join(path, file)
    #         all_files.append(version_path)
    #         pro_ver_path = os.path.join(data, dir)
    #         if not os.path.exists(pro_ver_path):
    #             os.mkdir(pro_ver_path)

    # import numpy as np
    #
    # arr = np.asarray([0, 10, 50, 80, 100])
    # for x in arr:
    #     x = float(x - arr.mean()) / arr.std()
    #     print(x)

    # import numpy as np
    #
    # v1 = np.mat([-1, 2, 3])
    # v2 = np.mat([1, 2, 6])
    # print((v1 - v2) * (v1 - v2).T)
    # res = np.sqrt((v1 - v2) * (v1 - v2).T)
    # print(res[0,0])

    # a = [1,0,0,1,0,1,1,0,1,1]
    # b = [0,0,0,1,1,0,1,0,1,1]
    # c = [1,0,0,1,1,0,1,0,1,1]
    #
    # d = [1,0,0,1,0,0,1,1,1,0]
    # e = [0,1,0,1,1,0,1,1,1,1]
    # f = [1,1,0,1,0,0,1,1,1,1]
    # g = [0,0,1,1,1,0,1,1,1,0]
    #
    #
    #
    # cov = []
    # cov.append(d)
    # cov.append(a)
    # cov.append(b)
    # cov.append(c)
    # cov.append(e)
    # cov.append(f)
    # cov.append(g)
    #
    # X = pdist(cov, 'euclidean')
    # print(X)
    #
    # all = []
    # all.append(a)
    # all.append(b)
    # all.append(c)
    # all = np.array(all)
    # stf = []
    # for col in range(np.shape(all)[1]):
    #     col_val = all[:, col]
    #     same_flag = 0
    #     for x in col_val:
    #         if x != 1:
    #             same_flag = 1
    #     if same_flag == 0:
    #         stf.append(col)
    # print(stf)
    #
    # index = [1,2,3]
    #
    # true_cc = []
    # for row_cov in range(len(cov)):
    #     if row_cov not in index:
    #         correct_flag = 0
    #         for stf_index in stf:
    #             if cov[row_cov][stf_index] != 1:
    #                 correct_flag = 1
    #         if correct_flag == 0:
    #             true_cc.append(row_cov)
    # print("s")

    # for i in range(len(index)):
    #     cov.pop(index[i]-i)

    # ds = 'D:\\CC\\data_test\\Chart\\1b\\source\\fun_change_name\\org\\jfree\\chart\\axis\\DateAxis.java'
    # file_path = 'D:\\CC\\code\\Main.java'
    #
    # function_list = cal_code(file_path)
    # for function_index in function_list:
    #     print("s")
    #
    # fd = open(file_path, "r", encoding="utf-8")  # 读取Java源代码
    # lines = fd.readlines()
    # content = ''.join(lines)
    # tree = javalang.parse.parse(content)  # 根据源代码解析出一颗抽象语法树
    # analysisList(tree.children)
    # file_var, all_var, all_local_var = calVariable()
    # print(file_var, all_var, all_local_var)

    # from sklearn.neighbors import NearestNeighbors
    # import numpy as np
    #
    # X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    # nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
    # distances, indices = nbrs.kneighbors(X)
    #
    # print(indices)
    #
    # print(distances)

    # X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    # for row_index in range(len(X)):
    #         for col_index in X[row_index]:
    #            print(col_index)

    # a = {
    #     3:"s",
    #     6:"q",
    #     8:"ww"
    # }
    # cc = []
    # for x in a:
    #     cc.append(x)
    # print(cc)
    #
    # s = [1,2,3]
    # if 3 not in s:
    #     print("haha")

    # a = '[1,0.0,20]'
    # a = a.replace('[','').replace(']','')
    # s = a.split(',')
    # print(s)
    # print(1/9)

    # Cal_code()
    # py_names = ['ss','dd','ee']
    # last_name = ''
    # for index in range(len(py_names)):
    #     if index == 0:
    #         continue
    #     else:
    #         last_name = last_name + '.' +  py_names[index]
    # print(last_name)

    # fd = open("D:\CC\code\jkl.java", "r", encoding="utf-8")  # 读取Java源代码
    # tree = javalang.parse.parse(fd.read())  # 根据源代码解析出一颗抽象语法树
    # #print(tree)
    # for i in range(0, len(tree.children)):
    #     print(tree.children[i])
    # for path, node in tree:
    #     print(path)

    # two()
    # h = {'h_N': 1}
    # print(h['h_N'])
    # s = []
    # s.append([1, 2, 3])
    # s.append([4, 5, 6])
    # s.append([7, 8, 9])
    #
    # temp = []
    #
    # for x in range(0,3):
    #     arr = []
    #     for y in range(0,3):
    #         arr.append(y)
    #     temp.append(arr)
    # q = np.array(temp)
    #
    # print(q.shape)

    # err_loc = 'D:\\CC\\data_test\\error\\Chart\\5b' + '\\failing_tests'
    # f = open(err_loc, 'rb')
    # f.read()
    # error_con = pickle.load(f)
    # print(error_con)
    # f.close()

    # err_loc = 'D:\\CC\\1_back\\faultHuge'
    #
    # f = open(err_loc, 'rb')
    # error_con = pickle.load(f)
    # print(error_con)
    # f.close()

    # dict = cov_dict[algorithm]
    # 将字典变成numpy数组
    # cov = []
    # for index in dict:
    #    cov.append(dict[index])
    # cov = np.array(cov)

    # for key_index in trange(len(cov)):
    #     if key_index not in failIndex:
    #         # 记录tp与其余测试用例之间的距离
    #         dis = {}
    #         # 计算最大距离
    #         max_dis = 0
    #         for other_key_index in range(len(cov)):
    #             if key_index != other_key_index:
    #                 res = vec_distance(cov[key_index], cov[other_key_index])
    #                 dis[other_key_index] = res
    #                 # 计算最大距离
    #                 if res > max_dis:
    #                     max_dis = res
    #         # 模糊加权公式 分子(numerator) / 分母(denominator)
    #         numerator = 0
    #         denominator = 0
    #         for dis_index in dis:
    #             func = 0
    #             if dis_index in failIndex:
    #                 func = 1
    #             numerator = numerator + ((max_dis - dis[dis_index]) * func)
    #             denominator = denominator + (max_dis - dis[dis_index])
    #         # 计算当前测试用例cc概率
    #         WPcc[key_index] = numerator / denominator

    # if not os.path.exists(os.path.join(versionPath[1], "faultHuge")):
    #     # 获取程序版本路径
    #     formatCodePath = versionPath[0]
    #     key = os.path.basename(os.path.dirname(versionPath[0]))
    #     versionInt = int(os.path.basename(versionPath[0].replace('b', '')))
    #     java_file_path = getSrcPath(formatCodePath, key, versionInt)
    #
    #     file_list = getJavaFile(java_file_path)
    #     err_loc = os.path.join(versionPath[0],'faultLineSimpleIndex.txt')
    #     f = open(err_loc, 'rb')
    #     error_con = pickle.load(f)
    #     err_location = 0
    #     ff = open(os.path.join(versionPath[1], 'faultHuge'), 'wb')
    #     err_con = {}
    #     for tmp_path in file_list:
    #         java_path = versionPath[1]+'\\source\\org'+ tmp_path
    #         error_key = ('\\source'+java_path.split("source")[1]).replace('\\','/')
    #         if error_con.__contains__(error_key):
    #             err_con[error_key] = err_location
    #         else:
    #             with open(java_path,'r',encoding='utf-8') as f:
    #                 err_location += len(f.readlines())
    #     pickle.dump(err_con, ff)
    #     ff.close()
