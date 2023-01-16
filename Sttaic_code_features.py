# coding=utf-8

import math
import os
import copy
# 使用时打开注释
import javalang
import lizard
import linecache
import pickle
import Static_code_machine
import pandas as pd
import numpy as np
import Sttaic_code_features
from sklearn.preprocessing import MinMaxScaler


# lizard计算程序静态特征 按函数划分
#from DealComplexity import getDeclareList, get_folder, getAllJavaFile

# 获取所有java文件
import Tool_io


def getAllJavaFile(sourcePath):
    FileList = []
    for root, d_names, f_names in os.walk(sourcePath):
        for fileName in f_names:
            if fileName.lower().endswith(".java"):
                originPath = os.path.join(root, fileName)
                filePath = originPath.replace(sourcePath, "")
                FileList.append(filePath)
    FileList.sort()
    return FileList

def getfiles(sourcePath):
    FileList = []
    for root, d_names, f_names in os.walk(sourcePath):
        for fileName in f_names:
            if fileName.lower().endswith(".java"):
                originPath = os.path.join(root, fileName)
                FileList.append(originPath)
    FileList.sort()
    return FileList


def cal_code(file_path):
    # 使用时打开注释
    res = lizard.analyze_file(file_path)
    return res.function_list


# 统计Halstead 操作数和操作符数量
def halstead_cal(codes):
    Ioc_blank = 0
    op, op_key = all_operators()
    operators = op_key
    operands = {}
    isAllowed = True
    for text in codes:
        # 统计空白行
        if text == '':
            Ioc_blank += 1
        else:
            # 统计操作符和操作数的数量
            if text.startswith("/*"):
                isAllowed = False
            if (not text.startswith("//")) and isAllowed == True:
                text_tmp = ' ' + text.split('//')[0]
                for key in operators.keys():
                    if key not in op:
                        count = text_tmp.count(' ' + key + ' ')
                        operators[key] = operators[key] + count
                        if count > 0:
                            text_tmp = text_tmp.replace(' ' + key + ' ', ' ')
                    else:
                        operators[key] = operators[key] + text_tmp.count(key)
                        text_tmp = text_tmp.replace(key, ' ')
                for key in text_tmp.split():
                    if key in operands:
                        operands[key] = operands[key] + 1
                    else:
                        if key != '':
                            operands[key] = 1

            if text.endswith("*/"):
                isAllowed = True
        # 计算操作符数量和操作符种类数量
        operators.pop(')')
        operators.pop(']')
        operators.pop('}')
        num_operators = 0
        num_unique_operators = 0
        for k, v in operators.items():
            num_operators += v
            if v != 0:
                num_unique_operators += 1
        # 计算操作数数量和操作数种类数量
        num_operands = 0
        num_unique_operands = 0
        for m, n in operands.items():
            num_operands += n
            if n != 0:
                num_unique_operands += 1
        h = halstead_fun(num_operators, num_operands, num_unique_operators, num_unique_operands)
        return h, Ioc_blank


# java 操作数&操作符
def all_operators():
    op = {'(': 0, ')': 0, '{': 0, '}': 0, '[': 0, ']': 0, ',': 0, '.': 0, ':': 0, '>': 0, '<': 0, '!': 0,
          '~': 0, '?': 0, '::': 0, '<:': 0, '>:': 0, '!:': 0, '&&': 0, '||': 0, '++': 0, '--': 0, '+': 0,
          '-': 0, '*': 0, '/': 0, '&': 0, '|': 0, '^': 0, '%': 0, '->': 0, '::': 0, '+:': 0, '-:': 0, '*:': 0, '/:': 0,
          '&:': 0, '|:': 0, '^:': 0, '%:': 0, '<<:': 0, '>>:': 0, '>>>:': 0, '@': 0, '...': 0, '==': 0, '=': 0}

    op_key = {'(': 0, ')': 0, '{': 0, '}': 0, '[': 0, ']': 0, ';': 0, ',': 0, '.': 0, ':': 0, '>': 0, '<': 0, '!': 0,
              '~': 0, '?': 0, '::': 0, '<:': 0, '>:': 0, '!:': 0, '&&': 0, '||': 0, '++': 0, '--': 0, '+': 0,
              '-': 0, '*': 0, '/': 0, '&': 0, '|': 0, '^': 0, '%': 0, '->': 0, '::': 0, '+:': 0, '-:': 0, '*:': 0,
              '/:': 0, '&:': 0, '|:': 0, '^:': 0, '%:': 0, '<<:': 0, '>>:': 0, '>>>:': 0, '@': 0, '...': 0, '==': 0,
              '=': 0, 'abstract': 0, 'assert': 0, 'boolean': 0, 'break': 0, 'byte': 0, 'case': 0, 'catch': 0, 'char': 0,
              'class': 0, 'const': 0, 'continue': 0, 'default': 0, 'do': 0, 'double': 0, 'else': 0, 'enum': 0,
              'extends': 0, 'final': 0, 'finally': 0, 'float': 0, 'for': 0, 'if': 0, 'goto': 0, 'implements': 0,
              'import': 0, 'instanceof': 0, 'int': 0, 'interface': 0, 'long': 0, 'native': 0, 'new': 0, 'package': 0,
              'private': 0, 'protected': 0, 'public': 0, 'return': 0, 'short': 0, 'static': 0, 'strictfp': 0,
              'super': 0, 'switch': 0, 'synchronized': 0, 'this': 0, 'throw': 0, 'throws': 0, 'transient': 0,
              'try': 0, 'void': 0, 'volatile': 0, 'while': 0, 'null': 0, 'Integer': 0, 'Long': 0,
              'String': 0, 'Double': 0, 'Float': 0}

    return op, op_key


# 计算Halstead复杂度
def halstead_fun(n1, n2, u1, u2):
    N = n1 + n2
    U = u1 + u2
    V = N * math.log(U, 2)
    D = (u1 * n2) / (2 * u2)
    E = D * V
    T = E / 18
    B = (E ** (2 / 3)) / 3000
    L = 1 / D
    h = {'h_N': N, 'h_V': V, 'h_D': D, 'h_E': E, 'h_B': B, 'h_T': T, 'h_L': L, 'h_n1': n1, 'h_n2': n2, 'h_u1': u1, 'h_u2': u2}
    return h


# 遍历项目所有java文件
def java_files(files, root_path):
    code_all_feature = []
    for file_path in files:
        tmp = str(file_path).split('/fun_change_name/')
        pkg = tmp[1].split('/')
        if len(pkg) == 1:
            pkg_name = pkg[0].split('.')[0]
        else:
            pkg_name = '.'.join(pkg)[:-5]
        code_file_feature = cal_file_complexity(file_path, pkg_name, root_path)
        code_all_feature = code_all_feature + code_file_feature
    return code_all_feature


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
    for key,value in file_var.items():
        for k,v in value.items():
            if k == 'var':
                all_var = all_var + v
            else:
                all_local_var = all_local_var + v
    return file_var, all_var, all_local_var


# 计算单文件复杂度
def cal_file_complexity(file_path, pkg_name, root_path):
    function_list = cal_code(file_path)
    code_file_feature = []

    # 统计变量（FCCI暂无）
    fd = open(file_path, "r", encoding="utf-8")  # 读取Java源代码
    var_lines = fd.readlines()
    var_content = ''.join(var_lines)
    # 使用时打开注释
    tree = javalang.parse.parse(var_content)  # 根据源代码解析出一颗抽象语法树
    analysisList(tree.children)
    # 统计变量数量
    file_var, all_var, all_local_var = calVariable()

    for function_index in function_list:
        code_info = {}
        tmp = function_index.__dict__
        codes = []
        for statement_index in range(tmp['start_line'], tmp['end_line'] + 1):
            text = linecache.getline(file_path, statement_index).strip()
            codes.append(text)
        h, Ioc_blank = halstead_cal(codes)

        # 以下特征FCCI暂无
        #code_info['parameters'] = len(tmp['full_parameters'])
        #code_info['functions'] = len(function_list)
        #code_info['token_count'] = tmp['token_count']
        #code_info['top_nesting_level'] = tmp['top_nesting_level']

        code_info['h_N'] = h['h_N']
        code_info['h_V'] = h['h_V']
        code_info['h_D'] = h['h_D']
        code_info['h_E'] = h['h_E']
        code_info['h_B'] = h['h_B']
        code_info['h_T'] = h['h_T']
        code_info['h_L'] = h['h_L']
        code_info['h_n1'] = h['h_n1']
        code_info['h_n2'] = h['h_n2']
        code_info['h_u1'] = h['h_u1']
        code_info['h_u2'] = h['h_u2']
        code_info['number_of_lines'] = tmp['end_line'] - tmp['start_line'] + 1
        code_info['loc_executable'] = tmp['nloc']
        code_info['loc_blank'] = Ioc_blank
        code_info['loc_comments'] = tmp['end_line'] - tmp['start_line'] + 1 - tmp['nloc'] - Ioc_blank
        code_info['loc_code_and_comment'] = tmp['end_line'] - tmp['start_line'] + 1 - Ioc_blank
        code_info['cyclomatic_complexity'] = tmp['cyclomatic_complexity']
        code_info['error_flag'] = 0

        # 统计变量数量（fcci暂无）
        # na_len = len(str(tmp['name']).split('::'))
        # na = str(tmp['name']).split('::')[na_len-1]
        # for k, v in file_var.items():
        #     for m,n in v.items():
        #         if m == na:
        #             code_info['local_var'] = n
        # code_info['all_var'] = all_var
        # code_info['all_local_var'] = all_local_var

        #函数是否包含了错误（暂时注释）
        error_path = os.path.join(root_path, 'faultLineSimpleIndex.in')
        #err_loc = str(file_path).split("\\source\\")[0] + '\\faultLineSimpleIndex.txt'
        f = open(error_path, 'rb')
        error_con = pickle.load(f)
        start = ''
        if error_path.__contains__('Chart'):
            start = 'source.'
        if error_path.__contains__('Mockito'):
            start = 'src.'
        if error_path.__contains__('Time'):
            start = 'src.main.java.'
        if error_path.__contains__('Math'):
            start = 'src.java.'
        if error_path.__contains__('Lang'):
            start = 'src.main.java.'
        if error_path.__contains__('Closure'):
            start = 'src.'
        if error_path.__contains__('Cli'):
            start = 'src.java.'
        if error_path.__contains__('Codec'):
            start = 'src.java.'
        if error_path.__contains__('Compress'):
            start = 'src.main.java.'
        if error_path.__contains__('Csv'):
            start = 'src.main.java.'
        if error_path.__contains__('Gson'):
            start = 'gson.src.main.java.'
        if error_path.__contains__('JacksonCore'):
            start = 'src.main.java.'
        if error_path.__contains__('JacksonDatabind'):
            start = 'src.main.java.'
        if error_path.__contains__('JacksonXml'):
            start = 'src.main.java.'
        if error_path.__contains__('Jsoup'):
            start = 'src.main.java.'
        if error_path.__contains__('JxPath'):
            start = 'src.java.'
        error_key = '/' + (start + pkg_name).replace('.', '/') + '.java'
        if error_con.__contains__(error_key):
            arr_err = error_con[error_key]
            for ae in arr_err:
                if tmp['start_line'] <= ae <= tmp['end_line']:
                    code_info['error_flag'] = 1

        # 拼接函数名称
        names = pkg_name.split('.')
        name_len = len(pkg_name.split('.'))
        py_names = str(tmp['name']).split('::')
        last_name = ''
        for index in range(len(py_names)):
            if index == 0:
                continue
            else:
                last_name = last_name + '.' + py_names[index]
        if names[name_len - 1] == str(tmp['name']).split('::')[0]:
            code_info['name'] = pkg_name + last_name
        else:
            names[name_len - 1] = str(tmp['name']).split('::')[0]
            pkg_name = '.'.join(names)
            code_info['name'] = pkg_name + last_name
        if code_info['name'] != 'org.jfree.chart.axis.SegmentedTimeline.while':
            code_file_feature.append(code_info)
    return code_file_feature

def getSourcePath(formatCodePath, programName, versionInt):
    if not formatCodePath.lower().endswith(".java"):
        fileline=os.path.join(formatCodePath,"src")+"/"
    else:
        fileline=formatCodePath
    key=programName
    id=int(versionInt)
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


# 处理特征数据(整理成二维数组形式)
def deal_features_data(data):
    features_arr = []
    fun_name = []
    fun_res = []
    for features in data:
        arr = []
        fun_name.append(features['name'])
        fun_res.append(features['error_flag'])
        for k, v in features.items():
            if k != 'name' and k != 'error_flag':
                arr.append(v)
        features_arr.append(arr)
    return features_arr, fun_name, fun_res


# 递归遍历文件夹
def show_files(base_path, all_files=[]):
    file_list = os.listdir(base_path)
    # 准备循环判断每个元素是否是文件夹还是文件，是文件的话，把名称传入list，是文件夹的话，递归
    for file in file_list:
        # 利用os.path.join()方法取得路径全名，并存入cur_path变量，否则每次只能遍历一层目录
        cur_path = os.path.join(base_path, file)
        # 判断是否是文件夹
        if os.path.isdir(cur_path):
            show_files(cur_path, all_files)
        else:
            if not file.endswith('.java'):
                continue
            else:
                all_files.append(cur_path)
    return all_files


# 读取复杂度文件
def comp(comp_path):
    file_comp = open(comp_path, 'r', encoding='utf-8')
    list_stat = file_comp.readlines()
    complexity = {}
    name = []
    for index_row in range(len(list_stat)):
        row = str(list_stat[index_row]).strip().split('-')
        name.append(row[0])
        complexity[row[0]] = row[2] + '-' + row[3]
        # if complexity.__contains__(row[0]):
        #     res = complexity[row[0]]
        #     res = res + 1
        #     complexity[row[0]] = res
        # else:
        #     complexity[row[0]] = 0
    return complexity, name


# 生成静态特征向量
def static_code(code_path,fea,da_path):
    # path = all_path + '\\' + dir + '\\source\\fun_change_name'
    files = getAllJavaFile(code_path)
    matrix = []
    for file in files:
        java_f = code_path+file
        with open(java_f,'r',encoding='utf-8') as f:
            lines = f.readlines()
            matrix_tmp = [0] * len(lines)
            fun_list = cal_code(java_f)
            for fun_index in fun_list:
                tmp = fun_index.__dict__
                start = tmp['start_line'] - 1
                end = tmp['end_line']

                names = java_f[:-5].split('/')
                name_len = len(java_f[:-5].split('/'))
                py_names = str(tmp['name']).split('::')
                last_name = ''
                true_name = ''
                for index in range(len(py_names)):
                    if index == 0:
                        continue
                    else:
                        last_name = last_name + '.' + py_names[index]
                if names[name_len - 1] == str(tmp['name']).split('::')[0]:
                    true_name = java_f[:-5] + last_name
                else:
                    names[name_len - 1] = str(tmp['name']).split('::')[0]
                    true_name = '/'.join(names) + last_name

                key_name = true_name.split('fun_change_name/')[1].replace('/','.')
                #if key_name == ''
                if key_name != 'org.jfree.chart.axis.SegmentedTimeline.while':
                    if fea.__contains__(key_name):
                        matrix_tmp[start:end] = (end - start) * [fea[key_name]]
        matrix = matrix+matrix_tmp

    con = []
    delestat = Tool_io.checkAndLoad(da_path, "data_saveAgain_del_statement_index")
    for list_index in range(len(matrix)):
        if delestat[list_index] == False:
            tmp = matrix[list_index]
            con.append(tmp)
    if os.path.exists(os.path.join(da_path,'static_fea.txt')):
        print('exist ', os.path.join(da_path,'static_fea.txt'))
    else:
        static_file = open(os.path.join(da_path,'static_fea.txt'), 'w', encoding='utf-8')
        static_file.write(str(con))
        static_file.close()


# 读取一个文件夹，返回里面所有的文件名
def get_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        return dirs
    return []


# 计算静态特征
def exection_fcci(source_path, comp_path, root, data_path):
    programs = get_folder(source_path)
    # 遍历不同程序
    for pro_name in programs:
        if pro_name != 'Closure':
            continue
        # 程序路径
        pro_path = os.path.join(source_path, pro_name)
        vers = get_folder(pro_path)
        # 遍历程序版本
        for ver_name in vers:
            if ver_name != '165b':
                continue
            # 判断该版本是否存在数据
            exist_path = os.path.join(os.path.join(data_path, pro_name), ver_name)
            data_static = Tool_io.checkAndLoad(exist_path, "data_static")
            if data_static != None:
                continue
            if not os.path.exists(os.path.join(exist_path,'normalization')):
                continue
            # 找到改名后的源码
            ver_path = os.path.join(pro_path, ver_name)
            ver_id = ver_name[0:len(ver_name) - 1]
            code_path = getSourcePath(ver_path, pro_name, ver_id)
            change_path = os.path.join(code_path, 'fun_change_name')
            # 计算静态特征
            files = getfiles(change_path)
            root_path = os.path.join(os.path.join(root,pro_name), ver_name)
            files_feature = java_files(files, root_path)
            comp_info_path = os.path.join(os.path.join(comp_path,pro_name),pro_name+'_'+ver_name+'.txt')
            complexity, name = comp(comp_info_path)
            print(ver_name)

            special = []
            for features in files_feature:
                name = features['name']
                if complexity.__contains__(name):
                    #print("s")
                    features['essential_complexity'] = int(str(complexity[name]).split('-')[0])
                    features['design_complexity'] = int(str(complexity[name]).split('-')[1])
                else:
                    special.append(name)
            result_end = []
            for res_features in files_feature:
                if not special.__contains__(res_features['name']):
                    result_end.append(res_features)
            Tool_io.checkAndSave(exist_path, "data_static", result_end)


# 划分测试集和数据集
def deal_pca(root, data_path, model_path, static_model,source_path,origin_path):
    pros = get_folder(data_path)
    # 遍历程序
    for pro_name in pros:
        if pro_name != 'Mockito':
            continue
        print(pro_name)
        pro_path = os.path.join(data_path,pro_name)
        #vers = get_folder(pro_path)
        all_pros = Tool_io.checkAndLoad(origin_path, 'program.in')
        vers = all_pros[pro_name]
        vers_res = leave_one(vers)
        #f = open(os.path.join(model_path, pro_name + '.td'), 'rb')
        #td = pickle.load(f)
        # 遍历程序版本
        for train_index in vers_res:
            test_dict = {}
            train_dict = {}
            for ver_name in train_index['test']:
                ver_path = os.path.join(pro_path, ver_name)
                test_feature = Tool_io.checkAndLoad(ver_path,'data_static')
                test_dict[ver_name] = test_feature
            for ver_name in train_index['train']:
                ver_path = os.path.join(pro_path, ver_name)
                if os.path.exists(os.path.join(ver_path,'normalization')):
                    root_path = os.path.join(os.path.join(root,pro_name),ver_name)
                    fault = []
                    fault_loc = Tool_io.checkAndLoad(root_path, "faultHuge.in")
                    if fault_loc is not None:
                        for jfile in fault_loc:
                            for index in fault_loc[jfile]:
                                fault.append(index)
                    if len(fault) > 0:
                        train_feature = Tool_io.checkAndLoad(ver_path, 'data_static')
                        train_dict[ver_name] = train_feature

            # 训练模型
            learning_method(train_dict, test_dict,static_model,pro_name,source_path,data_path)


# 机器学习
def learning_method(train_dict, test_dict, static_model, pro_name,source_path,data_path):
    # for ceshi in train_dict:
    #     for lst in train_dict[ceshi]:
    #         if len(lst) != 21:
    #             print("sss")
    for ver_one in test_dict:
        ver_name = ver_one
    data_train, f_res_train, f_name_train = deal_data(train_dict)
    model = Static_code_machine.ridge_sklearn(data_train, f_res_train, static_model, pro_name, ver_name)
    # for tdata in test_dict:
    #     ver_data = test_dict[tdata]
    #     print(tdata)
    #     # if tdata == '26b':
    #     #     print("s")
    #     fun_features, fun_name, fun_res = deal_features_data(ver_data)
    #     res_data = Static_code_machine.pca_sklearn(fun_features)
    #     predication = model.predict(res_data)
    #     fea_dict = {}
    #     for index in range(len(fun_name)):
    #         fea_dict[fun_name[index]] = predication[index]
    #     # 存储结果信息（函数计算的值->函数中的每一条语句值）
    #     ver_id = tdata[0:len(tdata) - 1]
    #     sou_path = os.path.join(os.path.join(source_path, pro_name), tdata)
    #     code_path = os.path.join(getSourcePath(sou_path, pro_name, ver_id),'fun_change_name')
    #     da_path = os.path.join(os.path.join(data_path,pro_name),tdata)
    #     static_code(code_path, fea_dict, da_path)
    #     print("s")

    for tdata in test_dict:
        ver_data = test_dict[tdata]
        # print(tdata)
        # if tdata != '165b':
        #     continue
        fun_features, fun_name, fun_res = deal_features_data(ver_data)
        res_data = Static_code_machine.pca_sklearn(fun_features)
        predication = model.predict(res_data)
        fea_dict = {}
        for index in range(len(fun_name)):
            fea_dict[fun_name[index]] = predication[index]
        # 存储结果信息（函数计算的值->函数中的每一条语句值）
        ver_id = tdata[0:len(tdata) - 1]
        sou_path = os.path.join(os.path.join(source_path, pro_name), tdata)
        code_path = os.path.join(getSourcePath(sou_path, pro_name, ver_id),'fun_change_name')
        da_path = os.path.join(os.path.join(data_path,pro_name),tdata)
        static_code(code_path, fea_dict, da_path)
        print(tdata)
    print("end")


# 处理准备数据
def deal_data(info):
    f_name = []
    f_res = []
    lst = list()
    for train in info:
        features_arr, fun_name, fun_res = deal_features_data(info[train])
        array = np.array(features_arr)
        lst.append(array)
        f_name = f_name + fun_name
        f_res = f_res + fun_res
    # 纵向拼接
    arrn_train = np.concatenate(lst, axis=0)
    # PCA
    data = Static_code_machine.pca_sklearn(arrn_train)
    return data, f_res, f_name


# 留一个为测试集，其余为训练集,验证集
def leave_one(ver_list):
    res = []
    for index in range(len(ver_list)):
        train_data = copy.deepcopy(ver_list)
        train_data.pop(index)
        test_data = list(set(ver_list).difference(train_data))
        data_dict = {}
        data_dict['train'] = train_data
        data_dict['test'] = test_data
        res.append(data_dict)
    return res

if __name__ == '__main__':

    root = '/home/tianshuaihua/base_dataset'
    source_path = '/home/tianshuaihua/fcci/code'
    comp_path = '/home/tianshuaihua/fcci/comp'
    data_path = '/home/tianshuaihua/tpydata'
    model_path = '/home/tianshuaihua/model'
    static_model = '/home/tianshuaihua/static_model'
    origin_path = '/home/tianshuaihua/origin'

    #s = Tool_io.checkAndLoad('/home/tianshuaihua/tpydata/Chart/14b', 'data_Coverage_InVector_saveAgain')

    # 计算静态特征
    #exection_fcci(source_path, comp_path, root,data_path)

    # 计算回归模型
    deal_pca(root, data_path,model_path,static_model,source_path,origin_path)


    # 以下暂时不用

    # print("s")
    # pa='/home/tianshuaihua/fcci/code/Chart/10b/source/fun_change_name'
    # files = show_files(pa)
    # files_feature = java_files(files)

                    # for index in files_feature:
                    #     if not index.__contains__('local_var'):
                    #         print(index)

    # complexity, name = comp('D:\\CC\\data_test\\Chart\\1b\\complexity.txt')
    #
    # for features in files_feature:
    #     name = features['name']
    #     if not name.find("fun"):
    #         print(name)
    #     if complexity.__contains__(name):
    #         features['essential_complexity'] = int(str(complexity[name]).split('-')[0])
    #         features['design_complexity'] = int(str(complexity[name]).split('-')[1])

                    # tt = {}
                    # for er in files1:
                    #     key = er['name']
                    #     tt[key] = 1
                    #     if key == 'org.jfree.chart.axis.SegmentedTimeline.while':
                    #         print("ex")
                    # # print(complexity['org.jfree.data.xy.YWithXInterval.equals'])
                    # for k,v in tt.items():
                    #     if complexity.__contains__(k):
                    #         complexity.pop(k)
                    #     else:
                    #         print(k)
                    # print("s")

                    # f = open("D:\\CC\\data_test\\error\\Chart\\1b\\faultLineSimpleContent.txt", 'rb')
                    # line = pickle.load(f)
                    # print("s")
                    # for sss in files_feature:
                    #     if sss['error_flag'] == 1:
                    #         print(sss)

    # features_arr, fun_name, fun_res = deal_features_data(files_feature)
    # array = np.array(features_arr)
    # pca_data = Static_code_machine.pca_sklearn(array)
    # res = Static_code_machine.ridge_sklearn(pca_data, fun_res)


    # fea_dict = {}
    # ss ={}
    # for index in range(len(fun_name)):
    #     fea_dict[fun_name[index]] = res[index]
                    # arr = fun_name[index].split('.')
                    # arr_len = len(arr)
                    # if ss.__contains__(arr[arr_len-1]):
                    #     tmp = ss[arr[arr_len-1]]
                    #     tmp = tmp + 1
                    #     ss[arr[arr_len - 1]] = tmp
                    # else:
                    #     ss[arr[arr_len - 1]] = 0
                    # if ss[arr[arr_len - 1]]>0:
                    #     print(fun_name[index])

    # dirs, folder = get_folder('D:\\CC\\data_test\\Chart')
    # for dir in dirs:
    #     static_code('D:\\CC\\data_test\\Chart',dir,fea_dict)
    #
    # print(fea_dict)


