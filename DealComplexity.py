#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import csv
import pickle
import shutil
# 使用时打开注释
import javalang

import Sttaic_code_features


# 读取一个文件夹，返回里面所有的文件名
def get_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        return dirs, files
    return []


def del_file(path):
    p = path
    if not os.listdir(path):
        print('目录为空！')
    else:
        for i in os.listdir(path):
            path_file = os.path.join(p, i)  # 取文件绝对路径
            path = path_file + '\\complexity.txt'
            if os.path.exists(path):
                os.remove(path_file + '\\complexity.txt')


# 整合文件夹下所有文件复杂度信息(idea插件生成的复杂度信息)
def deal_files_complexity(folder_path, result_path):
    dirs, folder = get_folder(folder_path)
    res = open(result_path, 'a+', encoding='utf-8')
    for file_name in folder:
        file_path = folder_path + '\\' + file_name
        with open(file_path, 'r', encoding='GBK') as f:
            reader = csv.reader(f)
            result = list(reader)
            for row in result[2:]:  # 去掉前两行，表头
                if len(row) != 0:
                    length = len(str(row[0]).split('.'))
                    arr = str(row[0]).split('.')
                    name = arr[length - 2] + "." + arr[length - 1].replace(' ', '')
                    res.write(name + '-' + row[1] + '-' + row[2] + '-' + row[3] + '-' + row[4] + '\n')
                else:
                    break


# 整合文件复杂度信息(idea插件生成的复杂度信息)
def deal_file_complexity(file_path, result_path):
    res = open(result_path, 'a+', encoding='utf-8')
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        result = list(reader)
        for row in result[2:]:  # 去掉前两行，表头
            if len(row) != 0:
                # length = len(str(row[0]).replace(', ...','').split('.'))
                # length = len(str(row[0]).split('('))[0]
                # if length == 1:
                #    res.write(row[0] + '-0-0-0-0' + '\n')
                # else:
                #     tmp = str(row[0]).replace(', ...','')
                #     arr = tmp.split('.')
                name = str(row[0]).split('(')[0]
                if row[4] != 'n/a':
                    res.write(name + '-' + row[1] + '-' + row[2] + '-' + row[3] + '-' + row[4] + '\n')
            else:
                break
    res.close()

def deal_folders(folder_path):
    dirs, folders = get_folder(folder_path)
    for file_name in dirs:
        file_path = folder_path + '\\' + file_name + '\\comp.csv'
        result_path = folder_path + '\\' + file_name + '\\complexity.txt'
        deal_file_complexity(file_path, result_path)


# 函数改名
def fun_change_file(data_location):
    dirs, files = get_folder(data_location)
    for dir in dirs:
        source_path = data_location + '\\' + dir + '\\source'
        target_path = source_path + '\\fun_change_name'
        if os.path.exists(target_path):
            continue
        else:
            shutil.copytree(source_path, target_path)


# 函数改名 删除文件夹
def fun_del_file(data_location):
    dirs, files = get_folder(data_location)
    for dir in dirs:
        source_path = data_location + '\\' + dir + '\\source'
        target_path = source_path + '\\fun_change_name'
        if os.path.exists(target_path):
            shutil.rmtree(target_path)


# 获取所有java文件
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


ClassDeclareList = {}
FunctionDeclareList = {}
fun_name = 10000


# 分析代码ast树
def getDeclareList(tree):
    global ClassDeclareList
    global FunctionDeclareList
    if type(tree) == list:
        for item in tree:
            if item != None:
                getDeclareList(item)
    else:
        if ("ConstructorDeclaration" in str(type(tree)) or "MethodDeclaration" in str(type(tree))):
            functionName = tree.name + ":" + str(tree.parameters)
            lineNum = tree.position.line - 1
            FunctionDeclareList[functionName] = tree
        if ("InterfaceDeclaration" in str(type(tree)) or "ClassDeclaration" in str(type(tree))):
            ClassName = tree.name
            lineNum = tree.position.line - 1
            ClassDeclareList[ClassName] = tree
        if hasattr(tree, 'body'):
            childList = tree.body
            if "BlockStatement" not in str(type(childList)):
                getDeclareList(childList)
            else:
                childList = childList.children
                getDeclareList(childList)


# 函数改名
def chang_name():
    global fun_name
    global ClassDeclareList
    global FunctionDeclareList
    files = Sttaic_code_features.show_files('D:\\CC\\data_test\\Chart\\1b\\source\\fun_change_name\\org')
    for java_file in files:
        if java_file == 'D:\\CC\\data_test\\Chart\\1b\\source\\fun_change_name\\org\\jfree\\chart\\renderer\\LookupPaintScale.java':
            print("e")
        fd = open(java_file, "r", encoding="utf-8")  # 读取Java源代码
        lines = fd.readlines()
        content = ''.join(lines)
        tree = javalang.parse.parse(content)  # 根据源代码解析出一颗抽象语法树
        getDeclareList(tree.children)
        for k, v in FunctionDeclareList.items():
            old_fun_name = str(k).split(':')[0]
            fun_loc = v.position.line
            lines[fun_loc-1] = lines[fun_loc-1].replace(old_fun_name, 'fun' + str(fun_name))
            fun_name = fun_name + 1
        con_new = ''.join(lines)
        fd.close()
        ClassDeclareList = {}
        FunctionDeclareList = {}
        fd_w = open(java_file, "w", encoding="utf-8")
        fd_w.write(con_new)
        fd_w.close()
    print("s")


# 生成hugeCode2
def huge_code2(all_path):
    dirs, folder = get_folder(all_path)
    for dir in dirs:
        path = all_path + '\\' + dir + '\\source\\fun_change_name'
        files = getAllJavaFile(path)
        createHugeSourceFile(path, files, all_path + '\\' + dir + '\\hugeCode2.txt')


# 生成hugeCode2
def createHugeSourceFile(sourcePath,FileList,hugeFilePath):
    newContent=[]
    for item in FileList:
        fileName=sourcePath+item
        with open(fileName) as f:
            lines=f.readlines()
            for i in range(len(lines)):
                if not lines[i].endswith("\n") :
                    lines[i]=lines[i]+"\n"
            newContent.extend(lines)
    with open(hugeFilePath,"w") as f:
        f.writelines(newContent)

# 删除codehuge2
def del_huge_code2(path):
    p = path
    if not os.listdir(path):
        print('目录为空！')
    else:
        for i in os.listdir(path):
            path_file = os.path.join(p, i)  # 取文件绝对路径
            path = path_file + '\\hugeCode2.txt'
            if os.path.exists(path):
                os.remove(path_file + '\\hugeCode2.txt')


def test():
    global fun_name
    global ClassDeclareList
    global FunctionDeclareList
    files = Sttaic_code_features.show_files('D:\\CC\\data_test\\Chart\\1b\\source\\fun_change_name\\org')
    for java_file in files:
        fd = open('D:\\CC\\data_test\\Chart\\1b\\source\\fun_change_name\\org\\jfree\\chart\\renderer\\LookupPaintScale.java', "r", encoding="utf-8")  # 读取Java源代码
        lines = fd.readlines()
        content = ''.join(lines)
        tree = javalang.parse.parse(content)  # 根据源代码解析出一颗抽象语法树
        getDeclareList(tree.children)
        for k, v in FunctionDeclareList.items():
            old_fun_name = str(k).split(':')[0]
            fun_loc = v.position.line
            lines[fun_loc - 1] = lines[fun_loc - 1].replace(old_fun_name, 'fun' + str(fun_name))
            fun_name = fun_name + 1
        con_new = ''.join(lines)
        fd.close()
        ClassDeclareList = {}
        FunctionDeclareList = {}
        fd_w = open(java_file, "w", encoding="utf-8")
        fd_w.write(con_new)
        fd_w.close()
    print("s")


if __name__ == '__main__':
    # 复制源代码到新文件夹fun_change_name
    fun_change_file('D:\\CC\\data_test\\Chart')
    # 删除文件夹fun_change_name
    #fun_del_file('D:\\CC\\data_test\\data\\Chart')

    # 函数改名
    chang_name()

    # 生成复杂度txt文件
    #deal_folders('D:\\CC\\data_test\\Chart')
    # 删除复杂度txt文件
    #del_file('D:\\CC\\data_test\\Chart')


    # 生成hugeCode2
   #  huge_code2('D:\\CC\\data_test\\Chart')
    # 删除hugeCode2
    # del_huge_code2('D:\\CC\\data_test\\Chart')
    #test()


