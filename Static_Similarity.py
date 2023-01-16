import os
import re

from tqdm import trange
from tree_sitter import Language, Parser
import nltk

def index_to_code_token(index,code):
    start_point=index[0]
    end_point=index[1]
    if start_point[0]==end_point[0]:
        s=code[start_point[0]][start_point[1]:end_point[1]]
    else:
        s=""
        s+=code[start_point[0]][start_point[1]:]
        for i in range(start_point[0]+1,end_point[0]):
            s+=code[i]
        s+=code[end_point[0]][:end_point[1]]
    return s

def tree_to_token_index(root_node):
    if (len(root_node.children)==0 or root_node.type=='string') and root_node.type!='comment':
        return [(root_node.start_point,root_node.end_point)]
    else:
        code_tokens=[]
        for child in root_node.children:
            code_tokens+=tree_to_token_index(child)
        return code_tokens

def check(str):
    my_re = re.compile(r'[A-Za-z]',re.S)
    res = re.findall(my_re,str)
    if len(res):
        return True
    else:
        return False

def cleanTokens(thisFailCodeTokens):
    newTokens=[]
    for item in thisFailCodeTokens:
        if check(item):
            if item.islower():
                newTokens.append(item)
            else:
                start=0
                for index in range(len(item)):
                    # print(item[index])
                    if index==len(item)-1:
                        newTokens.append(item[start:len(item)])
                    if item[index].isupper() and start!=index:
                        newTokens.append(item[start:index])
                        start=index
    newTokens = list(dict.fromkeys(newTokens))
    newnewTokens=[]
    for item in newTokens:
        if check(item):
            newnewTokens.append(item)
    return newnewTokens


def deal(rootPath,failedTest):
    JAVA_LANGUAGE = Language('/home/wuyonghao/Defeats4JExecutor/my-languages.so', 'java')
    parser = Parser()
    parser.set_language(JAVA_LANGUAGE)

    fileList=os.listdir(rootPath)
    fileNum=len(fileList)

    failTestContent=[]
    for failIndex in range(len(failedTest)):
        with open(os.path.join(rootPath,str(failedTest[failIndex])+".txt")) as f:
            thisFailTestContent=f.read()

            tree = parser.parse(bytes(thisFailTestContent,"utf8"))
            root_node = tree.root_node
            tokens_index = tree_to_token_index(root_node)
            code = thisFailTestContent.split('\n')
            thisFailCodeTokens = [index_to_code_token(x, code) for x in tokens_index]

            thisFailCodeTokens=cleanTokens(thisFailCodeTokens)

            failTestContent.append(thisFailCodeTokens)

    result=[]
    for index in trange(fileNum):
        if index not in failedTest:
            thisPassTestContent=""
            with open(os.path.join(rootPath, str(index) + ".txt")) as f:
                thisPassTestContent=f.read()

                tree = parser.parse(bytes(thisPassTestContent, 'utf8'))
                root_node = tree.root_node
                tokens_index = tree_to_token_index(root_node)
                code = thisPassTestContent.split('\n')
                thisPassCodeTokens = [index_to_code_token(x, code) for x in tokens_index]

                thisPassCodeTokens = cleanTokens(thisPassCodeTokens)
                tempSame=0
                for item in failTestContent:
                    sameElement=set(thisPassCodeTokens) & set(item)
                    sameLength=len(sameElement)
                    thisSame=sameLength/len(item)
                    if thisSame>tempSame:
                        tempSame=thisSame
                result.append(tempSame)
        else:
            result.append(1)
    return result


if __name__ == "__main__":
    # nltk.download('punkt')
    rootPath=r"D:\Defeats4JTest\output\Chart\1b\TestCaseContent"
    failedTest=[769,1739]
    deal(rootPath,failedTest)