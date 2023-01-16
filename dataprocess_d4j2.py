# -*- coding: UTF-8 -*-
import operator
import random
import json
import re
import pickle
import javalang
import configparser
OPsType=[]
proj='Lang'
dataset=[]
mutantors={}
mapp={'Lang':'lang'}
#Lang里面应该排除的确实是23；23没有错误方法，只能添加方法，其他
dict={
    'Lang': {0: 1, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 9, 8: 10, 9: 11, 10: 12, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 24,22: 25,23: 26, 24: 27, 25: 28, 26: 29, 27: 30, 28: 31, 29: 32, 30: 33, 31: 34, 32: 35, 33: 36, 34: 37, 35: 38, 36: 39, 37: 40, 38: 41, 39: 42, 40: 43, 41: 44, 42: 45, 43: 46, 44: 51, 45: 52, 46: 53, 47: 54, 48: 55, 49: 57, 50: 58, 51: 59, 52: 60, 53: 61, 54: 62, 55: 63, 56: 64, 57: 65}
}
nodesmax=0
linenodesmax=0

stTypes=[]
#传统的MBFL开始
MBFLs={}
#传统的MBFL结束
def S_Jaccard(totalfail,totalpass,Tf,Tp):
    sus=Tf*1.0/(Tp+totalfail)
    return sus

def S_Tarantula(totalfail,totalpass,Tf,Tp):
    sus=1.0*(Tf*totalpass)/(Tf*totalpass+Tp*totalfail)
    return sus

def S_Ochiai(totalfail,totalpass,Tf,Tp):
    sus=Tf*Tf*1.0/(totalfail*(Tf+Tp))
    return sus

def S_OP2(totalfail,totalpass,Tf,Tp):
    sus=Tf-Tp*1.0/(totalpass+1)
    return sus

S_Functions=[S_Jaccard,S_Tarantula,S_Ochiai,S_OP2]

def SBFL_information(proj,projversion):
    dfpath = '../d4j/' + proj + '/' + str( projversion ) + 'b/'
    f = open( dfpath + 'inVector.in', 'rb' )
    allTestRes = pickle.load( f )
    f.close()
    f = open( dfpath + 'CoverageMatrix.in', 'rb' )
    CoverageMatrix = pickle.load( f )
    f.close()
    #获得所有语句的测试覆盖信息
    Statements={}
    totalfail=0
    SUSs={}
    for i in range(len(CoverageMatrix)):
        t_st=CoverageMatrix[i]
        res=allTestRes[i]
        totalfail+=res
        for st in range(len(t_st)):
            if(t_st[st]==1):
                if st not in Statements:
                    Statements[st]={}
                    Statements[st]['Tf']=0
                    Statements[st]['Tp']=0
                if res==1:
                    Statements[st]['Tf']+=1
                else:
                    Statements[st]['Tp']+=1
    totalpass=len(allTestRes)-totalfail
    for st in Statements:
        SUSs[st]=[]
        for func in range(len(S_Functions)):
            SUSs[st].append(S_Functions[func](totalfail,totalpass,Statements[st]['Tf'],Statements[st]['Tp']))
    return SUSs
def M_Jaccard(akp,akf,anp,anf):
    sus= 0
    if (akf+anf+anp) > 0:
        sus=akf*1.0/(akf+anf+anp)
    return sus

def M_Tarantula(akp,akf,anp,anf):
    sus= 0
    if ((akf*1.0/(akf+anf))+(akp*1.0/(akp+anp)))> 0:
        sus = (akf * 1.0/(akf + akp))/((akf*1.0/(akf+anf))+(akp*1.0/(akp+anp)))
    return sus

def M_OP2(akp,akf,anp,anf):
    sus= akf-akp*1.0/(akp + anp+1)
    return sus


def M_Ochiai(akp,akf,anp,anf):
    sus=0
    if (akf+anf)*(akf+akp)>0:
        sus = akf*akf*1.0/((akf+anf)*(akf+akp))
    return sus

M_Fuctions=[M_Jaccard,M_Tarantula,M_Ochiai,M_OP2]


def judgeStatement(str):
    tokens = javalang.tokenizer.tokenize(str)
    parsr = javalang.parser.Parser(tokens)

    # LocalVariableDeclaration
    try:
        return parsr.parse_local_variable_declaration_statement().__class__.__name__
    except Exception:
        pass
    # Method    ConstructorDeclaration                   ThrowStatement ???
    try:
        if('.java' in str):
            return 'MethodDeclaration'
        return parsr.parse_member_declaration().__class__.__name__
    except Exception:
        pass

    # ReturnStatement   IfStatement   WhileStatement    BreakStatement  ContinueStatement          StatementExpression???
    try:
        return parsr.parse_statement().__class__.__name__
    except Exception:
        pass
    # control   ?
    try:
        return parsr.parse_for_control().__class__.__name__
        # return parsr.parse_for_var_control().__class__.__name__
    except Exception:
        pass

    # parameters   ?
    try:
        return parsr.parse_type_parameters().__class__.__name__
        # return parsr.parse_formal_parameters().__class__.__name__
    except Exception:
        pass

    return 'StatementExpression'

f = open(proj+'0.pkl', 'rb')
OldDataset = pickle.load(f)
f.close()
#63:[1864],62:[329],61:[1652],60:[1646],
#buchong={9:[2276],64:[1054],54:[357],52:[456],45:[703],44:[414],43:[1718],39:[504],38:[1908],23:[],13:[485,486],11:[440]}
#23是只有增添方法，没有错误方法。其他是未有对应的方法列表文件
#13都是遗漏错误
buchong={1: [1420], 2: [], 3: [1417], 4: [2091, 2092], 5: [456], 6: [2055], 7: [1408, 1415], 8: [2239,2253], 9: [2188], 10: [2201], 11: [449], 12: [449], 13: [486,487], 14: [523], 15: [1612,1623], 16: [1391], 17: [1941], 18: [2099], 19: [1958], 20: [553, 555], 21: [1984], 22: [1263], 23: [], 24: [1331], 25: [1934], 26: [2070], 27: [1270], 28: [1881], 29: [632], 30: [491, 493, 494, 495, 496, 499], 31: [494], 32: [711, 721, 722], 33: [350], 34: [882, 883], 35: [137, 147], 36: [1151, 1184], 37: [127], 38: [1909], 39: [505], 40: [421], 41: [290, 293], 42: [345], 43: [1719], 44: [415], 45: [704], 46: [462, 463, 464, 465, 466, 467], 47: [], 48: [], 49: [], 50: [], 51: [219], 52: [457], 53: [1888], 54: [358], 55: [1973], 56: [], 57: [359], 58: [1326], 59: [1608], 60: [1647, 1651], 61: [1653], 62: [330, 331], 63: [1866, 1868], 64: [1055], 65: [1841]}
for K in dict[proj]:
    # if K>=5:
    #     break
    projversion=dict[proj][K]
    dfpath = '../d4j/' + proj + '/' + str( projversion ) + 'b/'
    # print(dfpath)
    dataset.append({})
    dataset[K]['proj'] = proj + str(projversion)
    dataset[K]['ans'] = []
    # if(projversion!=46):
    #     continue
    #find all错误测试用例的索引
    f = open( dfpath + 'inVector.in', 'rb' )
    allTestRes = pickle.load( f )
    f.close()
    # 根据错误测试用例的索引，find错误测试用例的名字，并进行编号
    f = open( dfpath + 'all_tests.txt', 'rb' )
    all_tests = f.readlines()
    f.close()

    for x in OldDataset:
        if(x['proj']==dataset[K]['proj']):
            Base=x

    fTestIndexs={}
    ftest = {}
    for i in range(len(allTestRes)):
        if allTestRes[i]==1:
            testName =all_tests[i].decode().strip('\n')
            # ComN=testName.split('#')[1]
            # for comO in Base['ftest']:
            #     if comN in comO:
            if testName not in ftest:
                ftest[testName]=len(ftest)
                fTestIndexs[i]=testName

    # read index of fault method of proj
    f = open( dfpath + 'faultHuge_Function.in', 'rb' )
    #{'/src/java/org/apache/commons/lang/Entities.java': [329, 330, 330, 330]}
    tmp = list( pickle.load( f ).values() )  # 错误方法的索引
    f.close()
    ans=[]
    for i in buchong[projversion]:
        ans.append(i-1)
    print(ans)
    print(buchong[projversion])
    # ans = []  # 包含所有错误方法的index
    # for file_index in tmp:
    #     for m_index in file_index:
    #         if m_index not in ans:
    #             ans.append(m_index)
    # for a in buchong[projversion]:
    #     m_index=a-1
    #     if m_index not in ans:
    #         ans.append(m_index)
    # if len(ans)!=len(buchong[projversion]):
    #     print(projversion,ans,buchong[projversion],tmp)

    # 获取错误测试覆盖的方法的索引对应的方法名，并进行编号
    methods = {}
    f = open( dfpath + 'FunctionList.txt', 'rb' )
    FunctionList = f.readlines()
    f.close()
    correctnum = {}
    #根据错误测试用例，发现错误测试用例覆盖的方法的索引
    indexOfFunByFTest={}#被错误测试用例覆盖的方法的索引集
    f = open( dfpath + 'CoverageMatrix_Function.in', 'rb' )
    CoverageMatrix_Function = pickle.load( f )
    f.close()
    for i in fTestIndexs.keys():
        covf_i = CoverageMatrix_Function[i] #coverage informationn covf_i of test case i
        for j in range(len(covf_i)):
            if (covf_i[j] == 1):
                funName = FunctionList[j].decode().strip( '{\n' ).split('(')
                funNameP=""
                if(len(funName)>1):
                    funNameP=".".join(funName[1].replace(" ","").split(')')[:-1])
                funName = funName[0].split(' ')
                while '' in funName:
                    funName.remove('')
                funName = funName[0].split(':')[0] + "@" + funName[-1]+'.'+funNameP
                # 获取错误测试覆盖的方法的索引对应的方法名，并进行编号
                if(funName not in methods):
                    methods[funName] = len(methods)
                    indexOfFunByFTest[j]=funName
                    if j in ans:
                        dataset[K]['ans'].append(methods[funName])
                        correctnum[methods[funName]] = 4
    # print("methods",methods)
    # print(dataset[K]['ans'])
    if(len(ans)!=len(dataset[K]['ans'])):
        for j in ans:
            if j not in indexOfFunByFTest:
                funName = FunctionList[j].decode().strip( '{\n' ).split( '(' )
                # print(funName)
                funNameP = ""
                if (len( funName ) > 1):
                    funNameP = ".".join( funName[1].replace(" ","" ).split(')' )[:-1] )
                funName = funName[0].split( ' ' )
                while '' in funName:
                    funName.remove( '' )
                funName = funName[0].split( ':' )[0] + "@" + funName[-1] + '.' + funNameP
                if (funName not in methods):
                    methods[funName] = len( methods )
                    # print(funName)
                    indexOfFunByFTest[j] = funName
                    dataset[K]['ans'].append( methods[funName] )
                    correctnum[methods[funName]] = 4
        # if(len(ans)!=len(dataset[K]['ans'])):
        #     print(projversion, len( ans ), len( dataset[K]['ans'] ))

    SUSs_st=SBFL_information(proj,projversion)#语句的怀疑度公式
    indexOfStByFTest={}#被错误测试覆盖的语句的索引集
    #测试用例对语句的覆盖信息,获取失败测试用例覆盖的语句的索引
    f = open( dfpath + 'CoverageMatrix.in', 'rb' )
    CoverageMatrix = pickle.load(f)
    f.close()
    #每个语句索引对应的真正语句在HugeCode中的对应关系
    f = open( dfpath + 'HugeToFile.txt', 'rb')
    HugeToFile = f.readlines()
    f.close()
    #每个语句索引对应的真正语句
    f = open( dfpath + 'hugeCode.txt', 'rb' )
    HugeCode = f.readlines()
    f.close()
    lines={}##语句的编号：'org.apache.commons.lang3.math.NumberUtils:463': 8,
    edge=[]  # <语句，错误测试用例>
    ltype={}#<语句编号，类型>
    lSBFL={}
    lst={}
    edge2 = []  # <方法,语句> 方法而言，我们能得到方法的文件位置以及方法的名字， 语句我们能找到对应的文件位置，往上找发现它所属于方法
    lcorrectnum={}
    for fTIndex in fTestIndexs.keys():#coverage informationn covi of test case i
        cov_i=CoverageMatrix[fTIndex]
        for j in range(len(cov_i)):
            if(cov_i[j]==1):
                if(j not in indexOfStByFTest):
                    stIndex = j
                    while stIndex >= 0:
                        str_st = HugeCode[stIndex].decode().strip( '\n' )
                        if ' class ' in str_st and '{' in str_st:
                            stIndex=-1
                            break
                        # 从语句索引处往上找语句所属于的方法
                        #if str_st.startswith( '    ', 0, 4 ) and ')' in str_st and '(' in str_st and '{' in str_st and 'if' not in str_st and 'for' not in str_st and 'while' not in str_st and 'catch' not in str_st and 'switch' not in str_st:
                            # print(stIndex,str_st,"**"+str_st[:4]+"**
                        if '{' in str_st and '}' not in str_st:
                            stType = judgeStatement( str_st + '}' )
                        else:
                            stType = judgeStatement( str_st + ';' )
                        if stType in ['MethodDeclaration','ConstructorDeclaration']:
                            if stIndex==j:
                                stIndex = -1
                                # print("==")
                                break
                            str_st = str_st.split( '(' )
                            str_stP=""
                            if (len( str_st ) > 1):
                                str_stP = "".join( str_st[1].replace( " ", "" ).split( ')' )[:-1] )
                            str_st = str_st[0].split( ' ' )
                            while '' in str_st:
                                str_st.remove( '' )
                            while ' ' in str_st:
                                str_st.remove( ' ' )
                            for m in methods:
                                if str_st[-1] in m and str_stP in m:
                                    # print("find m in method：", projversion, j, stIndex, str_st)
                                    stInfor = HugeToFile[j].decode().strip( '\n' )
                                    stName = re.sub( '[\t]', '', stInfor )
                                    st = HugeCode[j].decode().strip( '\n' )
                                    #print("语句：",j, stName)
                                    if stName not in lines:
                                        lines[stName] = len(lines)
                                        edge2.append( (methods[m], lines[stName]) )
                                        lSBFL[lines[stName]]=SUSs_st[j]
                                    lcorrectnum[lines[stName]]=1
                                    indexOfStByFTest[j] = stName
                                    while(st[0]==' '):
                                        st=st.strip(' ')
                                    if '{' in st and '}' not in st:
                                        stType= judgeStatement( st + '}' )
                                    else:
                                        stType= judgeStatement(st+';')
                                    #ltype[lines[stName]] = st  # 现在跟着的是语句，后面换成13种类型
                                    ltype[lines[stName]] = stType  # 换成13种类型
                                    if stType not in stTypes:
                                        stTypes.append(stType)
                                    st=st.split(' ')
                                    while '' in st:
                                        st.remove( '' )
                                    while ' ' in st:
                                        st.remove( ' ' )
                                    lst[lines[stName]]=''.join(st) # 具体的语句
                                    break
                            break
                        stIndex -= 1
                    # if stIndex == -1:
                    #     print(stIndex, j, str_st)
                if(j in indexOfStByFTest):
                    edge.append((lines[indexOfStByFTest[j]],ftest[fTestIndexs[fTIndex]]))
    # print("lines",lines)
    # print('ltype',ltype)
    # print('lcorrectnum',lcorrectnum)
    # print(edge)
    # print(len(Base['lines']),len(Base['ftest']),len(Base['methods']))
    # print(len(lines),len(ftest),len(methods))
    rtest={}  #
    rTestIndexs={}
    edge10=[]  # <语句，正确测试用例>
    for rTIndex in range(len(CoverageMatrix)):
        for stindex in indexOfStByFTest:
            if((CoverageMatrix[rTIndex][stindex]==1) and (rTIndex not in fTestIndexs)):
                if(rTIndex not in rTestIndexs):
                    testName = all_tests[rTIndex].decode().strip( '\n' )
                    rtest[testName] = len(rtest)
                    # print(rTIndex,testName)
                    rTestIndexs[rTIndex] = testName
                edge10.append((lines[indexOfStByFTest[stindex]],rtest[rTestIndexs[rTIndex]]))
    # print("rtest",len(rtest))
    # print(edge10)
    #获得变异体的信息
    mutantsFP = '../d4j/' + proj + '_mutants/mutants/' + str(projversion) + '/'+proj+'-'+str(projversion)+'-'
    mutantsFP1 = '../d4j/' + proj + '_mutants/mutants/' + str( projversion ) + '/' + mapp[proj] + '-' + str(projversion ) + '-'
    killmapFP = '../d4j/' + proj + '_mutants/killmap/'+proj+'-' + str(projversion) + '-killmap'
    codeFP    = '../d4j/' + proj + '_code/'+proj+'-'+ str(projversion) + 'b'+'/src/main/java/'
    if (proj == 'Lang' and projversion>35):
        codeFP = '../d4j/' + proj + '_code/' + proj + '-' + str( projversion ) + 'b' + '/src/java/'

    # print(mutantsFP, killmapFP, codeFP)
    f = open( mutantsFP + 'mutants.log', 'rb' )
    mutantsInfors = f.readlines()
    f.close()
    mutation={}#变异体信息,这里没用变异体具体信息作为key，而是用的Lang-1-mutants.log的编号-1作为key，具体的变异体信息放在了inforOfMutation
    mtype={}  # <mutation,变异算子类型类型>
    inforOfMutation={}#<编号-1，具体变异体信息>
    edge12=[]  # <mutation,语句>
    edge13=[]  # <mutation,正确测试用例>————错误类型
    edge14=[]  # <mutation,错误测试用例>————修复
    # 获取传统MBFL的信息
    MBFL = {}
    new_d = {v: k for k, v in indexOfFunByFTest.items()}
    # 传统MBFL结束
    for i in range(len(mutantsInfors)):
        aMutantAllInfor_str=mutantsInfors[i].decode().strip( '\n' )
        aMutantAllInfor    =aMutantAllInfor_str.split(':')
        # print("aMutantAllInfor",aMutantAllInfor)
        # 4217:COR: & & (boolean, boolean):FALSE( boolean, boolean ):org.apache.commons.lang3.math.NumberUtils @ isNumber(
        #     java.lang.String ):1424:!allowSigns & & foundDigit |== > false
        # 4218:COR: & & (boolean, boolean):LHS( boolean, boolean ):org.apache.commons.lang3.math.NumberUtils @ isNumber(
        #     java.lang.String ):1424:!allowSigns & & foundDigit |== > !allowSigns
        # 4219:COR: & & (boolean, boolean):RHS( boolean, boolean ):org.apache.commons.lang3.math.NumberUtils @ isNumber(
        #     java.lang.String ):1424:!allowSigns & & foundDigit |== > foundDigit
        amutantFInfor=aMutantAllInfor[4].split('(')[0].split('@')
        # print(amutantFInfor)
        # print(aMutantAllInfor_str)

        amutantFP='/'.join( amutantFInfor[0].split( '.' ) ) + '.java'
        if(len(amutantFInfor)<2):
            continue
        amutantFInfor=amutantFP+'@'+amutantFInfor[1]
        MS=[]
        for m in methods:
            if(amutantFInfor in m):
                MS.append(m)
        if(len(MS)==0):
            continue
        mutationline=int(aMutantAllInfor[5])
        f = open( codeFP + amutantFP, 'rb' )
        # '../d4j/Lang_code/Lang-3b/src/main/java/org/apache/commons/lang3/StringUtils.java
        stInfors = f.readlines()
        f.close()
        st=stInfors[mutationline-1].decode().strip('\n')
        st = st.split( ' ' )
        while '' in st:
            st.remove( '' )
        while ' ' in st:
            st.remove( ' ' )
        st=''.join(st)
        # print("1:", st)
        for stkey in lst:
            if(st in lst[stkey] or lst[stkey] in st):
                flag=0
                for m in MS:
                    if(methods[m],stkey) in edge2:
                        flag=1
                        mu_index = len( mutation )
                        mutation[i] = mu_index
                        edge12.append( (mu_index, stkey))
                        ###MBFL 开始
                        MBFL[i]=[new_d[m],0,0,0,0]
                        # MBFL[i].append([])
                        # MBFL[i].append([])
                        # MBFL[i].append([])
                        # MBFL[i].append([])
                        ###MBFL 结束
                        inforOfMutation[i] = aMutantAllInfor_str
                        OPType = aMutantAllInfor[1]
                        OPindex = len(aMutantAllInfor[2])
                        if('(' in aMutantAllInfor[2]):
                            OPindex=aMutantAllInfor[2].index('(')
                        OPType+=('.'+aMutantAllInfor[2][:OPindex])
                        # print(aMutantAllInfor[2],aMutantAllInfor[2][:OPindex])
                        OPindex = len(aMutantAllInfor[3])
                        if ('(' in aMutantAllInfor[3]):
                            OPindex= aMutantAllInfor[3].index( '(' )
                        OPType +=('.'+aMutantAllInfor[3][:OPindex])
                        # print(aMutantAllInfor[3], aMutantAllInfor[3][:OPindex])
                        mtype[mu_index]=OPType
                        if(OPType not in OPsType):
                            OPsType.append(OPType)
                            # print(OPType)
                        break
                if flag==1:
                    break
    # print(mutation)
    # print(OPsType)
    # print(rtest)
    # print(edge12)
    f = open( mutantsFP1 + 'err.txt', 'rb' )
    testInforsInMu = f.readlines()
    f.close()
    f = open( killmapFP, 'rb' )
    killmap= f.readlines()
    f.close()
    i=-1
    #传统MBFL开始
    Test = {}  # 测试：索引
    # 获得失败测试用例的姓名：[索引，属性]
    for j in range( len( allTestRes ) ):
        if allTestRes[j] == 1:
            testName = all_tests[j].decode().strip( '\n' )
            Test[testName] = [j, 'failed']
        else:
            testName = all_tests[j].decode().strip( '\n' )
            Test[testName] = [j, 'passed']
    print(proj+str(projversion))
    #传统MBFL结束
    for k in range(len(testInforsInMu)):
        atestInfor=testInforsInMu[k].decode().strip( '\n' ).strip( ']' )
        if('[starting' not in atestInfor):
            continue
        i += 1
        atestName=atestInfor.split(': ')[-1]
        # 传统MBFL开始#这里问题在于一些测试用例丢失，问题正在解决
        if (atestName not in Test):
            print(proj + str( projversion ), ":", atestName)
            continue
        # 传统MBFL结束
        killLine = killmap[i].decode().strip( '\n' )
        if (atestName in rtest):
            # print(atestName)
            j=0
            while j*2 < len(killLine):
                if(killLine[j*2]=='1' and j in mutation):
                    # print(mutation[j],rtest[atestName])
                    edge13.append((mutation[j],rtest[atestName]))
                j+=1
        elif(atestName in ftest):
            j = 0
            while j*2 < len(killLine):
                if (killLine[j*2] == '1' and j in mutation):
                    # print(mutation[j], ftest[atestName])
                    edge14.append((mutation[j], ftest[atestName]))
                j += 1
        # 传统MBFL 开始
        if(Test[atestName][1]=='passed'):
            j = 0
            while j * 2 < len( killLine ):
                if(j in mutation):
                    if (killLine[j * 2] == '1'):
                        #MBFL[j][1].append( Test[atestName][0] )
                        MBFL[j][1]+=1
                    else:
                        #MBFL[j][3].append( Test[atestName][0] )
                        MBFL[j][3]+=1
                j += 1
        else:
            j = 0
            while j * 2 < len( killLine ):
                if (j in mutation):
                    if (killLine[j * 2] == '1'):
                        #MBFL[j][2].append( Test[atestName][0] )
                        MBFL[j][2]+=1
                    else:
                        #MBFL[j][4].append( Test[atestName][0] )
                        MBFL[j][4]+=1
                j += 1
        # 传统MBFL 结束
    mMBFL={}
    for M in MBFL:
        m_index=mutation[M]
        mMBFL[m_index]=[]
        for func in range(len(M_Fuctions)):
            mMBFL[m_index].append(M_Fuctions[func](MBFL[M][1],MBFL[M][2],MBFL[M][3],MBFL[M][4]))

    # 传统的MBFL开始
    MBFLs[projversion]=MBFL
    # 传统的MBFL结束
    dataset[K]['methods']=methods
    dataset[K]['ftest'] = ftest
    dataset[K]['rtest'] = rtest
    dataset[K]['lines'] = lines
    dataset[K]['ltype'] = ltype
    dataset[K]['edge'] = edge
    dataset[K]['edge10'] = edge10
    dataset[K]['edge2'] = edge2
    dataset[K]['mutation'] = mutation
    dataset[K]['mtype'] = mtype
    dataset[K]['edge12'] = edge12
    dataset[K]['edge13'] = edge13
    dataset[K]['edge14'] = edge14
    dataset[K]['lcorrectnum']=lcorrectnum
    dataset[K]['correctnum']=correctnum
    dataset[K]['lSBFL'] = lSBFL
    dataset[K]['mMBFL'] = mMBFL
    if(len(methods)+len(ftest)+len(rtest)>nodesmax):
        nodesmax=len(methods)+len(ftest)+len(rtest)
    if(len(lines)+len(mutation)>linenodesmax):
        linenodesmax=len(lines)+len(mutation)
    print(str(projversion)+':',dataset[K]['ans'],len(ftest),len(rtest),len(methods),len(lines))
    print(str(projversion) + ':', Base['ans'],len(Base['ftest'] ),len( Base['rtest'] ),len( Base['methods'] ),len( Base['lines']))

filename = proj+".json"
with open(filename, 'w') as file_obj:
    json.dump(dataset, file_obj)
filename = proj+"_MBFL.json"
with open(filename, 'w') as file_obj:
    json.dump(MBFLs, file_obj)
print("nodesmax",nodesmax)
print("linenodesmax",linenodesmax)

f = open(proj+'.pkl','wb' )
pickle.dump(dataset, f )
f.close()

f = open(proj+'.pkl', 'rb')
obj = pickle.load(f)
f.close()
for stType in stTypes:
    print(stType)
print(len(stTypes))
for OPType in OPsType:
    print(OPType)
# 1: [1] 1 31 3 19
# 1: [1] 1 31 3 19
# 3: [1] 1 30 4 37
# 3: [0] 1 30 4 37
# 4: [0, 1] 1 30 2 20
# 4: [1, 0] 1 40 3 23
# 5: [0, 3] 1 12 4 9
# 5: [0] 1 12 4 7
# 6: [6] 1 52 22 74
# 6: [2] 1 43 5 28
# 7: [10] 1 490 19 99
# 7: [0] 1 261 13 90
# 8: [32] 2 128 37 177
# 8: [14] 2 127 57 279
# 9: [26] 2 120 27 147
# 9: [0] 2 100 32 171
# 10: [7] 2 118 29 157
# 10: [16] 2 98 39 224
# 11: [2] 1 4 3 10
# 11: [0] 1 4 2 10
# 12: [3] 2 2 5 16
# 12: [1] 2 2 5 16
# 13: [4, 5] 1 24 6 29
# 13: [4, 2] 1 24 6 34
# 14: [1] 1 16 2 4
# 14: [0] 1 16 1 4
# 15: [18, 23] 2 361 51 251
# 15: [5, 14] 2 79 26 174
# 16: [2] 1 187 9 63
# 16: [0] 1 156 9 63
# 17: [5, 6] 1 41 23 83
# 17: [1] 1 34 5 29
# 18: [5] 1 49 31 179
# 18: [1] 1 47 54 258
# 19: [2] 2 30 3 24
# 19: [1] 2 32 5 26
# 20: [1, 4] 2 4 5 29
# 20: [3, 0] 2 4 5 29
# 21: [0] 1 0 1 2
# 21: [0] 1 0 1 2
# 22: [4] 2 23 8 55
# 22: [4] 2 22 8 57
# 23: [17] 1 190 18 108
# 23: [4] 2 22 8 57
# 24: [11] 1 126 12 137
# 24: [0] 1 126 12 142
# 25: [2] 1 22 3 8
# 25: [0] 1 126 12 142
# 26: [7] 1 46 12 87
# 26: [17] 1 44 20 125
# 27: [16] 1 369 25 127
# 27: [0] 1 145 12 83
# 28: [2] 1 27 3 28
# 28: [0] 1 29 5 31
# 29: [10] 1 243 14 46
# 29: [2] 1 1 3 10
# 30: [0, 4, 5, 6, 7, 8] 10 96 10 53
# 30: [5, 0, 7, 4, 8, 3] 10 96 10 53
# 31: [2] 2 92 4 13
# 31: [1] 2 92 4 13
# 32: [3, 5] 1 61 16 47
# 32: [13, 3, 15] 1 61 16 55
# 33: [1] 1 82 2 18
# 33: [0] 1 0 1 8
# 34: [17, 18] 27 211 88 291
# 34: [26, 19] 27 113 67 289
# 35: [0, 2] 1 10 3 8
# 35: [0] 1 10 2 8
# 36: [10] 2 378 20 164
# 36: [0, 12] 2 120 13 145
# 37: [0] 1 3 1 7
# 37: [0] 1 3 1 7
# 38: [9] 1 45 15 103
# 38: [19] 1 43 26 153
# 39: [1] 1 1 2 51
# 39: [0] 1 1 2 51
# 40: [2] 1 3 3 4
# 40: [0] 1 3 2 4
# 41: [4, 5] 2 76 6 28
# 41: [2, 0] 2 53 4 18
# 42: [10] 6 46 18 112
# 42: [4] 1 23 11 67
# 43: [13] 6 52 14 77******************
# 43: [3] 1 6 5 29
# 44: [9] 6 48 11 72******************
# 44: [0] 1 2 2 22
# 45: [11] 6 50 12 74*****************
# 45: [1] 1 4 3 24
# 46: [10, 11, 12, 13, 14] 6 50 15 70********
# 46: [1, 0, 2] 1 4 3 20
# 51: [0] 6 46 10 69**************
# 51: [0] 1 0 1 19
# 52: [12] 6 49 13 77**************
# 52: [1] 1 3 4 28
# 53: [10] 6 51 11 90**************
# 53: [0] 1 5 2 38
# 54: [9] 6 57 12 64***************
# 54: [0] 1 11 3 14
# 55: [11] 6 51 14 64***************
# 55: [4] 1 5 5 22
# 56: [31] 6 193 32 162**************
# 56: [4] 1 5 5 22
# 57: [10] 16 45 11 51**************
# 57: [1] 11 0 2 6
# 58: [11] 6 243 15 86***************
# 58: [1] 1 116 6 36
# 59: [11] 6 219 12 61*****************
# 59: [0] 1 174 3 12
# 60: [15, 22] 6 243 23 91***************
# 60: [5] 1 172 8 42
# 61: [16] 7 241 23 94******************
# 61: [0] 2 170 9 45
# 62: [3, 9] 6 44 10 71*****************
# 62: [0] 1 29 10 62
# 63: [11, 13] 6 61 20 166***************
# 63: [1] 1 16 11 108
# 64: [16] 8 235 17 84*******************
# 64: [7] 1 121 9 40
# 65: [10] 6 48 11 74********************
# 65: [0] 1 4 2 21
# nodesmax 510
# linenodesmax 776
# IfStatement
# ForStatement
# ReturnStatement
# LocalVariableDeclaration
# StatementExpression
# BreakStatement
# Statement
# ThrowStatement
# WhileStatement
# FieldDeclaration
# SwitchStatement
# ContinueStatement
# BlockStatement
# 13
# ROR.==.FALSE
# LVR.0.POS
# LVR.0.NEG
# ROR.==.<=
# ROR.==.>=
# COR.||.!=
# COR.||.RHS
# ROR.<.!=
# ROR.<.<=
# ROR.<.FALSE
# LVR.FALSE.TRUE
# ROR.==.LHS
# ROR.==.RHS
# COR.StringUtils.isBlank.TRUE
# COR.StringUtils.isBlank.FALSE
# COR.str.startsWith.TRUE
# COR.str.startsWith.FALSE
# STD.<ASSIGN>.<NO-OP>
# ROR.>.!=
# ROR.>.>=
# ROR.>.FALSE
# AOR.-.%
# AOR.-.*
# AOR.-.+
# AOR.-./
# LVR.POS.0
# LVR.POS.NEG
# AOR.+.%
# AOR.+.*
# AOR.+.-
# AOR.+./
# LVR.NEG.0
# LVR.NEG.POS
# COR.||.LHS
# COR.||.TRUE
# ROR.!=.<
# ROR.!=.>
# ROR.!=.TRUE
# COR.&&.==
# COR.&&.FALSE
# COR.&&.LHS
# COR.&&.RHS
# ROR.>=.==
# ROR.>=.>
# ROR.>=.TRUE
# AOR.*.%
# AOR.*.+
# AOR.*.-
# AOR.*./
# STD.<CALL>.<NO-OP>
# LVR.TRUE.FALSE
# STD.<DEC>.<NO-OP>
# COR.StringUtils.isEmpty.TRUE
# COR.StringUtils.isEmpty.FALSE
# COR.Character.isDigit.TRUE
# COR.Character.isDigit.FALSE
# ROR.<=.<
# ROR.<=.==
# ROR.<=.TRUE
# STD.<INC>.<NO-OP>
# COR.patternMatcher.lookingAt.TRUE
# COR.patternMatcher.lookingAt.FALSE
# COR.currentStrategy.addRegex.TRUE
# COR.currentStrategy.addRegex.FALSE
# COR.unquote.TRUE
# COR.unquote.FALSE
# COR.matcher.lookingAt.TRUE
# COR.matcher.lookingAt.FALSE
# COR.Character.isWhitespace.TRUE
# COR.Character.isWhitespace.FALSE
# COR.wasWhite.TRUE
# COR.wasWhite.FALSE
# COR.autoboxing.TRUE
# COR.autoboxing.FALSE
# COR.cls.equals.TRUE
# COR.cls.equals.FALSE
# COR.cls.isPrimitive.TRUE
# COR.cls.isPrimitive.FALSE
# COR.toClass.equals.TRUE
# COR.toClass.equals.FALSE
# COR.toParameterizedType.equals.TRUE
# COR.toParameterizedType.equals.FALSE
# COR.fromTypeVarAssigns.isEmpty.TRUE
# COR.fromTypeVarAssigns.isEmpty.FALSE
# COR.toGenericArrayType.equals.TRUE
# COR.toGenericArrayType.equals.FALSE
# COR.toWildcardType.equals.TRUE
# COR.toWildcardType.equals.FALSE
# COR.isAssignable.TRUE
# COR.isAssignable.FALSE
# COR.typeVarAssigns.containsKey.TRUE
# COR.typeVarAssigns.containsKey.FALSE
# COR.toClass.isPrimitive.TRUE
# COR.toClass.isPrimitive.FALSE
# COR.superClass.isInterface.TRUE
# COR.superClass.isInterface.FALSE
# ORU.-.+
# ORU.-.~
# LOR.&.^
# LOR.&.|
# COR.equals.TRUE
# COR.equals.FALSE
# AOR./.%
# AOR./.*
# AOR./.+
# AOR./.-
# COR.containsElements.TRUE
# COR.containsElements.FALSE
# COR.hasExp.TRUE
# COR.hasExp.FALSE
# COR.foundDigit.TRUE
# COR.foundDigit.FALSE
# COR.allowSigns.TRUE
# COR.allowSigns.FALSE
# COR.isHex.TRUE
# COR.isHex.FALSE
# COR.isRegistered.TRUE
# COR.isRegistered.FALSE
# COR.object.getClass.TRUE
# COR.object.getClass.FALSE
# COR.clazz.isArray.TRUE
# COR.clazz.isArray.FALSE
# COR.this.accept.TRUE
# COR.this.accept.FALSE
# COR.m.isEmpty.TRUE
# COR.m.isEmpty.FALSE
# COR.fieldSeparatorAtStart.TRUE
# COR.fieldSeparatorAtStart.FALSE
# COR.match.TRUE
# COR.match.FALSE
# COR.detail.TRUE
# COR.detail.FALSE
# COR.value.getClass.TRUE
# COR.value.getClass.FALSE
# COR.useShortClassName.TRUE
# COR.useShortClassName.FALSE
# COR.mTimeZoneForced.TRUE
# COR.mTimeZoneForced.FALSE
# COR.repeat.TRUE
# COR.repeat.FALSE
# AOR.%.*
# AOR.%.+
# AOR.%.-
# AOR.%./
# COR.val.startsWith.TRUE
# COR.val.startsWith.FALSE
# COR.bool.booleanValue.TRUE
# COR.bool.booleanValue.FALSE
# COR.escapeSingleQuote.TRUE
# COR.escapeSingleQuote.FALSE
# COR.offsetSet.TRUE
# COR.offsetSet.FALSE
# COR.Token.containsTokenWithValue.TRUE
# COR.Token.containsTokenWithValue.FALSE
# COR.padWithZeros.TRUE
# COR.padWithZeros.FALSE
# COR.entry.map.containsKey.TRUE
# COR.entry.map.containsKey.FALSE
