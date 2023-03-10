import copy
import os.path
import pickle
import sys
import csv

import FaultMe
import Tool_io


def cal_metrics(root, data, error_pro_ver, res_path):
    # 若存在结果 直接返回
    two_metric = Tool_io.checkAndLoad(res_path, "all_pro")
    if two_metric != None:
        return two_metric

    all = Tool_io.create_data_folder(root, data)
    pros_op = {}
    pros_op2 = {}
    for pros in all:
        vers_op = {}
        vers_op2 = {}
        for ver in all[pros]:
            if os.path.exists(os.path.join(ver[1],'fcci_sus_value')):
                print(ver)
                loc = 0
                sus_value = Tool_io.checkAndLoad(ver[1], "fcci_sus_value")
                res = Tool_io.checkAndLoad(ver[1], "data_Coverage_InVector_saveAgain")
                loc += len(open(os.path.join(ver[0], 'hugeCode.txt')).readlines())
                fault_index = res[2]
                tmp = {}
                tmp2 = {}
                for sv in sus_value:
                    top_res = getTop(sus_value[sv],fault_index,[1,2,3,5,10,50])
                    cost = getEXAM(sus_value[sv],fault_index,loc)
                    tmp[sv] = top_res
                    tmp2[sv] = cost
                vers_op[os.path.basename(ver[0])] = tmp
                vers_op2[os.path.basename(ver[0])] = tmp2
        pros_op[os.path.basename(pros)] = vers_op
        pros_op2[os.path.basename(pros)] = vers_op2
    two_metric = {}
    two_metric[0] = pros_op
    two_metric[1] = pros_op2
    Tool_io.checkAndSave(res_path, "all_pro", two_metric)
    return two_metric
    print("s")


# 计算exam
def getEXAM(dict, fault_location,loc):
    fault_of_cost = []
    cost = {}
    exam = {}
    sus_find_temp = []

    # cost_temp = {}

    for index in range(len(dict)):
        line_no = dict[index][0]
        if line_no in fault_location:
            sus_find_temp.append(dict[index][1])
            # cost_temp[4] = index + 1
            fault_of_cost.append(line_no)

    sus_find=[]
    for index in range(len(fault_location)):
        for index2 in range(len(fault_of_cost)):
            if fault_of_cost[index2]==fault_location[index]:
                sus_find.append(sus_find_temp[index2])
                break

    for sus_index in range(len(sus_find)):
        cost[sus_index] = {}
        exam[sus_index] = {}
        first_index = -1
        end_index = -1
        sus_record = sus_find[sus_index]
        for sus_item in range(len(dict)):
            if dict[sus_item][1] == sus_record and first_index == -1:
                first_index = sus_item + 1
            if first_index != -1 and dict[sus_item][1] != sus_record:
                end_index = sus_item
                break
        if end_index == -1:
            end_index = len(dict)
        average_index = (first_index + end_index) / 2
        cost[sus_index][1] = first_index
        cost[sus_index][2] = end_index
        cost[sus_index][3] = average_index
        # exam[sus_index][1] = first_index / len(dict)
        # exam[sus_index][2] = end_index / len(dict)
        # exam[sus_index][3] = average_index / len(dict)
        exam[sus_index][1] = first_index / loc
        exam[sus_index][2] = end_index / loc
        exam[sus_index][3] = average_index / loc
    return exam


# top-N
def getTop(dict, fault_location, TopList):
    TopList.sort(reverse=True)
    top_num = {}
    for item in TopList:
        top_num[item] = 0
    sus_find = []

    for index in range(len(dict)):
        line_no = dict[index][0]
        if line_no in fault_location:
            sus_find.append(dict[index][1])

    for sus_index in range(len(sus_find)):
        first_index = -1
        sus_record = sus_find[sus_index]
        for sus_item in range(len(dict)):
            if dict[sus_item][1] == sus_record and first_index == -1:
                first_index = sus_item + 1
                break
        for item in reversed(TopList):
            if first_index <= item:
                top_num[item] += 1
                # break

    return top_num


def cal_end(all_pro,res_path,model_path,ver_data):
    top_all = all_pro[0]
    wasted_all = all_pro[1]
    for t_name in top_all:
        # f = open(os.path.join(model_path, t_name + '.td'), 'rb')
        # td = pickle.load(f)
        td=ver_data
        delete = []
        for ex in top_all[t_name]:
            if ex not in td[t_name]:
                delete.append(ex)
        for d in delete:
            top_all[t_name].pop(d)

    for t_name in wasted_all:
        # f = open(os.path.join(model_path, t_name + '.td'), 'rb')
        # td = pickle.load(f)
        td=ver_data
        delete = []
        for ex in wasted_all[t_name]:
            if ex not in td[t_name]:
                delete.append(ex)
        for d in delete:
            wasted_all[t_name].pop(d)

    FaultMe.detail_info(top_all, wasted_all, res_path)
    p = {}
    for pro_top in top_all:
        tmp = {
            "ochiai": {},
            "ochiai_c": {},
            "ochiai_r": {},
            "ochiai_e": {},
            "ds": {},
            "ds_c": {},
            "ds_r": {},
            "ds_e": {},
            "op": {},
            "op_c": {},
            "op_r": {},
            "op_e": {},
            "ds2": {},
            "ds2_c": {},
            "ds2_r": {},
            "ds2_e": {},
        }
        for ver_top in top_all[pro_top]:
            dict = top_all[pro_top][ver_top]
            for di in dict:
                top_value = dict[di]
                if len(tmp[di]) == 0:
                    tmp[di] = top_value
                    continue
                for index, val in enumerate (top_value):
                    tmp[di][val] = tmp[di][val] +  top_value[val]
                print("s")
        p[pro_top] = tmp

    wasted = {}
    for pro_top in wasted_all:
        tmp = {
            "ochiai": {},
            "ochiai_c": {},
            "ochiai_r":{},
            "ochiai_e": {},
            "ds": {},
            "ds_c": {},
            "ds_r": {},
            "ds_e": {},
            "op": {},
            "op_c": {},
            "op_r": {},
            "op_e": {},
            "ds2": {},
            "ds2_c": {},
            "ds2_r": {},
            "ds2_e": {},
        }
        for ver_top in wasted_all[pro_top]:
            dict = wasted_all[pro_top][ver_top]
            for di in dict:
                if not dict[di].__contains__(0):
                    continue
                min_wasted = dict[di][0][3]
                dict_cost = dict[di]
                if len(dict_cost) == 1:
                    min_wasted = dict[di][0][3]
                else:
                    for index,cost_index in enumerate (dict_cost) :
                        if min_wasted > dict_cost[cost_index][3]:
                            min_wasted = dict_cost[cost_index][3]
                tmp[di][ver_top] = min_wasted
        wasted[pro_top] = tmp
    deal_csv(p,wasted,res_path)
    print("s")


# 存储wasted详细结果
def detail_wasted(rows,res_path,csv_name):
    path = os.path.join(res_path, csv_name) + ".csv"
    header = ['name', 'version', 'formula', 'wasted value...']
    with open(path, 'a+', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)

def deal_csv(p,wasted,res_path):
    for pro_name in wasted:
        # if pro_name != 'Chart':
        #     continue
        pro = wasted[pro_name]
        for sus_for in pro:
            ver_count = 0
            for version in pro[sus_for]:
                ver_count = ver_count + pro[sus_for][version]
            wasted[pro_name][sus_for] = ver_count/len(pro[sus_for])
    rows = []
    for pro in wasted:
        row = []
        row.append(pro)
        for val in wasted[pro]:
            row.append(wasted[pro][val])
        rows.append(row)
    creat_res_file(rows,res_path,'wasted_avg')
    print("s")

    contents = []
    for top in p:
        # if top != 'Chart':
        #     continue
        pro_top = p[top]
        for sus in pro_top:
            content = []
            content.append(top)
            content.append(sus)
            for value in pro_top[sus]:
                content.append(pro_top[sus][value])
            contents.append(content)
    creat_res_file_top(contents,res_path,'top')
    print("s")



# 存储结果
def creat_res_file_top(rows,res_path,csv_name):
    path = os.path.join(res_path, csv_name) + ".csv"
    header = ['', 'sus_formula', 'top-50','top-10','top-5','top-3','top-2','top-1']
    with open(path, 'a+', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)

# 存储结果
def creat_res_file(rows,res_path,csv_name):
    path = os.path.join(res_path, csv_name) + ".csv"
    header = ['', 'ochici', 'ochiai_c','ochiai_r','ochiai_e','ds','ds_c','ds_r','ds_e','op','op_c','op_r','op_e','ds2','ds2_c','ds2_r','ds2_e']
    with open(path, 'a+', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


if __name__ =="__main__":

    root = '/home/tianshuaihua/base_dataset'
    data = '/home/tianshuaihua/tpydata'
    error_pro_ver = '/home/tianshuaihua/error'
    model_path = '/home/tianshuaihua/model'
    res_path = '/home/tianshuaihua/fcci/res'

    vers_path = '/home/tianshuaihua/wang/fea'
    ver_data = Tool_io.checkAndLoad(vers_path, 'vers.in')

    all_pro = cal_metrics(root, data, error_pro_ver, res_path)

    cal_end(all_pro,res_path,model_path,ver_data)