import os
import copy
import random

import numpy as np

import Tool_io


# 获取数据集
def deal_program(args):
    # 若存在 直接返回
    total_res = Tool_io.checkAndLoad(args.origin, args.record)
    if total_res != None:
        return total_res

    pro_names = Tool_io.get_folder(args.data_path)
    datasets = {}
    # 遍历不同程序
    for pro_name in pro_names:
        pro_path = os.path.join(args.data_path, pro_name)
        ver_names = Tool_io.get_folder(pro_path)
        # 遍历程序不同版本
        pro_dataset = []
        for ver_name in ver_names:
            ver_path = os.path.join(pro_path, ver_name)
            # 获取程序版本信息
            coverage_info = Tool_io.checkAndLoad(ver_path, args.coverage_file)
            if coverage_info is not None:
                # 获取归一化后的特征
                res_nor = Tool_io.checkAndLoad(ver_path, args.normalization)
                if res_nor is not None and len(coverage_info[2]) > 0:
                    pro_dataset.append(ver_name)
        if len(pro_dataset) > 0:
            datasets[pro_name] = pro_dataset

    # 保存结果后返回
    Tool_io.checkAndSave(args.origin, args.record, datasets)
    return datasets


# 处理数据集
def deal_dataset(ver_list, features, formulas, pro_path, coverage_file, origin_path, pro_name):

    normal_data = Tool_io.checkAndLoad(origin_path, pro_name+'_data')
    if normal_data != None:
        return normal_data

    pro_dict = {}
    for ver_name in ver_list:
        ver_dict = {}
        ver_path = os.path.join(pro_path, ver_name)
        # 数据样本处理
        ver_data = get_origin_data(ver_path, features, formulas)
        # 样本标签处理
        data_label = Tool_io.checkAndLoad(ver_path, coverage_file)
        cc_vector = np.zeros(data_label[4] + data_label[5], dtype=np.int)
        for cc_index in data_label[6]:
            cc_vector[cc_index] = 1
        for fault_index in data_label[7]:
            cc_vector[fault_index] = 1

        ver_dict['sample'] = ver_data
        ver_dict['label'] = cc_vector
        pro_dict[ver_name] = ver_dict

    normal_data = data_normalization(pro_dict)
    Tool_io.checkAndSave(origin_path, pro_name+'_data', normal_data)
    print("s")


# 数据归一化
def data_normalization(pro_dict):
    data_list = list()
    data_len = []
    for ver in pro_dict:
        sample = pro_dict[ver]['sample']
        data_list.append(sample)
        data_len.append(sample.shape[1])
    # 归一化计算
    arrn = np.concatenate(data_list, axis=1)
    nor_res = []
    for row in range(np.shape(arrn)[0]):
        row_val = arrn[row, :]
        fe = []
        r_mean = row_val.mean()
        r_std = row_val.std()
        for x in row_val:
            if r_std != 0:
                x = float(x - r_mean) / r_std
                fe.append(x)
            else:
                fe.append(0)
        nor_res.append(fe)
    normalization = np.array(nor_res)
    # 分割版本数据
    tmp_list = []
    for index in range(len(data_len)):
        if index == 0:
            tmp_list.append(data_len[index])
        else:
            tmp_list.append(data_len[index] + tmp_list[index - 1])

    res = np.split(normalization, tmp_list, axis=1)
    res_dict = {}

    for i, key in enumerate(pro_dict):
        tmp_dict = {}
        tmp_dict['label'] = pro_dict[key]['label']
        tmp_dict['sample'] = res[i]
        res_dict[key] = tmp_dict

    return res_dict


# 获取原始数据
def get_origin_data(ver_path, features, formulas):
    # 获取特征
    feature_list = []
    for fea in features:
        cc_feature = Tool_io.checkAndLoad(ver_path, fea)
        # 增加文本相似度特征
        if fea == 'similarity.in':
            text_info = {}
            for index in range(len(cc_feature)):
                text_info[index] = cc_feature[index]
            data_info = {}
            data_info['text'] = text_info
            cc_feature = data_info
        # 增加静态特征
        if fea == 'static_all':
            deal_array = np.array(cc_feature).T
            stat_all = {}
            for sta_index in range(len(deal_array)):
                sta_row = deal_array[sta_index]
                sta_dict = {}
                for val_index in range(len(sta_row)):
                    sta_dict[val_index] = sta_row[val_index]
                stat_all['stat'+str(sta_index)] = sta_dict
            cc_feature = stat_all
        feature_list.append(cc_feature)
    # 筛选怀疑都公式
    deal_list = []
    for feature in feature_list:
        deal_fea = select_formula(feature, formulas)
        deal_list.append(deal_fea)
    # 整合数据
    lst = list()
    for dea_fea in deal_list:
        cov = []
        for key in dea_fea:
            row = []
            for k in dea_fea[key]:
                row.append(dea_fea[key][k])
            cov.append(row)
        tmp = np.array(cov)
        lst.append(tmp)

    arrn = np.concatenate(lst, axis=0)
    return arrn


# 选择公式
def select_formula(feature, formulas):
    copy_feature = copy.deepcopy(feature)
    for key in feature:
        if key == 'text' or key.__contains__('stat'):
            continue
        if key not in formulas:
            copy_feature.pop(key)
    return copy_feature



if __name__ == '__main__':
    a = ['1b','2b','3b','4b','5b','6b','7b','8b','9b','10b']
    # s = select_data(a,2,1,1)
    # d = one_test(a,1)