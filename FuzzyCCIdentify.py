import Tool_localization
from Tool_distance import get_distance_CC_weight

if __name__ == "__main__":
    covMatrix=[[]]#测试用例覆盖信息二维矩阵
    in_vector=[]#测试用例执行结果（1为失败，0为通过）
    fail_test=[]#失败测试用例下标
    pass_test=[]#通过测试用例下标
    sus_oc, sus_tu, sus_op, tf, tp, sus_ds, sus_cr, sus_oc_new = Tool_localization.CBFL_location([], covMatrix,in_vector)
    cc_pro_single, ochiai_cc, dstar_cc, turantula_cc, op2_cc, crosstab_cc, ochiai_cc_new = get_distance_CC_weight(covMatrix,sus_oc, fail_test, pass_test)