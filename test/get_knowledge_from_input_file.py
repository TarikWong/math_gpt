# -*- coding: utf-8 -*-
# @Time : 2023/11/21 2:52 下午
# @Author : tuo.wang
# @Version : 
# @Function :
import pandas as pd
import json


def dfs(data: list):
    names = []
    if not data:
        return names
    for item in data:
        if not item["childList"]:
            names.append(item["name"])
        else:
            names.extend(dfs(item["childList"]))
        if "labelKnowList" in item and isinstance(item["labelKnowList"], list):
            names.extend(dfs(item["labelKnowList"]))
    return names


# examOptionList
def get_kc1(column):
    if isinstance(column, str):
        d = json.loads(column)
        if "examOptionList" in d.keys():
            return dfs(d["examOptionList"])
    return []


# points
def get_kc2(column):
    if isinstance(column, str):
        d = json.loads(column)
        if "points" in d.keys():
            points = d["points"].split("||")
            return [item for item in points if '' != item]
    return []


def is_elements_in_string(target_string, elements_list):
    for element in elements_list:
        if element in target_string:
            return True
    return False


# kaodian
def get_kc3(column):
    if isinstance(column, str):
        d = json.loads(column)
        if "kaodian" in d.keys():
            kd_list = list(set(d["kaodian"]["value"]))
            output = []
            for kd in kd_list:
                if not is_elements_in_string(kd, ["分析", "解答", "点评", "故", "根据", "则", "\n", "本题"]):
                    output.append(kd)
            return output
    return []


df = pd.read_csv("/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step/data/math/senior/source3.csv")
# df = pd.read_csv("/Users/tuo/PycharmProjects/math_gpt/question_step/data/math/senior/source3_sample200.csv")
# data = df.sample(200)
# data.to_csv("/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step/data/math/senior/source3_sample200.csv", index=False)
# df = pd.read_csv("/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step/data/math/senior/source1.csv")


df["new_knowledge"] = df["original"].apply(get_kc3)
# print(df["original"])

# l = df["new_knowledge"].to_list()
# for i in l:
#     print(i)
#     print("------")
# print(df["original"].to_list()[0])
#
# 将所有列表合并成一个大列表
all_values = [item for sublist in df['new_knowledge'] for item in sublist]
#
# 使用set数据类型去除重复值，并转换回list
unique_values = list(set(all_values))
# print(unique_values)

with open('/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step/data/math/senior/kc3.txt.back', 'w') as f:
    f.write("@suzuki@".join(unique_values))

print("done.")
