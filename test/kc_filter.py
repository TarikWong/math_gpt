# -*- coding: utf-8 -*-
# @Time : 2023/11/21 7:45 下午
# @Author : tuo.wang
# @Version :
# @Function :

import pandas as pd
import pandas
from get_knowledge_from_input_file import *
import json

# df = pd.read_csv("/Users/tuo/PycharmProjects/math_gpt/question_step/data/math/senior/source2_sample200.csv")
df = pd.read_csv("/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step/data/math/senior/source3.csv")
df["new_knowledge"] = df["original"].apply(get_kc3)

with open('/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step/data/math/senior/kc3.txt', 'r') as file:
    # with open('/Users/tuo/PycharmProjects/math_gpt/question_step/data/math/senior/kc2.txt', 'r') as file:
    kc_list = file.readline().strip().split("@suzuki@")
    # 创建一个字典来跟踪filter_list中每个元素的出现次数
    count_dict = {item: 0 for item in kc_list}
    print("kc_list's length: ", len(kc_list))


def check_row(row, max_count=5, column_name="new_knowledge"):
    # 如果该行包含filter_list中的任何元素，增加该元素的计数
    flag = False
    scores = []
    for item in row[column_name]:
        count_dict[item] += 1
        scores.append(count_dict[item])

    # 计算该题目的考点平均分数，如果超过max_count，返回False以过滤掉该行
    if len(row[column_name]) > 0:
        # print("final score: ", score / len(row[column_name]))
        if min(scores) <= max_count:
            flag = True
    return flag


def extract_questions_from_topics(input_df, column_name="new_knowledge", max_count=3):
    filtered_df = input_df[
        input_df.apply(check_row, max_count=max_count, column_name=column_name, axis=1)]
    return filtered_df


if __name__ == "__main__":
    filtered_df = extract_questions_from_topics(input_df=df, column_name="new_knowledge", max_count=3)
    filtered_df_length = len(filtered_df)
    print(f'The length of filtered_df is: {filtered_df_length}')
    # print("kc_list: ", json.dumps(kc_list, ensure_ascii=False, indent=4))
    filtered_df.to_csv(
        "/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step/data/math/senior/source3_kc_sample{}.csv".format(
            filtered_df_length), index=False)
    print("success.")
