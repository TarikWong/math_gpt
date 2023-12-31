import pandas as pd
import json
from utils import obj_to_dict
from loguru import logger

# 读取生产文件中的一行数据
# df1 = pd.read_csv("/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step/data/math/junior/source1.csv")
# df2 = pd.read_csv("/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step/data/math/junior/source2.csv")
# df3 = pd.read_csv("/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step/data/math/junior/source3.csv")


# question_id = "08606ce772234db2b83343eea0526c11"
# result_row = df.loc[df['question_id'] == question_id]

# print("source1: ", len(df1))
# print("source2: ", len(df2))
# print("source3: ", len(df3))
# print("done.")


# import json
#
# with open('/Users/tuo/PycharmProjects/exquisite_analysis_knowledge_tagging/question_step/tmp/math/junior/source3_sample_input_batch_10000.json', 'r') as f:
#     data = json.load(f)
#
# print(len(data))
# print("done.")


import pandas as pd


def query_dataframe(df, column, lst):
    # 使用 isin() 函数过滤出包含在提供的 list 中的值
    return df[df[column].isin(lst)]


# 测试上述函数
df = pd.read_csv(
    "/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step/data/math/junior/精美解析结果库_subject_2_grade_group_2.csv")

lst = [
    'acc543961b654466a5df09890e9ffa6c',
    '89dfacbe679b411b94694866ec978b94',
    'fc118657e5444cb0a476ea21673135a2',
    '116b4a4d496a4ded84c02cc30f0786ee',
    '3a14debe08aa4dacb9993b0968a55015',
    '538c36047f9346ae9c34b62ed8ab4670',
    '0e444ec54e37442fba8d81838fe3b0d0',
    '41c1a4467ee74573b3e83183ececb0be',
    '86b47e76d8dc4004ad6aaab278c565d0',
    'a223013699eb4501a7c919020496c549',
    '22_17356.json_102057348',
    '43_54899.json_197262219',
    '43_71085.json_229169515',
    '2_89224.json_104102006',
    '22_1208.json_103774445',
    '121_120274.json_124596403',
    '2_1085.json_118984021',
    'shiti0724d95a12e477d02b50b025e1b367fc6ce1',
    'shiti0724427923e46b944de47f67fbed93ae781b',
    'shiti0724e385debe4349ec702a57fd268ffe083a',
    'shiti07246d1baecc39d890e7d257422fbd9c9851',
    'shiti071271cce94eb60cbc8b28a5f237323c5e6c',
    'shiti0814573b02af6b0c7509ca4264dccfad276a',
    'shiti08148f24d986c19e7c34336614e9138f3ba0',
    'shiti07120b1f8d2028e9fdee41ff9328d0be3817',
    'shiti0724ed03b5d5ff92cfe6d4e530de24aec811',
    'shiti0814607aeeabbc5861c1716714a25a67eb1c',
    'shiti08140b30befb3e930c5d3af40772d0e6ced2',
    'shiti0814d1c606422d1fd0c6eb76f20f111d669b',
    '49f77e228be5475da09252da0b9cf23a',
    'c10aa1d6531745f09cf52bc44960ddeb',
    'f21e0056d355418390a6356f10a10258',
    'efe07b6585c34e4fa48fa3d00b55d1b9',
    '121_128800.json_131133175',
    '2_66330.json_33076012',
]

new_df = query_dataframe(df, "question_id", lst)
new_df.to_csv("/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step/data/math/junior/new.csv")
print("done.")
