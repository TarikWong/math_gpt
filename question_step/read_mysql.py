# -*- coding: utf-8 -*-
# @Time : 2023/11/27 5:44 下午
# @Author : tuo.wang
# @Version :
# @Function :
import pandas as pd
from sqlalchemy import create_engine

# 创建数据库连接
username = "lable_read"
password = "j3T8im$d0Ss#7Ui2"
host = "pc-2zem9nkj5n63a2n3v.mysql.polardb.rds.aliyuncs.com"  # 如果MySQL容器和Jupyter在同一个Docker网络中，请使用MySQL容器名称作为主机
port = "3306"  # 默认MySQL端口
database = "spier_data"
table = "question_perfect"

# cnt = 100
engine = create_engine(
    f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
)

query = '''
SELECT * FROM question_perfect
WHERE question_id IN 
(
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
    '2_66330.json_33076012'
)
'''
df = pd.read_sql(query, engine)

# 查看前几行数据
output_dir = "/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step/data/math/senior/new.csv"
df.to_csv(output_dir, index=False)
print("done.")
