# -*- coding: utf-8 -*-
# @Time : 2023/10/16 2:19 下午
# @Author : tuo.wang
# @Version : 
# @Function :

import pymysql
from pandas import DataFrame
import json

mysql_config_dict = {
    "精美解析题库": {
        'host': 'pc-2zem9nkj5n63a2n.mysql.polardb.rds.aliyuncs.com',
        'user': 'lable_read',
        'pwd': 'j3T8im$d0Ss#7Ui2',
        'database': 'spier_data',
        'table': 'question_perfect'}
}
columns = ['question_id', 'response_json']


def init_mysql(config):
    return pymysql.connect(host=config['host'],
                           user=config['user'],
                           passwd=config['pwd'],
                           db=config['database'],
                           charset='utf8')

def rdb2df(conn, sql, columns, type='mysql'):
    cursor = conn.cursor()
    cursor.execute(sql)
    conn.commit()
    data = cursor.fetchall()
    return DataFrame(data)

def json_analysis(line):
    pass
    # line2list = json.loads(line)
    # new_dict = {}
    # for question in line2list:
    #     for step in question['solution']:
    #         # 返回数据的格式：题干的question_id___子题的question_id___步骤id
    #         key = "{}___{}___{}".format()
    #         new_dict[]







if __name__ == '__main__':
    config = mysql_config_dict['精美解析题库']
    conn = init_mysql(config)
    sql = "select {} from {} WHERE remark='成功' and subject_id=2 limit 10;".format(','.join(columns), config['table'])
    print('sql: ', sql)
    df = rdb2df(conn, sql, columns)
    print(df)
    print('done.')
