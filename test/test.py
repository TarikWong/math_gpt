import json

# import pandas as pd
#
# data = pd.DataFrame({'name': ['wencky', 'stany', 'barbio'],
#                      'age': [29, 29, 3],
#                      'gender': ['w', 'm', 'm']})
#
# print(data)
# # print('age去重', data["age"].unique(), sep='\n')
# print('去重后数量', len(data["age"].unique()), sep='\n')



json_string = """{
    "subjectName":"数学",
    "gradeGroupName":"初中",
    "content":" 某商品的价格标签已丢失，售货员只知道“它的进价为80元，打七折售出后，仍可获利5%”．你认为售货员应标在标签上的价格为$$underline{------------}$$元． ",
    "answer":[
        [
            " 120  解：设售货员应标在标签上的价格为x元，依据题意70%x=80×（1+5%）可求得：x=120，价格应为120元． "
        ]
    ],
    "analysis":[
        ""
    ],
    "logicQuesTypeName":"填空题",
    "que_id":"100551313",
    "examOptionList":"一元一次方程的定义,一元一次方程的解法,一元一次方程中的待定系数,一元一次方程的应用",
    "source_text":"魔方格",
    "origin_text":"抓取",
    "question_id":"xpombNmz3e60VXlg",
    "source_id":1,
    "source_group":"qingzhou",
    "en_question_id":"YTFZTXVEQ1h5TzFWUFNaTGZKQVU3ZFpuV1lYckVjZ2YvdWo2TERiaWdhK0k9",
    "path":"1_26368.json"
}"""

# a = json.loads(json_string)
# print(type(a))
# print(a.get('aaa', '-').split(','))

a = ['1', '2', '3', '4']
print(','.join(a))