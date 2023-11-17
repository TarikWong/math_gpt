from call_gpt import send_chat_request
import json

import pandas as pd

# i = """{
#     "kc":[
#         {"1":["平移的定义与性质"]},
#         {"2":["平移的定义与性质"]},
#         {"3":[]}
#     ],
#     "reason":[
#             {"1":{"平移的定义与性质":"第一步是理解平移的性质。在平面直角坐标系中，如果一个图形的所有点的纵坐标都减去一个相同的值，那么这个图形就会向下平移相应的单位。这是因为在平面直角坐标系中，纵坐标决定了点在垂直方向上（也就是上下）的位置。如果我们减小每个点的纵坐标值，那么每个点都会向下移动，所以整个图形也就向下平移了。"}},
#             {"2":{"平移的定义与性质":"第二步是应用刚才理解到的平移性质。根据题目描述，三角形所有顶点横坐标保持不变、纵坐标都减去5, 这意味着三角形沿垂直方向（即上下方向）发生了位移，并且位移量为5单位长度。由于我们已经知道当纵坐标减小时图形会向下平移，所以可以确定这等价于将三角形整体向下平移5个单位。"}},
#             {"3":{}}
#     ]
# }"""


i = """{
    "kc":[
        {"1":["平移的定义与性质"]},
        {"2":["平移的定义与性质"]},
        {"3":["测试"]}
    ],
    "reason":[
            {"1":{"平移的定义与性质":"第一步是理解平移的性质。在平面直角坐标系中，如果一个图形的所有点的纵坐标都减去一个相同的值，那么这个图形就会向下平移相应的单位。这是因为在平面直角坐标系中，纵坐标决定了点在垂直方向上（也就是上下）的位置。如果我们减小每个点的纵坐标值，那么每个点都会向下移动，所以整个图形也就向下平移了。"}},
            {"2":{"平移的定义与性质":"第二步是应用刚才理解到的平移性质。根据题目描述，三角形所有顶点横坐标保持不变、纵坐标都减去5, 这意味着三角形沿垂直方向（即上下方向）发生了位移，并且位移量为5单位长度。由于我们已经知道当纵坐标减小时图形会向下平移，所以可以确定这等价于将三角形整体向下平移5个单位。"}},
            {"3":{"测试":"测试原因"}}
    ]
}"""

# def modify_first_response(first_response, kc_list=["概率"]):
#     for kc in first_response["kc"]:
#         if kc not in kc_list:
#             first_response["kc"].remove(kc)
#             del first_response["reason"][kc]
#     return first_response


# def modify_second_response(second_response, first_response=["测试"], last_level_kc_set=set(["平移的定义与性质"])):
#     for idx in range(len(second_response["kc"])):
#         for k, v in second_response["kc"][idx].items():
#             for kc in set(list(v)):
#                 if kc not in last_level_kc_set:
#                     second_response["kc"][idx][k].remove(kc)
#                     del second_response["reason"][idx][k][kc]
#     return second_response

#
# second_response = json.loads(i)
# i_new = modify_second_response(second_response)
# print(i_new)

a = "1"
print(len(a.split(",")))
print(a.split(","))
