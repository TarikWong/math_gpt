# from call_gpt import send_chat_request
# import json
# import string
#
# import pandas as pd
#
# import string
#
#
# def million_to_billion(m):
#     return m * 1000000 / 1000000000
#
#
# def billion_to_million(m):
#     return m * 1000000000 / 1000000
#
#
# def trillion_to_billion(m):
#     return m * 1000
#
#
# if __name__ == "__main__":
#     approach2_models_params = [
#         {"parameters": 400, "tokens": billion_to_million(7.7)},
#         {"parameters": billion_to_million(1), "tokens": billion_to_million(20.0)},
#         {"parameters": billion_to_million(10), "tokens": billion_to_million(219.5)},
#         {"parameters": billion_to_million(67), "tokens": billion_to_million(trillion_to_billion(1.7))},
#         {"parameters": billion_to_million(175), "tokens": billion_to_million(trillion_to_billion(4.3))},
#         {"parameters": billion_to_million(280), "tokens": billion_to_million(trillion_to_billion(7.1))},
#         {"parameters": billion_to_million(520), "tokens": billion_to_million(trillion_to_billion(13.4))},
#         {"parameters": billion_to_million(trillion_to_billion(1)),
#          "tokens": billion_to_million(trillion_to_billion(26.5))},
#         {"parameters": billion_to_million(trillion_to_billion(10)),
#          "tokens": billion_to_million(trillion_to_billion(292.0))},
#     ]
#
#     for i in approach2_models_params:
#         print(
#             "parameters: {}, tokens: {}, rate: {}".format(i["parameters"], i["tokens"], i["tokens"] / i["parameters"]))
#
#     print("===")
#
#     approach3_models_params = [
#         {"parameters": 400, "tokens": billion_to_million(9.2)},
#         {"parameters": billion_to_million(1), "tokens": billion_to_million(27.1)},
#         {"parameters": billion_to_million(10), "tokens": billion_to_million(410.1)},
#         {"parameters": billion_to_million(67), "tokens": billion_to_million(trillion_to_billion(4.1))},
#         {"parameters": billion_to_million(175), "tokens": billion_to_million(trillion_to_billion(12.0))},
#         {"parameters": billion_to_million(280), "tokens": billion_to_million(trillion_to_billion(20.1))},
#         {"parameters": billion_to_million(520), "tokens": billion_to_million(trillion_to_billion(43.5))},
#         {"parameters": billion_to_million(trillion_to_billion(1)),
#          "tokens": billion_to_million(trillion_to_billion(94.1))},
#         {"parameters": billion_to_million(trillion_to_billion(10)),
#          "tokens": billion_to_million(trillion_to_billion(1425.5))},
#     ]
#
#     for i in approach3_models_params:
#         print(
#             "parameters: {}, tokens: {}, rate: {}".format(i["parameters"], i["tokens"], i["tokens"] / i["parameters"]))
#
#     print("Chinchilla")
#     print(
#         "parameters: {}, tokens: {}, rate: {}".format(billion_to_million(70),
#                                                       billion_to_million(trillion_to_billion(1.4)),
#                                                       billion_to_million(trillion_to_billion(1.4)) / billion_to_million(70)))
#
#
# # with open('/Users/tuo/PycharmProjects/math_gpt/question_step/data/math/senior/kc1.txt', 'r') as file:
# #     # with open('/Users/tuo/PycharmProjects/math_gpt/question_step/data/math/senior/kc2.txt', 'r') as file:
# #     kc_list = file.readline().strip().split("@suzuki@")
# #     count_dict = {item: 0 for item in kc_list}
# #     print(json.dumps(count_dict, ensure_ascii=False, indent=4))
# #     print(len(kc_list))
#

print(100- 77/663*100)