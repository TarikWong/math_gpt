import pandas as pd
import json
from typing import *

# class Test(object):
#     def __init__(self,
#                  sample_cnt: int,
#                  input_file: str,
#                  output_file: str,):
#         print(sample_cnt)
#         print(input_file)
#         print(output_file)
#
#
#
# if __name__ == "__main__":
#     c = [
#         'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions. USER: 给出解题步骤：$0.4444+0.5555+$?$=1$（   ）．A. $0.00001$B. $0.0001$C. $0.0111$D. $0.1111$ ASSISTANT: 题目中的式子是$0.4444+0.5555+$（   ）$=1$。我们可以通过计算$0.4444+0.5555$来找到缺失的数字。$0.4444+0.5555=$ <thought>这里需要计算加法，我需要在这里调用 Calculator API <API>[{"ActionName": "Calculator", "Args": {"equation":"0.4444+0.5555"}}] </API> => 1.0009</thought> $1.0009$。因此，缺失的数字是$1-1.0009=$ <thought>这里需要计算减法，我需要在这里调用 Calculator API <API>[{"ActionName": "Calculator", "Args": {"equation":"1-1.0009"}}] </API> => 0.0001</thought> $0.0001$。因此，答案是B. $0.0001$。</s>',
#         "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Which is equal to $0.11$ （   ）．A. $\\frac{1}{9}$B. $\\frac{10}{11}$C. $\\frac{11}{100}$D. $\\frac{1}{11}$ 帮我解答 ASSISTANT: To find which fraction is equal to $0.11$, we can convert $0.11$ to a fraction. $0.11$ can be written as $\\frac{11}{100}$, which is choice $\\text{C}$. Therefore, the answer is $\\text{C}$.</s>"
#     ]
#     print(len(c))


a = "unknown"

if "unknown" in a:
    print("yes")
else:
    print('no')
