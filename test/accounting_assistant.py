# -*- coding: utf-8 -*-
# @Time : 2023/10/30 11:44 上午
# @Author : tuo.wang
# @Version : 
# @Function :
from typing import *

outs = [733.79, 522.80]
ins = [465.88, 109.80]


def add(bill: List):
    result = 0.0
    for i in bill:
        result += i
    return result


if __name__ == "__main__":
    outs_result = add(outs)
    ins_result = add(ins)

    print("支出： ", outs_result)
    print("收入： ", ins_result)
