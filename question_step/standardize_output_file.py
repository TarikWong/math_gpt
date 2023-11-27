# -*- coding: utf-8 -*-
# @Time : 2023/11/27 11:32 上午
# @Author : tuo.wang
# @Version : 
# @Function : 打标结果数据标准化
import pandas as pd
import json
from typing import *
from dataclasses import dataclass


@dataclass
class StepInfo:
    step: str
    title: str
    detail: str
    kc: List[str]


@dataclass
class SubQuestion:
    question: str
    solution: List[StepInfo]


@dataclass
class ResultData:
    perfect_analysis_add_kc: List[SubQuestion]
    question: str
    analysis: str
    question_kc: str
    version: str


class DataFormater(object):

    def __init__(self,
                 file_path="/Users/tuo/PycharmProjects/math_gpt/question_step/tmp/math/junior/source3_sample_basic_test.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            self.input_data = json.load(f)

    def format_data(self):
        pass

    def __to_json_file(self):
        pass


if __name__ == "__main__":
    df = DataFormater()
    print(df.input_data)
    print(type(df.input_data))
    print("done.")
