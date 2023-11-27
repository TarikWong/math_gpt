# -*- coding: utf-8 -*-
# @Time : 2023/11/27 11:32 上午
# @Author : tuo.wang
# @Version : 
# @Function : 打标结果数据标准化
import pandas as pd
import json
from typing import *
from dataclasses import dataclass
from utils import obj_to_dict


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
    question_id: str
    grade_id: int
    subjuect_id: int
    combine_content: str
    combine_analysis: str
    perfect_analysis_add_kc: List[SubQuestion]
    question_kc: str
    version: str


class DataFormater(object):

    def __init__(self,
                 file_path="/Users/tuo/PycharmProjects/math_gpt/question_step/tmp/math/junior/source3_sample_basic_test.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            self.input_data_list = json.load(f)

    def format_data(self, version="v1.0"):
        standard_data_list = []

        for i in self.input_data_list:
            sub_question_list = []
            for sub_question in i["sub_question"]:
                steps = []
                for step in sub_question:
                    si = StepInfo(step=step["solution"]["step"],
                                  title=step["solution"]["title"],
                                  detail=step["solution"]["detail"],
                                  kc=step["kc"][step["solution"]["step"]])
                    steps.append(si)
                sb = SubQuestion(question=sub_question[0]["question"], solution=steps)
                sub_question_list.append(sb)

            rd = ResultData(question_id=i["question_id"],
                            grade_id=None,
                            subjuect_id=None,
                            combine_content=i["combine_content"],
                            combine_analysis=None,
                            perfect_analysis_add_kc=sub_question_list,
                            question_kc=i["first"],
                            version=version)
            standard_data_list.append(rd)
        return standard_data_list

    def __to_json_file(self):
        pass


if __name__ == "__main__":
    df = DataFormater()
    print(json.dumps(df.format_data(), indent=2, ensure_ascii=False, default=obj_to_dict))

    print("done.")
