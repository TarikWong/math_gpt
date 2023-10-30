# -*- coding: utf-8 -*-
# @Time : 2023/10/25 6:24 下午
# @Author : tuo.wang
# @Version : 
# @Function :
import pandas as pd
import json
from typing import *
from dataclasses import dataclass
from utils import obj_to_dict

INPUT_FILE = "/Users/tuo/PycharmProjects/math_gpt/question_step/tmp/source2_sample2000_result.json"
OUTPUT_EXCEL_FILE = "/Users/tuo/PycharmProjects/math_gpt/question_step/tmp/source2_sample_result.xlsx"
OUTPUT_JSON_FILE = "/Users/tuo/PycharmProjects/math_gpt/question_step/tmp/question_step.json"
SAMPLE_CNT = 3


@dataclass
class SolutionItem:
    step: str
    title: str
    detail: str


@dataclass
class SubQuestion:
    question: str
    solution: SolutionItem
    kc: dict = None


@dataclass
class Question:
    question_id: str
    source: str
    subject_id: str
    info: List[str]
    combine_content: str
    # sub_question: List[SubQuestion]
    sub_question_id: str
    sub_solution: SolutionItem
    sub_kc: str


class DataFormatProcessing(object):
    def __init__(self, sample_cnt: int, input_file: str, output_file: str):
        self.question_list = []  ## 每条数据解析成子题形式
        self.lastlevel_kc_dict = {}
        self.output_list = []
        self.sample_cnt = sample_cnt
        self.output_file = output_file
        with open(input_file, "r", encoding="utf-8") as f:
            self.input_json_list = json.load(f)
        for q in self.input_json_list:
            if len(q["sub_question"][0]) > 0:
                for sq in q["sub_question"][0]:
                    sub_question_kc = "unknown"
                    sq_kc_dict = sq["kc"]
                    if sq_kc_dict != None:
                        sq_kc_dict_key = sq["solution"]["step"]
                        sq_kc_dict_value = sq_kc_dict[sq_kc_dict_key]
                        if len(sq_kc_dict_value) > 0:
                            sub_question_kc = "，".join(sq_kc_dict_value)
                    self.question_list.append(
                        Question(question_id=q["question_id"],
                                 source=q["source"],
                                 subject_id=q["subject_id"],
                                 info=q["info"],
                                 combine_content=q["combine_content"],
                                 sub_question_id=sq["question"],
                                 sub_solution=SolutionItem(step=sq["solution"]["step"],
                                                           title=sq["solution"]["title"],
                                                           detail=sq["solution"]["detail"]),
                                 sub_kc=sub_question_kc))

    def new_sub_question(self, sub_question: List[List[Dict]]):
        real_list = sub_question[0]
        new_list = []
        for i in real_list:
            new_dict = i["solution"]
            new_dict["kc"] = i["kc"]
            new_list.append(new_dict)
        return new_list

    def to_string(self, first):
        return str(first)

    def check_sub_question(self, sub_question: List[List[Dict]]):
        real_list = sub_question[0]
        question_id_set = set([])
        for i in real_list:
            question_id_set.add(i["question"])
        if len(question_id_set) == 1:
            return list(question_id_set)[0]
        else:
            return "-"

    def data_process(self):
        # 对数据格式做处理
        for i in self.input_json_list:
            i["sub_question_id"] = self.check_sub_question(i["sub_question"])
            i["sub_question"] = self.new_sub_question(i["sub_question"])
            i.pop("info")
            if "first" in i:
                for first_kc in i["first"]:
                    self.lastlevel_kc_dict[first_kc] = 0

        # 根据知识点抽样
        for i in self.input_json_list:
            counter = 0
            if "first" in i:
                for first_kc in i["first"]:
                    if self.lastlevel_kc_dict[first_kc] < self.sample_cnt:
                        counter += 1
                        self.lastlevel_kc_dict[first_kc] += 1
            if counter > 0:
                self.output_list.append(i)

        df = pd.json_normalize(self.output_list)
        df["first_string"] = df['first'].apply(self.to_string)
        print("去重后数量", len(self.lastlevel_kc_dict))
        # print(self.lastlevel_kc_dict)
        df.to_excel(self.output_file, index=False, encoding='utf-8')


# 生产gpt训练数据
# 过滤知识点为空的的数据
def to_gpt_format(json_list):
    new_json_list = []
    for question_dict in json_list:
        print('question_dict["sub_kc"]: ', question_dict["sub_kc"])
        if "unknown" not in question_dict["sub_kc"]:
            conversations_list = []
            human_dict = {"from": "human", "value": question_dict["sub_solution"]["detail"]}
            gpt_dict = {"from": "gpt", "value": "该题的知识点为：{}。".format(question_dict["sub_kc"])}
            conversations_list.append(human_dict)
            conversations_list.append(gpt_dict)
            question_dict["conversations"] = conversations_list
            new_json_list.append(question_dict)
    return new_json_list


def to_json_file(file_name: str, obj: List[Any], default=obj_to_dict):
    with open(file=file_name, mode="w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=default)


if __name__ == "__main__":
    dfp = DataFormatProcessing(SAMPLE_CNT, INPUT_FILE, OUTPUT_EXCEL_FILE)
    # dfp.data_process()

    # python对象转为json对象
    json_list = json.loads(json.dumps(dfp.question_list, default=obj_to_dict))
    new_json_list = to_gpt_format(json_list)
    df = pd.json_normalize(new_json_list)

    print("调用接口生产的步骤数量：", len(json_list))
    print("步骤过滤知识点为空后数量：", len(new_json_list))
    to_json_file(OUTPUT_JSON_FILE, new_json_list)

    print("done.")
