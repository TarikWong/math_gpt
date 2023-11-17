# -*- coding: utf-8 -*-
# @Time : 2023/10/25 6:24 下午
# @Author : tuo.wang
# @Version : 
# @Function :
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from typing import *
from dataclasses import dataclass
from utils import obj_to_dict

INPUT_FILE = "/Users/tuo/PycharmProjects/math_gpt/question_step/tmp/source3_sample5000_result.json"
OUTPUT_EXCEL_FILE = "/Users/tuo/PycharmProjects/math_gpt/question_step/tmp/source3_sample_eval_20231111.xlsx"
OUTPUT_JSON_FILE = "/Users/tuo/PycharmProjects/math_gpt/question_step/tmp/source3_sample8_first_kc_zero.json"
SAMPLE_CNT = 10
MODEL_MAX_LENGTH = 2048


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
        self.question_list = []  ## 次末级知识点标签不为空的题目数据
        self.lastlevel_kc_dict = {}
        self.output_list = []
        self.first_kc_none_output_list = []
        self.first_kc_none_list = []
        self.sample_question_list = []
        self.sample_cnt = sample_cnt
        self.output_file = output_file
        with open(input_file, "r", encoding="utf-8") as f:
            self.input_json_list = json.load(f)
            print("input_file: ", input_file)
            print("self.input_json_list's length: ", len(self.input_json_list))

        self.first_kc_filter()  ## 过滤次末级知识点标签为空的数据
        for q in self.input_json_list:
            if q["question_id"] in self.sample_question_list:
                self.output_list.append(q)
            if q["question_id"] in self.first_kc_none_list:
                self.first_kc_none_output_list.append(q)
            else:
                self.question_list.append(q)

    def first_kc_filter(self):
        for i in self.input_json_list:
            if "first" in i.keys() and len(i["first"]) > 0:
                for first_kc in i["first"]:
                    self.lastlevel_kc_dict[first_kc] = 0
            else:
                self.first_kc_none_list.append(i["question_id"])
        print("次末级知识点为空的题目数量: ", len(self.first_kc_none_list))

        # 根据知识点抽样
        for i in self.input_json_list:
            sub_question_null_counter = 0
            for sub_question in i["sub_question"][0]:
                if sub_question["kc"] == None:
                    sub_question_null_counter += 1
            if "first" in i.keys() and len(i["first"]) > 0 and sub_question_null_counter == 0:
                distinct_set = set([0])
                for first_kc in i["first"]:
                    distinct_set.add(self.lastlevel_kc_dict[first_kc])
                    if self.lastlevel_kc_dict[first_kc] < self.sample_cnt:
                        self.lastlevel_kc_dict[first_kc] += 1
                if max(distinct_set) < self.sample_cnt:
                    self.sample_question_list.append(i["question_id"])
        print("该批数据的末级知识点标签的数量：", len(self.lastlevel_kc_dict.keys()))
        print("按知识点抽样的题目数量：", len(self.sample_question_list))


# 生产gpt训练数据
def to_gpt_format(question_list, model_max_length=2048):
    return_question_list = []
    for question in question_list:
        for step in question["sub_question"][0]:
            step.pop("kc")
        first_kc = question.pop("first")
        conversations_list = []
        hunman_question = json.dumps(question, ensure_ascii=False, default=obj_to_dict)
        gpt_qnswer = json.dumps(first_kc, ensure_ascii=False, default=obj_to_dict)
        if len(hunman_question) < model_max_length:
            print("len(hunman_question): ", len(hunman_question))
            human_dict = {"from": "human", "value": hunman_question}
            gpt_dict = {"from": "gpt", "value": gpt_qnswer}
            conversations_list.append(human_dict)
            conversations_list.append(gpt_dict)
            question["conversations"] = conversations_list
            return_question_list.append(question)
    print("return_question_list's length: ", len(return_question_list))
    return return_question_list


def to_json_file(file_name: str, obj: List[Any], default=obj_to_dict):
    with open(file=file_name, mode="w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=default)


def json_format(col):
    return json.dumps(col, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    dfp = DataFormatProcessing(SAMPLE_CNT, INPUT_FILE, OUTPUT_EXCEL_FILE)

    # python对象转为json对象
    json_list = json.loads(json.dumps(dfp.output_list, default=obj_to_dict))
    df = pd.json_normalize(json_list)
    df["sub_question"] = df["sub_question"].apply(json_format)
    df.to_excel(OUTPUT_EXCEL_FILE, index=False, encoding='utf-8')

    # to_json_file(OUTPUT_JSON_FILE, obj=dfp.first_kc_none_output_list)

    # 把数据处理成gpt训练语料数据
    # result_list = to_gpt_format(question_list=dfp.question_list, model_max_length=MODEL_MAX_LENGTH)
    # train_list, test_list = train_test_split(result_list, test_size=0.1, random_state=123)
    # print("train_list's length: ", len(train_list))
    # print("test_list's length: ", len(test_list))
    # to_json_file("/Users/tuo/PycharmProjects/math_gpt/question_step/tmp/source2_sample8_train_gpt.json", train_list)
    # to_json_file("/Users/tuo/PycharmProjects/math_gpt/question_step/tmp/source2_sample8_test_gpt.json", test_list)

    print("done.")
