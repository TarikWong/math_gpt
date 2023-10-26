# -*- coding: utf-8 -*-
# @Time : 2023/10/25 6:24 下午
# @Author : tuo.wang
# @Version : 
# @Function :
import pandas as pd
import json
from typing import *


class DataFormatProcessing(object):
    def __init__(self,
                 sample_cnt: int = 3,
                 input_file: str = "/Users/tuo/PycharmProjects/math_gpt/question_step/tmp/source2_sample2000_result.json",
                 output_file: str = "/Users/tuo/PycharmProjects/math_gpt/question_step/tmp/source2_sample_result.xlsx",
                 ):
        self.lastlevel_kc_dict = {}
        self.output_list = []
        self.sample_cnt = sample_cnt
        self.output_file = output_file
        with open(input_file, "r", encoding="utf-8") as f:
            self.input_list = json.load(f)

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
        for i in self.input_list:
            i["sub_question_id"] = self.check_sub_question(i["sub_question"])
            i["sub_question"] = self.new_sub_question(i["sub_question"])
            i.pop("info")
            if "first" in i:
                for first_kc in i["first"]:
                    self.lastlevel_kc_dict[first_kc] = 0

        # 根据知识点抽样
        for i in self.input_list:
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


if __name__ == "__main__":
    dfp = DataFormatProcessing()
    dfp.data_process()
    print("done.")
