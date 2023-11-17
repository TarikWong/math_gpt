import os
import json
import random
import pandas as pd
from typing import *
from dataclasses import dataclass
from kc_handler import Knowledge
from utils import obj_to_dict
from config import config_dict

config = config_dict["online"]
DATA_DIR = config["data_dir"]


def split_dataset(
        # 初中数学
        file_dir: str = os.path.join(DATA_DIR, "精美解析结果库_subject_2_grade_group_2.csv")
):
    """将初中数学的全量数据划分为教研云，题拍拍以及其他三部分数据。"""
    data = pd.read_csv(file_dir, low_memory=True)
    data[data["source"] == 1].to_csv(os.path.join(DATA_DIR, "source1.csv"), index=False)
    data[data["source"] == 2].to_csv(os.path.join(DATA_DIR, "source2.csv"), index=False)
    data[data["source"] == 3].to_csv(os.path.join(DATA_DIR, "source3.csv"), index=False)


def get_s1_kc(data: dict):
    """教研云试题数据处理。"""
    info = []
    if not data:
        return info
    tmp = data.get("examOptionList")
    if not tmp:
        return info
    for item in tmp:
        info.extend(dfs(item.get("childList")))
    return info


def get_s1_kc_2(data: dict):
    info = []
    if not data:
        return info
    tmp = data.get("childList")
    if not tmp:
        return info
    for item in tmp:
        st = item.get("examOptionList")
        if not st:
            return info
        for sitem in st:
            info.extend(dfs(sitem.get("childList")))
    return info


def dfs(data: list):
    names = []
    if not data:
        return names
    for item in data:
        if not item["childList"]:
            names.append(item["name"])
        else:
            names.extend(dfs(item["childList"]))
        if "labelKnowList" in item and isinstance(item["labelKnowList"], list):
            names.extend(dfs(item["labelKnowList"]))
    return names


def get_s2_kc(data: dict):
    """题拍拍试题数据处理。"""
    # return [data.get("queSource", "-")]
    return data.get("examOptionList", "-").split(',')


def get_s3_kc(data: dict):
    """外部试题数据处理。"""
    info = []
    kaodian = data.get("kaodian")
    if kaodian:
        info.extend(kaodian.get("value"))
    return info


FUNC_KC_DICT = {
    "1": [get_s1_kc, get_s1_kc_2],
    "2": [get_s2_kc, None],
    "3": [get_s3_kc, None],
}


class SingleSourceData:
    def __init__(self, file_dir: str = "source2.csv", output_dir: str = "source2_sample_2000_add_info.csv",
                 source: str = "2") -> None:
        self.source = source
        self.data = pd.read_csv(os.path.join(DATA_DIR, file_dir))
        self.output_dir = output_dir
        self.func1, self.func2 = FUNC_KC_DICT.get(source)
        self.knowledge = Knowledge().last_level_kc
        # if not self.func1:
        #     raise "fail"

    def parser_func1(self, original):
        """对original字段进行处理,通过处理之后获取有关知识点的信息数据."""
        original = json.loads(original)
        return self.func1(original)

    def parser_func2(self, original):
        """对original字段进行处理,通过处理之后获取有关知识点的信息数据."""
        original = json.loads(original)
        return self.func2(original)

    def filter_last_kc(self, info):
        """处理之后的知识点数据不一定是我们知识点体系中的末级知识点,可能是题型知识点,需要过滤处理."""
        return [item for item in info if item in self.knowledge]

    def concat_list(self, kc):
        return ','.join(kc)

    def format_data(self):
        self.data["original_kc1"] = self.data["original"].apply(self.parser_func1)
        # self.data["original_kc2"] = self.data["original"].apply(self.parser_func2)
        # self.data["original_kc"] = self.data["original_kc1"] + self.data["original_kc2"]
        self.data["original_kc"] = self.data["original_kc1"].apply(lambda x: list(set(x)))
        self.data["new_kc"] = self.data["original_kc"].apply(self.filter_last_kc)
        self.data["new_kc_str"] = self.data["new_kc"].apply(self.concat_list)
        # self.data["new_kc"] = self.data["original_kc"]

    @property
    def new_data(self):
        self.format_data()
        print('format_data done.')
        print(self.data)
        self.data.to_csv(os.path.join(DATA_DIR, self.output_dir), index=False)
        return self.data


@dataclass
class SolutionItem:
    step: str
    title: str
    detail: str


@dataclass
class SubQuestion:
    question: str
    solution: List[dict]
    kc: Optional[List[dict]] = None


@dataclass
class Question:
    question_id: str
    source: str
    subject_id: str
    info: List[str]
    combine_content: str
    sub_question: List[SubQuestion]

    def sample_sub_question(self, cnt: int = 1):
        """随机选择一个子题."""
        sub_question = random.sample(self.sub_question, cnt)
        self.sub_question = sub_question
        return self

    def add_kc_key(self, kc_list):
        for item, kc in zip(self.sub_question[0], kc_list):
            if hasattr(item, "kc"):
                setattr(item, "kc", kc)
            else:
                item["kc"] = kc

    def add_first_step(self, first_step):
        setattr(self, "first", first_step)

    def add_second_step(self, second_step):
        setattr(self, "second", second_step)


def to_json_file(file_name: str, obj: List[Any], default=obj_to_dict):
    with open(file=file_name, mode="w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=default)


class DataProcessor:
    def __init__(
            self,
            file_path: str,
            sample_cnt: int,
            out_tmp: str,
            out_tmp_sub: str,
            out_tmp_result: str) -> None:
        self.data = pd.read_csv(file_path)
        self.out_tmp = "{}{}_solutions.json".format(out_tmp, file_path.split("/")[-1].split(".")[0])
        self.out_tmp_sub = "{}{}_solutions_sub.json".format(out_tmp, file_path.split("/")[-1].split(".")[0])
        self.out_tmp_result = out_tmp_result
        self.sample_cnt = sample_cnt

    def transformer_question(self) -> List[Question]:
        question_list = []
        data = self.data.sample(self.sample_cnt)
        for _, line in data.iterrows():
            question_id = line["question_id"]
            source = line["source"]
            subject_id = line["subject_id"]
            response_json = json.loads(line["response_json"])
            # info = line["new_kc"]
            info = []
            combine_content = line["combine_content"]
            soul = []
            for sub_que in response_json:
                question = sub_que["question"]
                solution = sub_que["solution"]
                sub = []
                for item in solution:
                    si = SolutionItem(**item)
                    sub.append(SubQuestion(question=question, solution=si))
                soul.append(sub)
            question_list.append(
                Question(
                    question_id=question_id,
                    source=source,
                    subject_id=subject_id,
                    info=info,
                    combine_content=combine_content,
                    sub_question=soul,
                )
            )
        # to_json_file(file_name=self.out_tmp, obj=question_list)
        return question_list

    def generate_question(self):
        question_list = []
        data_list = self.transformer_question()
        for data in data_list:
            question_list.append(data.sample_sub_question())
        # to_json_file(file_name=self.out_tmp_sub, obj=question_list)
        return question_list

    def load_json(self):
        with open(file=self.out_tmp_sub, mode="r", encoding="utf-8") as f:
            data = json.load(f)
        return [Question(**item) for item in data]


def calculate_kc(
        data_dir: str = "/mnt/pfs/zitao_team/shiluyou/question_step/data/source1_add_info.csv",
):
    data = pd.read_csv(data_dir)
    print(len(data))
    d1 = sum(data["original_kc"].apply(lambda x: len(eval(x)) == 0))
    d2 = sum(data["new_kc"].apply(lambda x: len(eval(x)) == 0))
    print(d1)
    print(d2)


def sample_data(count: int = 2000, data_type: str = "1"):
    """对某个来源的数据进行采样.
    data_type: 1表示教研云试题,2表示题拍拍试题,3表示外部试题.
    """
    data = pd.read_csv(os.path.join(DATA_DIR, f"source{data_type}_add_info.csv"))
    data = data.sample(count)
    data.to_csv(os.path.join(DATA_DIR, f"source{data_type}_sample_{count}.csv"), index=False)


def check_data(file_dir: str = "source1_sample_2000.csv"):
    data = pd.read_csv(os.path.join(DATA_DIR, file_dir))
    print(data["original_kc"].to_list())
    print(data["new_kc"].to_list())


if __name__ == "__main__":
    # 解析某个字段
    # data = pd.read_csv('/Users/tuo/PycharmProjects/math_gpt/question_step/data/source2_sample20.csv').head(10)
    # print('=========== original start ===========')
    # for _, line in data.iterrows():
    #     print(line['original'])
    #     print('===')
    # print('=========== original end ===========')

    # 随机抽样数据写入临时文件
    # sample_cnt = 5000
    # df = pd.read_csv('/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step/data/source3.csv')
    # data = df.sample(sample_cnt)
    # data.to_csv(os.path.join(DATA_DIR, "source2_sample{}.csv".format(str(sample_cnt))), index=False)

    # 根据测试数据question_id重新抽取原始数据
    input_df = pd.read_csv("/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step/data/source3.csv")
    question_df = pd.read_excel(
        "/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step/tmp/source3_sample_eval_20231111.xlsx")
    question_list = question_df['question_id'].tolist()
    new_df = input_df[input_df['question_id'].isin(question_list)]
    new_df.to_csv(os.path.join(DATA_DIR, "source3_sample_input.csv"), index=False)

    print('done.')
