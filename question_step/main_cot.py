# -*- coding: utf-8 -*-
# @Time : 2023/10/19 6:19 下午
# @Author : tuo.wang
# @Version : CoT版本（包含推理过程）
# @Function : 知识点打标
import concurrent.futures
import json
import re
import warnings
from typing import *
from loguru import logger
from call_gpt import send_chat_request
import kc_handler
from process_data import *
from utils import obj_to_dict
from config import ConfigParser

warnings.filterwarnings("ignore")
cp = ConfigParser()
config = cp.get_config(input_file="source3_sample_input.csv", output_file="source3_sample_cot.json", env="本地",
                       sublevel_system_file="system-sublevel-style02.md",
                       lastlevel_system_file="system-lastlevel-style02.md",
                       sublevel_examples_list=["example-cot-sublevel-01.md", "example-cot-sublevel-02.md"],
                       lastlevel_examples_list=["example-cot-lastlevel-01.md", "example-cot-lastlevel-02.md"],
                       version="推理过程", sample_cnt=50, dir_check=True)

logger.add(config["log_file"])  ## 日志
INPUT_FILE = config["input_file"]  ## 输入文件
OUT_SOLUTIONS = "{}{}_solutions.json".format(config["output_dir"], config["output_file_name"].split(".")[0])
OUT_SUB = "{}{}_sub.json".format(config["output_dir"], config["output_file_name"].split(".")[0])
OUT_RESULT = config["output_file_name"]
OUTPUT_DIR = config["output_dir"]
SAMPLE_CNT = config["sample_cnt"]


class Level:
    def __init__(self, system: str, examples: list) -> None:
        self.system = system
        self.examples_list = examples
        self.knowledge = kc_handler.Knowledge()

    def load_examples(self):
        """加载few-shot文件."""
        examples = []
        for e in self.examples_list:
            example_md = open(e).read()
            user, assistant = self.find_code_blocks(example_md)
            examples.append({"role": "user", "content": user})
            examples.append({"role": "assistant", "content": assistant})
        return examples

    @staticmethod
    def find_code_blocks(md_text):
        """提取few-shot中的代码块."""
        pattern = r"```json(.+?)```"
        code_blocks = re.findall(pattern, md_text, re.DOTALL)
        return code_blocks

    def load_system(self):
        raise NotImplementedError

    def generate(self, sample, question_id):
        system = self.load_system()
        examples = self.load_examples()
        query = json.dumps(sample, ensure_ascii=False, default=obj_to_dict)
        # print("system:", system)
        # print("examples:", examples)
        # print("query:", query)
        response = send_chat_request(
            system=system,
            examples=examples,
            question=query,
            engine="GPT4",
            # engine="GPT4-FAST",
        )
        try:
            kc_result = json.loads(response["response"])
            assert type(kc_result) is list, "GPT4 response failed!"
            return kc_result
        except json.decoder.JSONDecodeError as e:
            logger.info("error")
            return []

    @property
    def sub_level_kc_set(self):
        return set(self.knowledge.sub_level_kc)

    @property
    def last_level_kc_set(self):
        return set(self.knowledge.last_level_kc)


class SubLevel(Level):
    def __init__(
            self,
            system=config["sublevel_system"],
            examples=config["sublevel_examples"],
    ) -> None:
        super().__init__(system=system, examples=examples)

    def load_system(self):
        system = open(self.system, mode="r").read()
        return system.format(kc_string=self.sub_level_kc_set)

    def generate(self, sample, question_id):
        system = self.load_system()
        examples = self.load_examples()
        query = json.dumps(sample, ensure_ascii=False, default=obj_to_dict)

        # messages = [{"role": "system", "content": system}]
        # messages += examples
        # messages.append({"role": "user", "content": query})

        # print("system:", system)
        # print("first examples:", examples)
        # print("query:", query)
        # print("messages: ", messages)
        # print("===")

        response = send_chat_request(
            system=system,
            examples=examples,
            question=query,
            engine="GPT4",
            # engine="GPT4-FAST",
        )
        try:
            kc_result = json.loads(response["response"])
            print("question_id: '{}', SubLevel kc_result: {}".format(question_id, kc_result))
            # assert type(kc_result) is list, "GPT4 response failed!"
            assert type(kc_result) is dict, "GPT4 response failed!"
            return kc_result
        except json.decoder.JSONDecodeError as e:
            logger.info("SubLevel generate error")
            return []


class LastLevel(Level):
    def __init__(
            self,
            system=config["lastlevel_system"],
            examples=config["lastlevel_examples"],
            sub_level_kc: List[str] = None,
    ) -> None:
        super().__init__(system=system, examples=examples)
        self.sub_list = sub_level_kc
        self.kc_mapping = self.knowledge.mapping

    def load_system(self):
        system = open(self.system, mode="r").read()
        return system.format(kc_string=self.fetch_last_kc_list())

    def fetch_last_kc_list(self) -> List[str]:
        kc_list = []
        for sub_kc in self.sub_list:
            kc_list.extend(self.kc_mapping[sub_kc])
        return kc_list

    def generate(self, sample, question_id):
        system = self.load_system()
        examples = self.load_examples()
        query = json.dumps(sample, ensure_ascii=False, default=obj_to_dict)

        # messages = [{"role": "system", "content": system}]
        # messages += examples
        # messages.append({"role": "user", "content": query})

        # print("system:", system)
        # print("second examples:", examples)
        # print("query:", query)
        # print("messages: ", messages)
        # print("===")

        response = send_chat_request(
            system=system,
            examples=examples,
            question=query,
            engine="GPT4",
            # engine="GPT4-FAST",
        )
        try:
            kc_result = json.loads(response["response"])
            print("question_id: '{}', LastLevel kc_result: {}".format(question_id, kc_result))
            # assert type(kc_result) is list, "GPT4 response failed!"
            assert type(kc_result) is dict, "GPT4 response failed!"
            return kc_result
        except json.decoder.JSONDecodeError as e:
            logger.info("LastLevel generate error")
            return []


class Service:
    def __init__(self, file_path: str, sample_cnt: int, out_tmp: str, out_tmp_sub: str, out_tmp_result: str) -> None:
        self.sub_level = SubLevel()
        self.sample_cnt = sample_cnt
        self.data_processor = DataProcessor(
            file_path=file_path,
            sample_cnt=sample_cnt,
            out_tmp=out_tmp,
            out_tmp_sub=out_tmp_sub,
            out_tmp_result=out_tmp_result)

    # def modify_first_response(self, first_response):
    #     return [
    #         item for item in first_response if item in self.sub_level.sub_level_kc_set
    #     ]
    def modify_first_response(self, first_response):
        for kc in first_response["kc"]:
            if kc not in self.sub_level.sub_level_kc_set:
                first_response["kc"].remove(kc)
                del first_response["reason"][kc]
        return first_response

    # def modify_second_response(self, second_response, first_response):
    #     for item in second_response:
    #         for k, v in item.items():
    #             item[k] = list(set(list(v)) & self.sub_level.last_level_kc_set)
    #             if not item.values:
    #                 item.values = first_response
    #     return second_response
    def modify_second_response(self, second_response):
        for idx in range(len(second_response["kc"])):
            for step_id, last_level_kcs in second_response["kc"][idx].items():
                for kc in set(list(last_level_kcs)):
                    if kc not in self.sub_level.last_level_kc_set:
                        second_response["kc"][idx][step_id].remove(kc)
                        del second_response["reason"][idx][step_id][kc]
        return second_response

    def predict_kc(self, q: Question):
        first = self.sub_level.generate(sample=q, question_id=q.question_id)  # 调用openai接口
        first = self.modify_first_response(first_response=first)
        print("first:", first)
        q.add_first_step(first)
        second = LastLevel(sub_level_kc=first["kc"]).generate(sample=q, question_id=q.question_id)  # 调用openai接口
        second = self.modify_second_response(second_response=second)
        print("second:", second)
        q.add_kc_key(second["kc"])
        q.add_second_step(second)
        logger.info(json.dumps(q, ensure_ascii=False, default=obj_to_dict))

    def parallel_execution(self, data_list, n_jobs=50):
        n_jobs = min(n_jobs, len(data_list))
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
            executor.map(self.predict_kc, data_list)

    def run(self, cnt: int = 5, is_new: bool = True):
        data = None
        if is_new:
            data = self.data_processor.generate_question()
        else:
            data = self.data_processor.load_json()

        print("len(data): ", len(data))
        # print(data)
        self.parallel_execution(data_list=data)
        # 最后结果写入json文件...
        to_json_file(file_name="{}{}".format(OUTPUT_DIR, self.data_processor.out_tmp_result), obj=data)


def pretty_print(**args_dict):
    for k, v in args_dict.items():
        print("参数 {}:".format(k))
        print("--------------------")
        print(v)
        print("--------------------\n")
    print("================ main接收输入参数打印完成 ================")


if __name__ == "__main__":
    pretty_print(INPUT_FILE=INPUT_FILE, SAMPLE_CNT=SAMPLE_CNT, OUTPUT_DIR=OUTPUT_DIR, OUT_RESULT=OUT_RESULT)

    server = Service(
        file_path=INPUT_FILE,
        sample_cnt=SAMPLE_CNT,
        out_tmp=OUTPUT_DIR,
        out_tmp_sub=OUT_SUB,
        out_tmp_result=OUT_RESULT)
    server.run()
    print('done.')
