# -*- coding: utf-8 -*-
# @Time : 2023/10/19 6:19 下午
# @Author : tuo.wang
# @Version : 
# @Function :
# cat main.py
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
from config import config_dict

warnings.filterwarnings("ignore")
config = config_dict["online"]

logger.add("{}{}.log".format(config["log_dir"], config["input_file_name"].split(".")[0]))  ## 日志文件目录
INPUT_FILE = "{}{}".format(config["input_dir"], config["input_file_name"])  ## 输入文件
OUT_SOLUTIONS = "{}{}_solutions.json".format(config["output_dir"], config["input_file_name"].split(".")[0])  ## 输出文件
OUT_SUB = "{}{}_sub.json".format(config["output_dir"], config["input_file_name"].split(".")[0])
OUT_RESULT = "{}_result.json".format(config["input_file_name"].split(".")[0])
OUTPUT_DIR = config["output_dir"]
SAMPLE_CNT = config["sample_cnt"]


class Level:
    def __init__(self, system: str, example: str) -> None:
        self.system = system
        self.example = example
        self.knowledge = kc_handler.Knowledge()

    def load_examples(self):
        """加载few-shot文件."""
        examples = []
        example_md = open(self.example).read()
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
        # print("example:", examples)
        # print("query:", query)
        response = send_chat_request(
            system=system,
            examples=examples,
            question=query,
            engine="GPT4",
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
            example=config["sublevel_example"],
    ) -> None:
        super().__init__(system=system, example=example)

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
        #
        # print("system:", system)
        # print("example:", examples)
        # print("query:", query)
        # print("messages: ", messages)
        # print("===")

        response = send_chat_request(
            system=system,
            examples=examples,
            question=query,
            engine="GPT4",
        )
        try:
            kc_result = json.loads(response["response"])
            print("question_id: '{}', SubLevel kc_result: {}".format(question_id, kc_result))
            assert type(kc_result) is list, "GPT4 response failed!"
            return kc_result
        except json.decoder.JSONDecodeError as e:
            logger.info("error")
            return []


class LastLevel(Level):
    def __init__(
            self,
            system=config["lastlevel_system"],
            example=config["lastlevel_example"],
            sub_level_kc: List[str] = None,
    ) -> None:
        super().__init__(system=system, example=example)
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
        #
        # print("system:", system)
        # print("example:", examples)
        # print("query:", query)
        # print("messages: ", messages)
        # print("===")

        response = send_chat_request(
            system=system,
            examples=examples,
            question=query,
            engine="GPT4",
        )
        try:
            kc_result = json.loads(response["response"])
            print("question_id: '{}', LastLevel kc_result: {}".format(question_id, kc_result))
            assert type(kc_result) is list, "GPT4 response failed!"
            return kc_result
        except json.decoder.JSONDecodeError as e:
            logger.info("error")
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

    def modify_first_response(self, first_response):
        return [
            item for item in first_response if item in self.sub_level.sub_level_kc_set
        ]

    def modify_second_response(self, second_response, first_response):
        for item in second_response:
            for k, v in item.items():
                item[k] = list(set(list(v)) & self.sub_level.last_level_kc_set)
                if not item.values:
                    item.values = first_response
        return second_response

    def predict_kc(self, q: Question):
        first = self.sub_level.generate(sample=q, question_id=q.question_id)  # 调用openai接口
        first = self.modify_first_response(first_response=first)
        print("first:", first)
        q.add_first_step(first)
        second = LastLevel(sub_level_kc=first).generate(sample=q, question_id=q.question_id)  # 调用openai接口
        second = self.modify_second_response(
            second_response=second, first_response=first
        )
        print("second:", second)
        q.add_kc_key(second)
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
        self.parallel_execution(data_list=data)
        # 最后结果写入json文件...
        to_json_file(file_name="{}{}".format(OUTPUT_DIR, self.data_processor.out_tmp_result), obj=data)


if __name__ == "__main__":
    parameters = """INPUT_FILE: {}\nSAMPLE_CNT: {}\nOUTPUT_DIR: {}\nOUT_SUB: {}\nOUT_RESULT: {}""".format(INPUT_FILE,
                                                                                                          SAMPLE_CNT,
                                                                                                          OUTPUT_DIR,
                                                                                                          OUT_SUB,
                                                                                                          OUT_RESULT)
    print(parameters)
    print("####################\n")

    server = Service(
        file_path=INPUT_FILE,
        sample_cnt=SAMPLE_CNT,
        out_tmp=OUTPUT_DIR,
        out_tmp_sub=OUT_SUB,
        out_tmp_result=OUT_RESULT)
    server.run()
    print('done.')
