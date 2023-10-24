# -*- coding: utf-8 -*-
# @Time : 2023/10/24 11:13 上午
# @Author : tuo.wang
# @Version : 
# @Function :
import json
import re
import warnings
from typing import *
from loguru import logger
from call_gpt import send_chat_request
from kc_handler import Knowledge
from process_data import DataProcessor
from utils import obj_to_dict

warnings.filterwarnings("ignore")
logger.add("./log/300.log")


class Level:
    def __init__(self, system: str, example: str) -> None:
        self.system = system
        self.example = example
        self.knowledge = Knowledge()

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

    def generate(self, sample):
        system = self.load_system()
        # print("system:", system)
        examples = self.load_examples()
        # print("example:", examples)
        query = json.dumps(sample, ensure_ascii=False, default=obj_to_dict)
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
            system="/mnt/pfs/zitao_team/shiluyou/question_step/prompts/system/step-1.md",
            example="/mnt/pfs/zitao_team/shiluyou/question_step/prompts/examples/example-1.md",
    ) -> None:
        super().__init__(system=system, example=example)

    def load_system(self):
        system = open(self.system, mode="r").read()
        return system.format(kc_string=self.sub_level_kc_set)


class LastLevel(Level):
    def __init__(
            self,
            system="/mnt/pfs/zitao_team/shiluyou/question_step/prompts/system/step-2.md",
            example="/mnt/pfs/zitao_team/shiluyou/question_step/prompts/examples/example-2.md",
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


class Service:
    def __init__(self, sample_cnt: int = 300) -> None:
        self.sub_level = SubLevel()
        self.sample_cnt = sample_cnt
        self.data_processor = DataProcessor(sample_cnt=sample_cnt)

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

    def run(self, cnt: int = 5, is_new: bool = True):
        data = None
        tmp = 0
        if is_new:
            data = self.data_processor.generate_question()
        else:
            data = self.data_processor.load_json()
        for sample in data:
            first = self.sub_level.generate(sample=sample)
            first = self.modify_first_response(first_response=first)
            print("first:", first)
            sample.add_first_step(first)
            second = LastLevel(sub_level_kc=first).generate(sample=sample)
            second = self.modify_second_response(
                second_response=second, first_response=first
            )
            print("second:", second)
            sample.add_kc_key(second)
            logger.info(json.dumps(sample, ensure_ascii=False, default=obj_to_dict))
            tmp += 1
            # if tmp == cnt:
            #     break
            print(tmp)


if __name__ == "__main__":
    server = Service()
    server.run()
    # server.run(is_new=False)
