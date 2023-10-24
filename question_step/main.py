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
from kc_handler import Knowledge
from process_data import *
from utils import obj_to_dict

warnings.filterwarnings("ignore")
logger.add("/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step/log/2000.log")  ## 日志文件目录

INPUT_FILE = "/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step/data/source2_sample_3000.csv"  ## 输入文件
OUT_TMP = "/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step/tmp/source2_sample_2000_solutions.json"  ## 输出文件
OUT_TMP_SUB = "/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step/tmp/source2_sample_2000_sub.json"
OUT_TMP_RESULT = "source2_sample_2000_result.json"
OUTPUT_DIR = '/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step/tmp/'  ## 输出文件目录
SAMPLE_CNT = 2000  ## 从输入文件中随机抽样数量


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
        # print("=== generate start ===")
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
        print("=== generate success ===")
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
            # system="/Users/tuo/PycharmProjects/math_gpt/prompts/system/step-1.md",
            # example="/Users/tuo/PycharmProjects/math_gpt/prompts/examples/example-1.md",
            system="/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step/prompts/system/step-1.md",
            example="/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step/prompts/examples/example-1.md",
    ) -> None:
        super().__init__(system=system, example=example)

    def load_system(self):
        system = open(self.system, mode="r").read()
        return system.format(kc_string=self.sub_level_kc_set)


class LastLevel(Level):
    def __init__(
            self,
            # system="/Users/tuo/PycharmProjects/math_gpt/prompts/system/step-2.md",
            # example="/Users/tuo/PycharmProjects/math_gpt/prompts/examples/example-2.md",
            system="/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step/prompts/system/step-2.md",
            example="/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step/prompts/examples/example-2.md",
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
    def __init__(self,
                 file_path: str = "/Users/tuo/PycharmProjects/math_gpt/question_step/data/source2_sample_20.csv",
                 sample_cnt: int = 1,
                 out_tmp: str = '/Users/tuo/PycharmProjects/math_gpt/question_step/tmp/source2_solutions_20.json',
                 out_tmp_sub: str = "/Users/tuo/PycharmProjects/math_gpt/question_step/tmp/source2_solutions_sub_20.json",
                 out_tmp_result: str = "source2_solutions_result_20.json",
                 ) -> None:
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
        # print("sample's type: ", type(sample))
        # print("sample: ", sample)
        first = self.sub_level.generate(sample=q)  # 调用openai接口
        first = self.modify_first_response(first_response=first)
        print("first:", first)
        q.add_first_step(first)
        second = LastLevel(sub_level_kc=first).generate(sample=q)  # 调用openai接口
        second = self.modify_second_response(
            second_response=second, first_response=first
        )
        print("second:", second)
        q.add_kc_key(second)
        logger.info(json.dumps(q, ensure_ascii=False, default=obj_to_dict))

    def parallel_execution(self, demo_json_list, n_jobs=50):
        n_jobs = min(n_jobs, len(demo_json_list))
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
            executor.map(self.predict_kc, demo_json_list)
        print('done.')

    def run(self, cnt: int = 5, is_new: bool = True):
        data = None
        if is_new:
            data = self.data_processor.generate_question()
        else:
            data = self.data_processor.load_json()

        self.parallel_execution(demo_json_list=data)
        # 最后结果写入json文件...
        to_json_file(file_name="{}{}".format(OUTPUT_DIR, self.data_processor.out_tmp_result), obj=data)


if __name__ == "__main__":
    server = Service(
        file_path=INPUT_FILE,
        sample_cnt=SAMPLE_CNT,
        out_tmp=OUT_TMP,
        out_tmp_sub=OUT_TMP_SUB,
        out_tmp_result=OUT_TMP_RESULT)
    server.run()
    print('done.')
    # server.run(is_new=False)
