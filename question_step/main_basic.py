# -*- coding: utf-8 -*-
# @Time : 2023/10/19 6:19 下午
# @Author : tuo.wang
# @Version : 基础版本
# @Function : 知识点打标
import concurrent.futures
import json
import re
import warnings
from typing import *
from loguru import logger
from call_gpt import send_chat_request, send_chat_request_async, run_multi_send_chat_request_async
import kc_handler
from process_data import *
from utils import obj_to_dict
from config import ConfigParser
import time

warnings.filterwarnings("ignore")
cp = ConfigParser()
config = cp.get_config(input_file="source1.csv", output_file="source1.json", env="线上", sample_cnt=20, dir_check=True)

logger.add(config["log_file"])  ## 日志
INPUT_FILE = config["input_file"]  ## 输入文件
OUT_SOLUTIONS = "{}{}_solutions.json".format(config["output_dir"], config["output_file_name"].split(".")[0])
OUT_SUB = "{}{}_sub.json".format(config["output_dir"], config["output_file_name"].split(".")[0])
OUT_RESULT = config["output_file_name"]
OUTPUT_DIR = config["output_dir"]
SAMPLE_CNT = config["sample_cnt"]


class Level:
    def __init__(self, system: str, example_list: list) -> None:
        self.system = system
        self.example_list = example_list
        self.knowledge = kc_handler.Knowledge()

    def load_examples(self):
        """加载few-shot文件."""
        examples = []
        for e in self.example_list:
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

        response = send_chat_request_async(
            system=system,
            examples=examples,
            question=query,
            engine="GPT4-async-super",
        )
        try:
            if response["response"] is not None:
                kc_result = response["response"]["choices"][0]["message"]["content"]
                print("question_id: '{}', Level kc_result: {}".format(question_id, kc_result))
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
            example_list=config["sublevel_examples"],
    ) -> None:
        super().__init__(system=system, example_list=example_list)

    def load_system(self):
        system = open(self.system, mode="r").read()
        return system.format(kc_string=self.sub_level_kc_set)

    def generate(self, sample, question_id):
        # print("SubLevel generate")
        system = self.load_system()
        examples = self.load_examples()
        query = json.dumps(sample, ensure_ascii=False, default=obj_to_dict)

        messages = [{"role": "system", "content": system}]
        messages += examples
        messages.append({"role": "user", "content": query})

        response = send_chat_request_async(
            system=system,
            examples=examples,
            question=query,
            engine="GPT4-async-super",
        )
        try:
            kc_result = []
            if response["response"] is not None:
                kc_result = json.loads(response["response"]["choices"][0]["message"]["content"])
            # assert type(kc_result) is list, "GPT4 response failed!"
            return kc_result
        except json.decoder.JSONDecodeError as e:
            logger.info("error")
            return []


class LastLevel(Level):
    def __init__(
            self,
            system=config["lastlevel_system"],
            example_list=config["lastlevel_examples"],
            sub_level_kc: List[str] = None,
    ) -> None:
        super().__init__(system=system, example_list=example_list)
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
        # print("LastLevel generate")
        system = self.load_system()
        examples = self.load_examples()
        query = json.dumps(sample, ensure_ascii=False, default=obj_to_dict)

        messages = [{"role": "system", "content": system}]
        messages += examples
        messages.append({"role": "user", "content": query})

        response = send_chat_request_async(
            system=system,
            examples=examples,
            question=query,
            engine="GPT4-async-super",
        )
        try:
            kc_result = []
            if response["response"] is not None:
                kc_result = json.loads(response["response"]["choices"][0]["message"]["content"])
            # assert type(kc_result) is list, "GPT4 response failed!"
            return kc_result
        except json.decoder.JSONDecodeError as e:
            logger.info("error")
            return []


class Service:
    def __init__(self, file_path: str, sample_cnt: int, out_tmp: str, out_tmp_sub: str, out_tmp_result: str,
                 need_sample: bool = True) -> None:
        self.sub_level = SubLevel()
        self.sample_cnt = sample_cnt
        self.data_processor = DataProcessor(
            file_path=file_path,
            sample_cnt=sample_cnt,
            out_tmp=out_tmp,
            out_tmp_sub=out_tmp_sub,
            out_tmp_result=out_tmp_result,
            need_sample=need_sample)

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
        start_time = time.time()
        first = self.sub_level.generate(sample=q, question_id=q.question_id)  # 调用openai接口
        first = self.modify_first_response(first_response=first)
        # print("first:", first)
        logger.info("first: {}".format(first))
        q.add_first_step(first)
        for sub_question_idx in range(len(q.sub_question)):
            new_q = Question(
                question_id=q.question_id,
                source=q.source,
                subject_id=q.subject_id,
                info=q.info,
                combine_content=q.combine_content,
                sub_question=[q.sub_question[sub_question_idx]],
            )
            second = LastLevel(sub_level_kc=first).generate(sample=new_q, question_id=q.question_id)  # 调用openai接口
            second = self.modify_second_response(second_response=second, first_response=first)
            # print("second:", second)
            logger.info("second: {}".format(second))
            q.add_kc_key(sub_question=q.sub_question[sub_question_idx], kc_list=second)
        logger.info(json.dumps(q, ensure_ascii=False, default=obj_to_dict))
        end_time = time.time()  # 记录代码结束执行的时间
        execution_time = end_time - start_time  # 计算代码执行的时间
        # print(f"代码执行时间为: {execution_time} 秒")
        logger.info(f"代码执行时间为: {execution_time} 秒")
        return 1

    def parallel_execution(self, data_list, n_jobs=50):
        n_jobs = min(n_jobs, len(data_list))
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(self.predict_kc, data_list))
        # print("results length: ", len(results))
        logger.info("results length: {}".format(len(results)))

    def run(self, cnt: int = 5, is_new: bool = True):
        data = None
        if is_new:
            data = self.data_processor.generate_question()
        else:
            data = self.data_processor.load_json()

        # print("len(data): ", len(data))
        logger.info("len(data): {}".format(len(data)))
        # 分批次打标签
        counter = 0
        while counter < len(data):
            start = counter
            end = counter + 10000
            self.parallel_execution(data_list=data[start: end], n_jobs=2000)
            to_json_file(
                file_name="{}{}_batch_{}.json".format(OUTPUT_DIR, self.data_processor.out_tmp_result.split(".")[0],
                                                      end), obj=data[start: end])
            counter += 10000
            # print("counter: ", counter)
            logger.info("counter: {}".format(counter))


def pretty_print(**args_dict):
    for k, v in args_dict.items():
        # print("参数 {}:".format(k))
        logger.info("参数 {}: {}".format(k, v))
    logger.info("main接收输入参数打印完成。")


if __name__ == "__main__":
    pretty_print(INPUT_FILE=INPUT_FILE, SAMPLE_CNT=SAMPLE_CNT, OUTPUT_DIR=OUTPUT_DIR, OUT_RESULT=OUT_RESULT)

    server = Service(
        file_path=INPUT_FILE,
        sample_cnt=SAMPLE_CNT,
        out_tmp=OUTPUT_DIR,
        out_tmp_sub=OUT_SUB,
        out_tmp_result=OUT_RESULT,
        need_sample=False)
    server.run()
    # print('done.')
    logger.info("done.")
