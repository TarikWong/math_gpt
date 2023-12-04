# -*- coding: utf-8 -*-
# @Time : 2023/10/19 6:19 下午
# @Author : tuo.wang
# @Version : 基础版本(线上刷数据)
# @Function : 知识点打标
import concurrent.futures
import json
import re
import warnings
from typing import *
from loguru import logger
# from call_gpt import send_chat_request, send_chat_request_async, run_multi_send_chat_request_async
import kc_handler
from process_data import *
from utils import obj_to_dict
from config import ConfigParser
import datetime
import time
import requests
import concurrent.futures
from tqdm import tqdm
from call_gpt.utils.openai_utils import get_engine_config
import random

warnings.filterwarnings("ignore")
cp = ConfigParser()
# config = cp.get_config(input_file="source3_sample_input.csv", output_file="source3_sample_input.json", env="本地", sample_cnt=5, dir_check=True)
config = cp.get_config(input_file="source1.csv", output_file="source1.json", env="线上", sample_cnt=5, dir_check=True)

logger.add(config["log_file"])  ## 日志
INPUT_FILE = config["input_file"]  ## 输入文件
OUT_SOLUTIONS = "{}{}_solutions.json".format(config["output_dir"], config["output_file_name"].split(".")[0])
OUT_SUB = "{}{}_sub.json".format(config["output_dir"], config["output_file_name"].split(".")[0])
OUT_RESULT = config["output_file_name"]
OUTPUT_DIR = config["output_dir"]
SAMPLE_CNT = config["sample_cnt"]

base_url = "http://apx-api.tal.com"  # 替换为你的 API 的基础 URL

TASK_STATUS_CODES = {1: "等待中", 2: "运行中", 3: "已完成", 4: "客户端失败", 5: "服务内部异常"}
RUNNING_CODES = {1: "等待中", 2: "运行中"}
FINISH_CODES = {4: "客户端失败", 5: "服务内部异常"}


def get_headers(engine):
    api_config = get_engine_config(engine)
    return {
        "api-key": f"{api_config.get('api_key')}",
        "Content-Type": "application/json; charset=utf-8",
        "Encoding": "utf-8",
        "x-apx-task-priority": "8",
    }


def send_chat_request_async(
        system,
        examples,
        question,
        engine="GPT4",
        temperature=0,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        return_when_finished=True,
        **kwargs,
):
    """调用 AsyncChat 接口"""
    messages = [{"role": "system", "content": system}]
    messages += examples
    messages.append({"role": "user", "content": question})

    data = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
    }

    url = f"{base_url}/v1/async/chat?api-version=2023-05-15"
    response = requests.post(url, headers=get_headers(engine), json=data).json()
    # logger.info(json.dumps(response, ensure_ascii=False))
    if return_when_finished:  # 则等待完成了再返回结果
        i = 0
        sleep_time = 60
        while True and i < 3600 // sleep_time * 24 * 10:  # 最多等待 10 天
            print(f"Start waiting for task {response['id']} to finish...")
            i += 1
            time.sleep(
                sleep_time + random.randint(-sleep_time // 2, sleep_time // 2)
            )  # 增加随机性，避免同时请求
            result = get_result_by_id(response["id"], engine=engine, delete=True)
            if result is not None:
                if result["status"] not in RUNNING_CODES.keys():
                    # logger.info(json.dumps(result, ensure_ascii=False))
                    return result
                print(
                    f"Waiting for task {response['id']} to finish..., {i * sleep_time} seconds passed"
                )
        return {"error": "Timeout"}
    else:
        return response


def get_result_by_id(id, engine="GPT4", delete=False):
    url = f"{base_url}/v1/async/results/{id}"
    response = None
    try:
        response = requests.get(url, headers=get_headers(engine)).json()
    except ValueError as e:
        logger.info("get_result_by_id_list's JSONDecodeError")
    finally:
        if delete and response is not None:
            finish_and_delete(response, engine=engine)
        return response


def finish_and_delete(data, engine="GPT4"):
    """完成对结果的处理后（比如入库），调用 AckResult 对结果进行确认"""
    id = data["id"]
    task_status = data["status"]
    if task_status not in RUNNING_CODES:  # 任务已完成或者失败了
        delete_result(id, engine)
        print(f"Deleted task {id} with status {TASK_STATUS_CODES[task_status]}")


def get_result_by_id_list(id_list, engine="GPT4", delete=False):
    if len(id_list) > 100:
        raise Exception("id_list length should be less than 100")
    url = f"{base_url}/v1/async/results/detail"

    data = {"task_ids": id_list}
    response = None
    try:
        response = requests.post(url, headers=get_headers(engine), json=data).json()
    except ValueError as e:
        logger.info("get_result_by_id_list's  JSONDecodeError.")
    finally:
        if delete and response is not None:
            for id in response["data"]:
                data = response["data"][id]
                finish_and_delete(data, engine=engine)
        return response


def delete_result(id, engine="GPT4"):
    """调用 AckResult 接口"""
    url = f"{base_url}/v1/async/results/{id}"

    response = requests.delete(url, headers=get_headers(engine))
    return response.status_code


def run_multi_send_chat_request_async(
        all_input_data, n_jobs, add_input=True, return_when_finished=True
):
    num_workers = min(n_jobs, len(all_input_data))
    print(f"all_input_data length: {len(all_input_data)}")
    print(f"num_workers: {num_workers}")

    def send_chat_request_wrapper(data):
        response = send_chat_request_async(
            data["system"],
            data["examples"],
            data["question"],
            engine=data.get("engine", "GPT35"),
            temperature=data.get("temperature", 1),
            max_tokens=data.get("max_tokens", 1024),
            top_p=data.get("top_p", 1),
            frequency_penalty=data.get("frequency_penalty", 0.0),
            presence_penalty=data.get("presence_penalty", 0.0),
            stop=data.get("stop", []),
            max_retry=data.get("max_retry", 10),
            return_when_finished=return_when_finished,
        )
        if add_input:
            response["input"] = data
        return response

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(send_chat_request_wrapper, all_input_data),
                total=len(all_input_data),
            )
        )

    return results


def get_result_by_page(page=1, engine="GPT4", limit=100):
    url = f"{base_url}/v1/async/results?page={page}&limit={limit}"
    response = requests.get(url, headers=get_headers(engine)).json()
    return response


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
        # response = send_chat_request(
        #     system=system,
        #     examples=examples,
        #     question=query,
        #     engine="GPT4",
        # )
        try:
            kc_result = []
            if response["response"] is not None:
                # kc_result = response["response"]["choices"][0]["message"]["content"]
                kc_result = json.loads(response["response"])
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
        # response = send_chat_request(
        #     system=system,
        #     examples=examples,
        #     question=query,
        #     engine="GPT4",
        # )
        try:
            kc_result = []
            if "response" in response.keys() and response["response"] is not None:
                kc_result = json.loads(response["response"]["choices"][0]["message"]["content"])
                # kc_result = json.loads(response["response"])
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
        # response = send_chat_request(
        #     system=system,
        #     examples=examples,
        #     question=query,
        #     engine="GPT4",
        # )
        try:
            kc_result = []
            if "response" in response.keys() and response["response"] is not None:
                kc_result = json.loads(response["response"]["choices"][0]["message"]["content"])
                # kc_result = json.loads(response["response"])
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

        logger.info(repr("first: {}".format(first)))
        q.add_first_step(first)
        for sub_question_idx in range(len(q.sub_question)):
            new_q = Question(
                question_id=q.question_id,
                source=q.source,
                subject_id=q.subject_id,
                grade_id=q.grade_id,
                info=q.info,
                combine_content=q.combine_content,
                sub_question=[q.sub_question[sub_question_idx]],
            )
            second = LastLevel(sub_level_kc=first).generate(sample=new_q, question_id=q.question_id)  # 调用openai接口
            second = self.modify_second_response(second_response=second, first_response=first)

            logger.info(repr("second: {}".format(second)))
            q.add_kc_key(sub_question=q.sub_question[sub_question_idx], kc_list=second)
        logger.info(json.dumps(q, ensure_ascii=False, default=obj_to_dict))
        end_time = time.time()  # 记录代码结束执行的时间
        execution_time = (end_time - start_time) / 60  # 计算代码执行的时间
        logger.info(repr(f"代码执行时间为: {execution_time} 分钟"))
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

        logger.info("len(data): {}".format(len(data)))
        # 分批次打标签
        counter = 30000
        while counter < len(data):
            start = counter
            end = counter + 10000
            self.parallel_execution(data_list=data[start: end], n_jobs=3000)
            to_json_file(
                file_name="{}{}_batch_{}.json".format(OUTPUT_DIR, self.data_processor.out_tmp_result.split(".")[0],
                                                      end), obj=data[start: end])
            counter += 10000
            logger.info("counter: {}".format(counter))


def pretty_print(**args_dict):
    for k, v in args_dict.items():
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
    logger.info("done.")
