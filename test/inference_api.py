# -*- coding: utf-8 -*-
# @Time : 2023/11/6 4:48 下午
# @Author : tuo.wang
# @Version : 
# @Function :
"""Inference for FastChat models."""
# from fastchat.train.llama2_flash_attn_monkey_patch import (
#     replace_llama_attn_with_flash_attn,
# )

# replace_llama_attn_with_flash_attn()

import abc
import gc
import math
from typing import Iterable, Optional
import sys
import warnings
import os
import copy
import transformers
from fastchat.model.model_adapter import load_model, get_conversation_template

base_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(base_path, "../"))
from utils.parser import extract_api
# from utils.cal_api_remote import *
from utils.cal_api import *
import json
import jsonlines

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModel,
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
    AutoConfig,
)
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from tqdm import tqdm
import pandas as pd
import argparse
from fastchat.utils import is_partial_stop


def prepare_logits_processor(
        temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


def calculate(cal_json, logger=None):
    # TODO: should fix in future, ugly writing style
    try:
        api_name = cal_json["ActionName"].replace(" ", "")
        api_args = cal_json["Args"]
        result = api_map[api_name](api_args)
        output = {"result": result}
    except Exception as e:
        output = {"result": ""}
        if logger is not None:
            logger.error(f"Error in calculate: {e},input: {cal_json}")
    return output


def find_nearest_thought(generated_text):
    # Split the text at each occurrence of "<thought>"
    segments = generated_text.split("<thought>")

    # Return the last segment (which is the content after the last occurrence of "<thought>")
    return "<thought>" + segments[-1]


@torch.inference_mode()
def generate_stream(
        model, tokenizer, params, device, context_len=2048, stream_interval=2, current_prefix_with_api=5,
        bagging_times=5
):
    print("Generating stream...")
    print(f"Params: {params}")
    use_plugin = params.get("use_plugin", True)
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)
    echo = bool(params.get("echo", True))
    stop_token_ids = params.get("stop_token_ids", None) or []
    stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )

    input_ids = tokenizer(prompt).input_ids
    input_echo_len = len(input_ids)
    output_ids = list(input_ids)
    max_src_len = context_len - max_new_tokens - 8

    input_ids = input_ids[-max_src_len:]

    past_key_values = out = None
    gen_tokens = []
    for i in range(max_new_tokens):
        # print(f"Input ids: {tokenizer.decode(input_ids)}")
        if i == 0 or restart_gen:
            out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values
            restart_gen = False
        else:
            out = model(
                input_ids=torch.as_tensor([[token]], device=device),
                use_cache=True,
                past_key_values=past_key_values,
            )
            logits = out.logits
            past_key_values = out.past_key_values

        if logits_processor:
            if repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
        else:
            last_token_logits = logits[0, -1, :]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-5 or top_p < 1e-8:  # greedy
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        gen_tokens.append(token)
        output_ids.append(token)
        api_token_index = tokenizer.additional_special_tokens.index("</API>")
        api_token_id = tokenizer.additional_special_tokens_ids[api_token_index]

        if token == api_token_id and use_plugin:  # </API>
            print("开始检测API")
            msg = tokenizer.decode(gen_tokens).replace(" ", "")
            api_json = extract_api(msg)
            print(f"api_json is {api_json}")
            if len(api_json) != 0:
                api_json = api_json[-1]
            if api_json is None:
                print("没有检测到任何API")
                pass
            if type(api_json) != dict:
                print(f"API Json不是合法的形式: {api_json}")
            elif "ActionName" not in api_json or "Args" not in api_json:
                print(f"检测到API, 但是参数不完整: {api_json}")
            else:
                print(f"尝试调用 API: {api_json['ActionName']}, 参数是: {api_json['Args']}")
                answer = calculate(api_json)

                if answer is None:
                    print(f"调用 {api_json['ActionName']} 失败, 参数是 {api_json['Args']}")
                else:
                    answer = answer["result"]
                    print(f"{api_json['ActionName']} 结果为 {answer}")
                    answer = f"=> {str(answer)}</thought>"
                    answer_tokens = tokenizer(
                        [answer],
                    ).input_ids[0][1:]
                    gen_tokens.extend(answer_tokens)
                    output_ids.extend(answer_tokens)
                    # 重启生成
                    input_ids += gen_tokens
                    gen_tokens = []
                    restart_gen = True
                    past_key_values = out = None
                    print("重启生成")

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len_prompt
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(
                tmp_output_ids,
                spaces_between_special_tokens=False,
            )
            if stop_str:
                if isinstance(stop_str, str):
                    pos = output.rfind(stop_str, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                elif isinstance(stop_str, Iterable):
                    for each_stop in stop_str:
                        pos = output.rfind(each_stop, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                            break
                else:
                    raise ValueError("Invalid stop field type.")

            yield {
                "text": output,
                "usage": {
                    "prompt_tokens": input_echo_len,
                    "completion_tokens": i,
                    "total_tokens": input_echo_len + i,
                },
                "finish_reason": None,
            }

        if stopped:
            break

    # Finish stream event, which contains finish reason
    if i == max_new_tokens - 1:
        finish_reason = "length"
    elif stopped:
        finish_reason = "stop"
    else:
        finish_reason = None

    yield {
        "text": output,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
    }

    # Clean
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()


def load_json(file_path):
    print("file path here: ", file_path)
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def save_jsonl(file_path, data):
    # 写入JSONL文件
    with jsonlines.open(file_path, mode="w") as writer:
        writer.write_all(data)


def generate_chat_response(model, tokenizer, device, prompt):
    system_template = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"""
    params = {
        "prompt": system_template.format(prompt),
        "temperature": 0.01,
        "top_p": 1.0,
        "max_new_tokens": 1024,
        "use_plugin": True,
        "stream_interval": 1,
    }
    completion = generate_stream(model, tokenizer, params, device)
    for one_text in completion:
        pass
    return one_text


def process_data_with_chat_responses(data, model, tokenizer, device, output_file, dataset_name):
    processed_data = []
    output_file = output_file.replace(".json", ".jsonl")
    fw = open(output_file, "a")
    for item in tqdm(data):
        if "dataset_name" in item and item["dataset_name"] != dataset_name:
            continue
        prompt = item["question"]
        response = generate_chat_response(model, tokenizer, device, prompt)
        item["response"] = response["text"]
        item["raw_response"] = response
        processed_data.append(item)
        print("Raw prompt:", prompt)
        print("Raw answer:", item["answer"])
        print("Generated chat response:", response)
        cols = ["que_id", "kc", "question", "analysis", "response", "answer"]
        if "dataset_name" in item:
            cols.append("dataset_name")
        new_item = {}
        for col in cols:
            new_item[col] = item[col]
        new_item["response"] = response["text"].replace("</s>", "")
        new_item["response_last_10_words"] = new_item["response"][-20:]
        new_item["analysis_last_10_words"] = item["analysis"][-20:]
        fw.write(json.dumps(new_item, ensure_ascii=False) + "\n")
        fw.flush()
    fw.close()
    return processed_data


def generate_chat_responses(model_path, data_file, output_file, args):
    # output dir is
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    device = "cpu" if not torch.cuda.is_available() else "cuda"
    model, tokenizer = load_model(model_path, device=device, num_gpus=2)

    data = load_json(data_file)
    data = [x for x in data if "应用题" in x["kc"]]  # [:2]
    print("Number of samples:", len(data))
    model_name = args.model_name
    dataset_name = args.dataset_name
    processed_data = process_data_with_chat_responses(data, model, tokenizer, device, output_file, dataset_name)
    # save_jsonl(output_file, processed_data)

    # save to excel
    if "dataset_name" in processed_data[0]:
        save_cols = ["que_id", "kc", "dataset_name", "question", "analysis", "response", "answer"]
    else:
        save_cols = ["que_id", "kc", "question", "analysis", "response", "answer"]
    df = pd.DataFrame(processed_data)
    for col in save_cols:
        if col not in df.columns:
            df[col] = ""
    df = df[save_cols]
    df["response"] = df["response"].apply(lambda x: x.replace("</s>", ""))
    df["response_last_10_words"] = df["response"].apply(lambda x: x[-20:])
    df["analysis_last_10_words"] = df["analysis"].apply(lambda x: x[-20:])
    df.to_excel(output_file.replace(".json", ".xlsx"), index=False, engine="xlsxwriter")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="chatglm2-6b", help="Name of the model"
    )
    parser.add_argument("--model_path", type=str,
                        default="/mnt/pfs/zitao_team/tianqiaoliu/public_github/ChatGLM2-6B/ptuning/output/mathgpt-chatglm2-6b-ft-2e-5/checkpoint-POINTNUM")
    parser.add_argument("--data_file", type=str, help="Path to the input data file",
                        default="/mnt/pfs/zitao_team/big_model/processed_data/test_data_junior_small.json")
    parser.add_argument("--output_file", type=str, help="Path to the output file",
                        default='./results/chatglm2-6b/test_data_small_with_response_chatglm2_POINTNUM.json')
    parser.add_argument("--gpu_id", type=str, default="7", help="ID of the GPU to use")
    parser.add_argument("--device_num", type=int, default=1, help="number of gpus to use")
    parser.add_argument("--checkpoint", type=str, default="420")
    parser.add_argument("--no_api", type=bool, default=True)
    parser.add_argument("--dataset_name", type=str, default="GSM8K")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # source_modeling_py_file = "/mnt/pfs/zitao_team/tianqiaoliu/public_github/ChatGLM2-6B/ptuning/output/mathgpt-chatglm2-6b-ft-2e-5/checkpoint-420/modeling_chatglm.py"
    point_num = args.checkpoint

    # cmd_str = "scp {} {}".format(source_modeling_py_file, args.model_path)
    # os.system(cmd_str)
    # tokeniz_files = glob.glob("/mnt/pfs/zitao_team/tianqiaoliu/public_github/ChatGLM2-6B/ptuning/output/mathgpt-chatglm2-6b-ft-2e-5/checkpoint-420/tokeniz*")
    # for one_tokeniz_file in tokeniz_files:
    #     cmd_str = "scp {} {}".format(one_tokeniz_file, os.path.join(args.model_path, "checkpoint-{}".format(str(point_num))))
    #     os.system(cmd_str)
    args.model_path = args.model_path.replace("POINTNUM", str(point_num))
    args.output_file = args.output_file.replace("POINTNUM", str(point_num))
    generate_chat_responses(
        args.model_path,
        args.data_file,
        args.output_file.format(model_name=args.model_name),
        args,
    )

    ### 美丽的分割线

    # model_path = "/mnt/pfs/zitao_team/yinzhibo/llm_team/source/training/sft/train_scripts/models/llama2-70B-api-data-sft/checkpoint-400"
    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #         model_path,
    #         model_max_length=2048,
    #         padding_side="right",
    #         use_fast=False,
    #         trust_remote_code=True
    #     )
    # os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
    # print(f"Using GPU is CUDA:{os.environ['CUDA_VISIBLE_DEVICES']}")
    # device = "cpu" if not torch.cuda.is_available() else "cuda"

    # model, tokenizer = load_model(model_path, device=device, num_gpus=2)
    # tokenizer.pad_token = tokenizer.unk_token
    # model = transformers.AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     trust_remote_code=True
    # )
    # model.eval()
    # model.to(device)
    # while True:
    #     query = input("query: ")
    #     query_question = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions." + "USER: " + query + " ASSISTANT: "
    #     print("question is: {}".format(query_question))
    #     params = {
    #         "prompt": query_question,
    #         "temperature": 0.5,
    #         "top_p": 1.0,
    #         "max_new_tokens": 512,
    #         "use_plugin": True,
    #         "stream_interval": 3,
    #     }
    #     output_stream = generate_stream(model, tokenizer, params, device, context_len=2048, stream_interval=3, bagging_times=5)
    #     for one_stream in output_stream:
    # print(one_stream)
