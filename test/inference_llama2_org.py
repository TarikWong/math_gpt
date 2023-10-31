"""Inference for FastChat models."""
import abc
import gc
import math
from typing import Iterable, Optional
import sys
import time
import warnings
import argparse
import os, json
import re
import signal

import psutil
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

from fastchat.conversation import get_conv_template, SeparatorStyle
from fastchat.model.model_adapter import load_model, get_conversation_template
# from fastchat.model.chatglm_model import chatglm_generate_stream
# from fastchat.model.falcon_model import falcon_generate_stream
# from fastchat.modules.gptq import GptqConfig
from fastchat.utils import is_partial_stop
import queue


class Streamer:
    def __init__(self, ):
        self.stream_data = queue.Queue()
        self.cnt = 0
        self.chunk_cache = []

    def write_step(self, status, item, skip_mode=False, output_tokens_collector=None):
        if output_tokens_collector is None:
            output_tokens_collector = []
        if skip_mode:
            return
        output_tokens_collector.append(item)
        self.stream_data.put([status, item])
        self.cnt += 1

    def add_cache(self, text):
        self.chunk_cache.append(text)

    def return_cache(self, last_word=False):
        if last_word:
            output = "".join(self.chunk_cache).split()[-1]
        else:
            output = "".join(self.chunk_cache)
        self.chunk_cache = []
        return output

    def __iter__(self):
        return self

    def __next__(self):
        while self.stream_data.empty():
            time.sleep(0.1)
            continue
        return self.stream_data.get()


def put_queue(status, word):
    print([status, word])
    # raise NotImplementedError


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


@torch.inference_mode()
def generate_stream(
        model, tokenizer, params, device, context_len=2048, stream_interval=2
):
    output_streamer = Streamer()
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)
    echo = bool(params.get("echo", False))
    inner_stop_str = params.get("inner_stop", None)
    stop_token_ids = params.get("stop_token_ids", None) or []
    stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = None

    input_ids = tokenizer(prompt).input_ids
    output_ids = list(input_ids)

    if model.config.is_encoder_decoder:
        max_src_len = context_len
    else:
        max_src_len = context_len - max_new_tokens - 8

    input_ids = input_ids[-max_src_len:]
    input_echo_len = len(input_ids)

    if model.config.is_encoder_decoder:
        encoder_output = model.encoder(
            input_ids=torch.as_tenosr([input_ids], device=device)
        )[0]
        start_ids = torch.as_tensor(
            [[model.generation_config.decoder_start_token_id]],
            dtype=torch.int64,
            device=device,
        )
    # whether stop when meet special token during generation
    stopped_inner = False

    past_key_values = out = None

    for i in range(max_new_tokens):
        if i == 0:
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=start_ids,
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                )
                logits = model.lm_head(out[0])
            else:
                out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits
            past_key_values = out.past_key_values
        else:
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=torch.as_tensor([[token]], device=device),
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                    past_key_values=past_key_values,
                )

                logits = model.lm_head(out[0])
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

        output_ids.append(token)
        # TODO: output by token

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
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
            )

            partially_stopped = False
            if inner_stop_str:
                for each_stop in inner_stop_str:
                    pos = output.rfind(each_stop, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                        stopped_inner = True
                        break
                    else:
                        partially_stopped = is_partial_stop(output, each_stop)
                        if partially_stopped:
                            break

            if stop_str:
                if isinstance(stop_str, str):
                    pos = output.rfind(stop_str, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                    else:
                        partially_stopped = is_partial_stop(output, stop_str)
                elif isinstance(stop_str, Iterable):
                    for each_stop in stop_str:
                        pos = output.rfind(each_stop, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                            break
                        else:
                            partially_stopped = is_partial_stop(output, each_stop)
                            if partially_stopped:
                                break
                else:
                    raise ValueError("Invalid stop field type.")

            # prevent yielding partial stop sequence
            if not partially_stopped:
                yield {
                    "text": output,
                    "usage": {
                        "prompt_tokens": input_echo_len,
                        "completion_tokens": i,
                        "total_tokens": input_echo_len + i,
                    },
                    "finish_reason": None,
                }
        if stopped_inner:
            break

        if stopped:
            break

    # finish stream event, which contains finish reason
    if i == max_new_tokens - 1:
        finish_reason = "length"
    elif stopped_inner:
        finish_reason = "inner_stop"
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

    # clean
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()


def transfer_test_data_to_template(test_data):
    new_test_data = []
    for one_data in test_data:
        one_data["conversations"] = [{"value": "求解以下题目:" + one_data["question"]}, {"value": one_data["answer"]}]
        new_test_data.append(one_data)
    return new_test_data


def predict_one_iter(input_prompt, args):
    # Set CUDA visible devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if "simple_cal" in args.model_name:
        args.template = "simple_cal"
    elif "galactica" in args.model_name:
        args.template = "galactica"
    elif "llama" in args.model_name:
        args.template = "llama"

    print(args)
    # Set device
    device = "cpu" if not torch.cuda.is_available() else "cuda"

    # Load test data
    test_file_path = args.test_file_path
    test_data = json.load(open(test_file_path, encoding="utf-8"))
    test_data = transfer_test_data_to_template(test_data)

    # Load model and tokenizer
    model_name = args.model_name
    model_path = args.model_path
    if "llama" in model_name:
        model, tokenizer = load_model(model_path, device=device, num_gpus=args.num_gpus)
    elif "galactica" in model_name:
        from transformers import AutoTokenizer, OPTForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = OPTForCausalLM.from_pretrained(model_path, device_map="auto")

    gen_params = {
        "model": args.model_path,
        "prompt": input_prompt,
        "temperature": args.sample_temperature,
        "repetition_penalty": 1.0,
        "max_new_tokens": 1024,
        "stop": ["</thought>", "</s>"],  # </s> is the end of sentence
        "echo": False,
    }
    output_stream = generate_stream(model, tokenizer, gen_params, device)
    return output_stream


def main(args):
    # Set CUDA visible devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if "simple_cal" in args.model_name:
        args.template = "simple_cal"
    elif "galactica" in args.model_name:
        args.template = "galactica"
    elif "llama" in args.model_name:
        args.template = "llama"

    print(args)
    # Set device
    device = "cpu" if not torch.cuda.is_available() else "cuda"

    # Load test data
    test_file_path = args.test_file_path
    test_data = json.load(open(test_file_path, encoding="utf-8"))
    test_data = transfer_test_data_to_template(test_data)

    # Load model and tokenizer
    model_name = args.model_name
    model_path = args.model_path
    if "llama" in model_name:
        model, tokenizer = load_model(model_path, device=device, num_gpus=args.num_gpus)
    elif "galactica" in model_name:
        from transformers import AutoTokenizer, OPTForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = OPTForCausalLM.from_pretrained(model_path, device_map="auto")


def construct_first_prompt(user_input_prefix):
    input_question = user_input_prefix

    prefix_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions. USER: "

    constructed_prompt = prefix_prompt + input_question + ' ASSISTANT: '
    return constructed_prompt


def extract_thought(target_generation):
    # list of string where the string is between <thought> and </though>
    return re.findall("<thought>.*?</thought>", target_generation, re.S)


def timeout_handler(signum, frame):
    raise TimeoutError("Function timed out.")


def set_timeout(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Set the timeout alarm
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)

            try:
                # Call the function
                result = func(*args, **kwargs)
                return result
            finally:
                # Cancel the timeout alarm
                signal.alarm(0)

        return wrapper

    return decorator


@set_timeout(5)
def execute_code_list(code_list):
    variables = {}
    try:
        code_doc = "\n".join(code_list)
        if "result" not in code_list[-1]:
            variables['result'] = ""
            return variables
        exec(code_doc, variables)
    except Exception as e:
        variables['result'] = ""
        print(f"meet error: {e}")
    return variables


def add_quotes(input_json):
    count_quotes = input_json.count('"')
    if count_quotes % 2 == 1:
        # not valid
        if input_json[-2] != '"':
            input_json = input_json[:-1] + '"' + input_json[-1]
    return input_json


def extract_thought_code_result(the_final_thought):
    # the final thought is the thought of the last step
    new_target_thought = the_final_thought
    extracted_code = re.findall("<code>(.*?)</code>", the_final_thought, re.S)
    # 如果抽取的代码不为1
    if len(extracted_code) != 1:
        # 不对的code形式
        return False, new_target_thought
    new_code_str = extracted_code[0]

    code_list = json.loads(new_code_str)

    result_variable = execute_code_list(code_list)

    if result_variable['result'] != "":
        previous_thought = re.findall("<thought>.*?</code>", new_target_thought, re.S)[0]
        new_target_thought = previous_thought + " => 运行结果为{}".format(
            json.dumps(result_variable['result'], ensure_ascii=False, default=str)).replace('"', "'") + "</thought>"
    # if "=>" in new_target_thought:
    #     exec_res = new_target_thought.split("=>")[1].strip().replace("运行结果为","")
    # elif "运行结果为" in new_target_thought:
    #     exec_res = new_target_thought.split("运行结果为")[1].strip().replace("运行结果为","")
    #     split_new_thought = new_target_thought.split("运行结果为")
    #     new_target_thought = split_new_thought[0].strip() + " => 运行结果为" + split_new_thought[1]
    # else:
    #     # not generated the result
    #     exec_res = {}
    # exec_res_obj = eval(exec_res)
    # if result_variable['result'] != "":
    #     if not result_variable['result'] == exec_res_obj:
    #         # the predict is not equal to origin generated result
    #         previous_thought = re.findall("<thought>.*?</code>", new_target_thought, re.S)[0]
    #         new_target_thought = previous_thought + " => 运行结果为{}".format(json.dumps(result_variable['result'], ensure_ascii=False)).replace('"',"'") + "</thought>"
    return True, new_target_thought


def refresh_thought_and_code(generated_result, round_int, calculated_thought):
    extracted_thought = extract_thought(generated_result)
    round_int += 1
    if len(extracted_thought) > 0:
        ori_thought = extracted_thought[-1]
        if ori_thought in calculated_thought:
            return generated_result, round_int, calculated_thought, "error"
        else:
            calculated_thought.add(ori_thought)
        try:
            whether_leagal, new_thought = extract_thought_code_result(ori_thought)
            generated_result = generated_result.replace(ori_thought, new_thought)
        except Exception as e:
            print(e)
            new_thought = ori_thought
    if not generated_result.endswith(" "):
        generated_result = generated_result + " "
    return generated_result, round_int, calculated_thought, "normal"


def get_one_generation_result(input_question, args, model, tokenizer, device):
    round_int = 0
    basic_prompt = construct_first_prompt(input_question)
    gen_target = ""
    new_target = ""
    one_calculated_thought = set()
    while True:
        if round_int == 0:
            input_prompt = basic_prompt
        else:
            input_prompt = basic_prompt + gen_target
        gen_params = {
            "model": args.model_path,
            "prompt": input_prompt,
            "temperature": args.sample_temperature,
            "repetition_penalty": 1.0,
            "max_new_tokens": 1024,
            "inner_stop": ["</thought>"],
            "stop": ["</s>"],  # </s> is the end of sentence
            "echo": False,
        }
        output_stream = generate_stream(model, tokenizer, gen_params, device)
        for one_stream in output_stream:
            if round_int == 0:
                gen_target = one_stream['text']
                # print(gen_target)
            else:
                new_target = one_stream['text']
                # print(gen_target+new_target)
        gen_target = gen_target + new_target
        if one_stream["finish_reason"] == "inner_stop":
            gen_target = gen_target + "</thought>"

        if one_stream["finish_reason"] == "stop":
            break

        if one_stream['finish_reason'] == "length":
            break

        refreshed_target, round_int, one_calculated_thought, status = refresh_thought_and_code(gen_target, round_int,
                                                                                               one_calculated_thought)
        if status == "error":
            break
        gen_target = refreshed_target
        print(gen_target)
    return gen_target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate responses using trained models"
    )
    parser.add_argument("--model_name", type=str, help="Name of the model to use", default="llama")
    parser.add_argument("--model_path", type=str, help="Path to the model checkpoint",
                        default="/mnt/pfs/zitao_team/wangtuo/llm_team/source/training/sft/train_scripts/models/llama_7b_hf_sft_question_step/checkpoint-60")
    parser.add_argument("--test_file_path", type=str, help="Path to the test data file",
                        default="/mnt/pfs/zitao_team/big_model/processed_data/test_data_junior_small_with_full.json")
    parser.add_argument(
        "--result_save_dir",
        type=str,
        default="results",
        help="Path to the test data file",
    )
    parser.add_argument(
        "--test_num", type=int, default=20, help="Number of test items to use"
    )
    parser.add_argument(
        "--num_gen", type=int, default=1, help="Number of generations per test item"
    )
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--sample_temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default="1",
        help="CUDA visible devices (comma-separated indices)",
    )
    parser.add_argument("--template", type=str, default="llama")
    parser.add_argument("--lazy_preprocess", type=str, default="llama")
    args = parser.parse_args()
    # main(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if "simple_cal" in args.model_name:
        args.template = "simple_cal"
    elif "galactica" in args.model_name:
        args.template = "galactica"
    elif "llama" in args.model_name:
        args.template = "llama"

    # print(args)
    # Set device
    device = "cpu" if not torch.cuda.is_available() else "cuda"

    # Load test data
    test_file_path = args.test_file_path
    test_data = json.load(open(test_file_path, encoding="utf-8"))
    test_data = transfer_test_data_to_template(test_data)

    # Load model and tokenizer
    model_name = args.model_name
    model_path = args.model_path
    if "llama" in model_name:
        model, tokenizer = load_model(model_path, device=device, num_gpus=args.num_gpus)
    elif "galactica" in model_name:
        from transformers import AutoTokenizer, OPTForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = OPTForCausalLM.from_pretrained(model_path, device_map="auto")

    for idx in [1]:
        input_question = test_data[idx]['conversations'][0]['value']
        one_res = get_one_generation_result(input_question, args, model, tokenizer, device)
        print("question is: {}".format(input_question))
        print("result is: {}".format(one_res))
        print("answer is: {}".format(test_data[idx]['answer']))

    print("done.")

    # input_question = "求解以下题目:小明一共有12890个苹果，再给小明10个苹果，小明一共有多少个苹果？"
    # one_res = get_one_generation_result(input_question, args, model, tokenizer, device)
    # print("question is: {}".format(input_question))
    # print("result is: {}".format(one_res))

    # --------------------------------
    # while True:
    #     query = input("query: ")
    #     query_question = "求解以下题目:" + query
    #     one_res = get_one_generation_result(query_question, args, model, tokenizer, device)
    #     print("question is: {}".format(query_question))
    #     print("result is: {}".format(one_res))

    # --------------------------------
    # input_question = r"求解以下题目:小明参加知识竞赛，一共回答了$20$道题，答对一题加$5$分，答错一题不但不加分，还倒扣$3$分，小明一共得了$76$分，求小明答对了 $\\underline\{\}$ 道题."
    # round_int = 0
    # basic_prompt = construct_first_prompt(input_question)
    # gen_target = ""
    # new_target = ""
    # while True:
    #     if round_int == 0:
    #         input_prompt = basic_prompt
    #     else:
    #         input_prompt = basic_prompt + gen_target
    #     gen_params = {
    #         "model": args.model_path,
    #         "prompt": input_prompt,
    #         "temperature": args.sample_temperature,
    #         "repetition_penalty": 1.0,
    #         "max_new_tokens": 1024,
    #         "inner_stop": ["</thought>"],
    #         "stop": ["</s>"], # </s> is the end of sentence
    #         "echo": False,
    #     }
    #     output_stream = generate_stream(model, tokenizer, gen_params, device)
    #     for one_stream in output_stream:
    #         if round_int == 0:
    #             gen_target = one_stream['text']
    #             print(gen_target)
    #         else:
    #             new_target = one_stream['text']
    #             print(gen_target+new_target)
    #     gen_target = gen_target + new_target
    #     if one_stream["finish_reason"] == "inner_stop":
    #         gen_target = gen_target + "</thought>"

    #     if one_stream["finish_reason"] == "stop":
    #         break

    #     refreshed_target,round_int = refresh_thought_and_code(gen_target, round_int)
    #     gen_target = refreshed_target
    #     print("*"*10)
    # print("end")
    # --------------------------------
    # input_question = "某快递公司收费方法如下，$1$千克以内（含$1$千克）收费$12$元；超过$1$千克的部分（不足$1$千克按$1$千克算）按每千克$2$元收费．涛涛的快递包裹重$3.4$千克，他需要付快递费 $\\underline{          }$ 元．"

    # prefix_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions. USER: "

    # input_prompt = prefix_prompt + input_question + " ASSISTANT: "

    # tmp_output = predict_one_iter(input_prompt, args)
    # print("hi")
