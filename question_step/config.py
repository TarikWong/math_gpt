# -*- coding: utf-8 -*-
# @Time : 2023/10/25 9:19 上午
# @Author : tuo.wang
# @Version : 
# @Function : 配置文件
import json
import os
import datetime


class ConfigParser(object):

    def __init__(self):
        self.outline = {
            "学科": {"数学": "math", "语文": "chinese", "英语": "english"},
            "年级": {"小学": "primary", "初中": "junior", "高中": "senior"},
            "环境": {"线上": "online", "本地": "local"},
            "版本": {"基础": "basic", "推理过程": "CoT"},
        }
        self.base_dir = {
            "online": "/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step",
            "local": "/Users/tuo/PycharmProjects/math_gpt/question_step"
        }

    # default=basic初中数学
    def get_config(self, input_file="input.csv", output_file="output.json",
                   sublevel_system_file="system-sublevel.md", lastlevel_system_file="system-lastlevel.md",
                   sublevel_examples_list=None, lastlevel_examples_list=None,
                   kc_file="初中知识点（纯）.xlsx", sample_cnt=10, env="线上", version="基础", subject="数学", grade="初中",
                   dir_check=False) -> dict:
        if lastlevel_examples_list is None:
            lastlevel_examples_list = ["example-lastlevel-01.md", "example-lastlevel-02.md"]
        if sublevel_examples_list is None:
            sublevel_examples_list = ["example-sublevel-01.md", "example-sublevel-02.md"]

        config_dict = {
            "sample_cnt": sample_cnt,
            "kc_file": kc_file,
            "log_dir": "{base_dir}/log".format(
                **{
                    "base_dir": self.base_dir[self.outline["环境"][env]]
                }
            ),
            "log_file": "{base_dir}/log/{file_name}.log".format(
                **{
                    "base_dir": self.base_dir[self.outline["环境"][env]],
                    "file_name": input_file.split(".")[0]
                }
            ),

            "input_file_name": input_file,
            "input_dir": "{base_dir}/data/{subject}/{grade}".format(
                **{
                    "base_dir": self.base_dir[self.outline["环境"][env]],
                    "subject": self.outline["学科"][subject],
                    "grade": self.outline["年级"][grade],
                }
            ),
            "input_file": "{base_dir}/data/{subject}/{grade}/{file_name}".format(
                **{
                    "base_dir": self.base_dir[self.outline["环境"][env]],
                    "subject": self.outline["学科"][subject],
                    "grade": self.outline["年级"][grade],
                    "file_name": input_file,
                }
            ),

            "output_file_name": output_file,
            "output_dir": "{base_dir}/tmp/{subject}/{grade}/".format(
                **{
                    "base_dir": self.base_dir[self.outline["环境"][env]],
                    "subject": self.outline["学科"][subject],
                    "grade": self.outline["年级"][grade],
                }
            ),
            "output_file": "{base_dir}/tmp/{subject}/{grade}/{file_name}".format(
                **{
                    "base_dir": self.base_dir[self.outline["环境"][env]],
                    "subject": self.outline["学科"][subject],
                    "grade": self.outline["年级"][grade],
                    "file_name": output_file,
                }
            ),

            "system_dir": "{base_dir}/prompts/system/{version}/{subject}/{grade}/".format(
                **{
                    "base_dir": self.base_dir[self.outline["环境"][env]],
                    "version": self.outline["版本"][version],
                    "subject": self.outline["学科"][subject],
                    "grade": self.outline["年级"][grade],
                }
            ),
            "example_dir": "{base_dir}/prompts/examples/{version}/{subject}/{grade}/".format(
                **{
                    "base_dir": self.base_dir[self.outline["环境"][env]],
                    "version": self.outline["版本"][version],
                    "subject": self.outline["学科"][subject],
                    "grade": self.outline["年级"][grade],
                }
            ),

            "sublevel_system": "{base_dir}/prompts/system/{version}/{subject}/{grade}/{system_file}".format(
                **{
                    "base_dir": self.base_dir[self.outline["环境"][env]],
                    "version": self.outline["版本"][version],
                    "subject": self.outline["学科"][subject],
                    "grade": self.outline["年级"][grade],
                    "system_file": sublevel_system_file
                }
            ),
            "sublevel_examples": ["{base_dir}/prompts/examples/{version}/{subject}/{grade}/{example_file}".format(
                **{
                    "base_dir": self.base_dir[self.outline["环境"][env]],
                    "version": self.outline["版本"][version],
                    "subject": self.outline["学科"][subject],
                    "grade": self.outline["年级"][grade],
                    "example_file": example_file
                }
            ) for example_file in sublevel_examples_list],

            "lastlevel_system": "{base_dir}/prompts/system/{version}/{subject}/{grade}/{system_file}".format(
                **{
                    "base_dir": self.base_dir[self.outline["环境"][env]],
                    "version": self.outline["版本"][version],
                    "subject": self.outline["学科"][subject],
                    "grade": self.outline["年级"][grade],
                    "system_file": lastlevel_system_file
                }
            ),
            "lastlevel_examples": ["{base_dir}/prompts/examples/{version}/{subject}/{grade}/{example_file}".format(
                **{
                    "base_dir": self.base_dir[self.outline["环境"][env]],
                    "version": self.outline["版本"][version],
                    "subject": self.outline["学科"][subject],
                    "grade": self.outline["年级"][grade],
                    "example_file": example_file
                }
            ) for example_file in lastlevel_examples_list],

            "kc_dir": "{base_dir}/kc/{version}/{subject}/{grade}".format(
                **{
                    "base_dir": self.base_dir[self.outline["环境"][env]],
                    "version": self.outline["版本"][version],
                    "subject": self.outline["学科"][subject],
                    "grade": self.outline["年级"][grade],
                }
            ),
        }

        if dir_check:
            # 目录校验，不存在就创建目录，但不会创建文件
            dir_dict = self.__search_dir(config_dict)
            for k, v in dir_dict.items():
                dir_dict[k] = self.__create_dir(v)

            print("【{}】 config_dict: ".format(self.__get_current_time_string()),
                  json.dumps(config_dict, ensure_ascii=False, indent=4), "\n==========================================")
            print("【{}】 dir_dict: ".format(self.__get_current_time_string()),
                  json.dumps(dir_dict, ensure_ascii=False, indent=4))
            print("================ 加载参数配置完成 ================")
        return config_dict

    def __search_dir(self, target_dict: dict, search_str="_dir"):
        return {k: v for k, v in target_dict.items() if search_str in k}

    def __create_dir(self, target_dir: str):
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            return "mkdir successfully"
        else:
            return "already existed"

    def __get_current_time_string(self, format='%Y-%m-%d %H:%M:%S'):
        return datetime.datetime.now().strftime(format)


if __name__ == "__main__":
    c = ConfigParser()
    # d = c.get_config(input_file="输入文件.csv", output_file="输出文件.json", sample_cnt=10, env="本地", version="基础",
    #                  subject="数学", grade="初中")
    # d = c.get_config(input_file="输入文件.csv", output_file="输出文件.json", sample_cnt=10, env="线上", version="基础",
    #                  subject="数学", grade="初中")
    print("done.")
