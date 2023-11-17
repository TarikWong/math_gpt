# -*- coding: utf-8 -*-
# @Time : 2023/10/25 9:19 上午
# @Author : tuo.wang
# @Version : 
# @Function :
import json


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

    def get_config(self, input_file="input.csv", output_file="output.json", kc_file="初中知识点（纯）.xlsx", sample_cnt=10, env="线上", version="基础", subject="数学", grade="初中") -> dict:
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

            "sublevel_system": "{base_dir}/prompts/system/{version}/{subject}/{grade}/system-sublevel.md".format(
                **{
                    "base_dir": self.base_dir[self.outline["环境"][env]],
                    "version": self.outline["版本"][version],
                    "subject": self.outline["学科"][subject],
                    "grade": self.outline["年级"][grade],
                }
            ),
            "sublevel_examples": [
                "{base_dir}/prompts/examples/{version}/{subject}/{grade}/example-sublevel-01.md".format(
                    **{
                        "base_dir": self.base_dir[self.outline["环境"][env]],
                        "version": self.outline["版本"][version],
                        "subject": self.outline["学科"][subject],
                        "grade": self.outline["年级"][grade],
                    }
                ),
                "{base_dir}/prompts/examples/{version}/{subject}/{grade}/example-sublevel-02.md".format(
                    **{
                        "base_dir": self.base_dir[self.outline["环境"][env]],
                        "version": self.outline["版本"][version],
                        "subject": self.outline["学科"][subject],
                        "grade": self.outline["年级"][grade],
                    }
                ),
            ],
            "lastlevel_system": "{base_dir}/prompts/system/{version}/{subject}/{grade}/system-lastlevel.md".format(
                **{
                    "base_dir": self.base_dir[self.outline["环境"][env]],
                    "version": self.outline["版本"][version],
                    "subject": self.outline["学科"][subject],
                    "grade": self.outline["年级"][grade],
                }
            ),
            "lastlevel_examples": [
                "{base_dir}/prompts/examples/{version}/{subject}/{grade}/example-lastlevel-01.md".format(
                    **{
                        "base_dir": self.base_dir[self.outline["环境"][env]],
                        "version": self.outline["版本"][version],
                        "subject": self.outline["学科"][subject],
                        "grade": self.outline["年级"][grade],
                    }
                ),
                "{base_dir}/prompts/examples/{version}/{subject}/{grade}/example-lastlevel-02.md".format(
                    **{
                        "base_dir": self.base_dir[self.outline["环境"][env]],
                        "version": self.outline["版本"][version],
                        "subject": self.outline["学科"][subject],
                        "grade": self.outline["年级"][grade],
                    }
                ),
            ],

            "kc_dir": "{base_dir}/kc/{version}/{subject}/{grade}".format(
                **{
                    "base_dir": self.base_dir[self.outline["环境"][env]],
                    "version": self.outline["版本"][version],
                    "subject": self.outline["学科"][subject],
                    "grade": self.outline["年级"][grade],
                }
            ),
        }
        return config_dict


if __name__ == "__main__":
    c = ConfigParser()
    d = c.get_config(input_file="输入文件.txt", output_file="输出文件.json", sample_cnt=10, env="本地", version="基础", subject="数学", grade="初中")
    print(json.dumps(d, ensure_ascii=False, indent=4))

