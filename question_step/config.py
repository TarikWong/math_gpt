# -*- coding: utf-8 -*-
# @Time : 2023/10/25 9:19 上午
# @Author : tuo.wang
# @Version : 
# @Function :

config_dict = {

    "online": {
        "log_dir": "/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step/log/",
        "input_file_name": "source3_sample5000.csv",
        "input_dir": "/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step/data/",
        "output_dir": "/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step/tmp/",
        "sublevel_system": "/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step/prompts/system/step-1.md",
        "sublevel_example": "/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step/prompts/examples/example-1.md",
        "lastlevel_system": "/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step/prompts/system/step-2.md",
        "lastlevel_example": "/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step/prompts/examples/example-2.md",
        "sample_cnt": 5000,

        "kc_dir": "/mnt/pfs/zitao_team/shiluyou/question_step/kc",
        "kc_file": "初中知识点（纯）.xlsx",

        "data_dir": "/mnt/pfs/zitao_team/big_model/wangtuo_data/question_step/data/"},

    "test": {
        "log_dir": "/Users/tuo/PycharmProjects/math_gpt/log/",
        "input_file_name": "source3_sample30.csv",
        "input_dir": "/Users/tuo/PycharmProjects/math_gpt/question_step/data/",
        "output_dir": "/Users/tuo/PycharmProjects/math_gpt/question_step/tmp/",
        "sublevel_system": "/Users/tuo/PycharmProjects/math_gpt/prompts/system/step-1.md",
        "sublevel_example": "/Users/tuo/PycharmProjects/math_gpt/prompts/examples/example-1.md",
        "lastlevel_system": "/Users/tuo/PycharmProjects/math_gpt/prompts/system/step-2.md",
        "lastlevel_example": "/Users/tuo/PycharmProjects/math_gpt/prompts/examples/example-2.md",
        "sample_cnt": 20,

        "kc_dir": "/Users/tuo/PycharmProjects/math_gpt/question_step/kc",
        "kc_file": "初中知识点（纯）.xlsx",

        "data_dir": "/Users/tuo/PycharmProjects/math_gpt/question_step/data/"}
}
