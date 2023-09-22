# -*- coding: utf-8 -*-
# @Time : 2023/9/20 3:57 下午
# @Author : tuo.wang
# @Version :
# @Function :
from datasketch import MinHash, MinHashLSH
import jieba
import json
from pathlib import Path
import datetime

## 测试集语料目录
CONSTANT__TEST_PATH = '/mnt/pfs/zitao_team/big_model/wangtuo_data/test_data/'
# CONSTANT__TEST_PATH = '/Users/tuo/test_data/'

## 训练集语料路径
CONSTANT__TRAIN_PATH_LIST = [
    '/mnt/pfs/zitao_team/big_model/wangtuo_data/train_data/cloud_paper/',
    '/mnt/pfs/zitao_team/big_model/wangtuo_data/train_data/peiyou_paper/',
    '/mnt/pfs/zitao_team/big_model/wangtuo_data/train_data/personal_paper/',
    '/mnt/pfs/zitao_team/big_model/wangtuo_data/train_data/tipaipai_paper/',
    # '/Users/tuo/train_data/',
]

## 输出文件路径
CONSTANT__OUTPUT_PATH_DIR = '/mnt/pfs/zitao_team/big_model/wangtuo_data/output_data_rough/'


# CONSTANT__OUTPUT_PATH_DIR = '/Users/tuo/output_data/'


# 加载指定目录下的所有文件，返回dict，key为文件全路径，value为json列表
def load_data_from_specific_dir(param_path):
    result = {}
    input_path = Path(param_path)
    files = [file.name for file in input_path.rglob("*.*")]

    # 遍历所有文件
    for file in files:
        json_list = []
        with open(param_path + file, 'r', encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                json_list.append(data)
        result[param_path + file] = json_list
    return result


# 加载不同目录下的文件，返回列表->dict->key为文件全路径，value为json列表
def load_train_data(param_path_list):
    return [load_data_from_specific_dir(path) for path in param_path_list]


def load_test_data(param_input_path):
    result = {}
    input_path = Path(param_input_path)
    files = [file.name for file in input_path.rglob("*.*")]

    for file in files:
        with open(param_input_path + file, 'r', encoding="utf-8") as f:
            test_data_list = json.load(f)
            result[param_input_path + file] = test_data_list

    return result


def text2words(param_text):
    return list(jieba.cut(param_text))


def words2hash(param_set, param_num_perm=128):
    m = MinHash(num_perm=param_num_perm)
    for d in param_set:
        m.update(d.encode('utf8'))
    return m


def initialize_lsh(param_path_list, param_json_prompt='prompt'):
    lsh = MinHashLSH(threshold=0.5, num_perm=128)
    counter = 0

    data_list = load_train_data(param_path_list=param_path_list)
    print('[{}]训练语料加载完成.'.format(get_current_time_string()))
    for data in data_list:
        for key_path, value_json_list in data.items():
            if '题拍拍' not in key_path:
                for value_json in value_json_list:
                    # print("非题拍拍prompt: ", value_json[param_json_prompt])
                    prompt_text = value_json[param_json_prompt]
                    words_set = set(text2words(param_text=prompt_text))
                    hash = words2hash(param_set=words_set)
                    lsh.insert(str(counter), hash)
                    counter += 1
            else:
                for value_json in value_json_list:
                    for c in value_json['conversations']:
                        # print("题拍拍prompt: ", c['value'])
                        words_set = set(text2words(param_text=c['value']))
                        hash = words2hash(param_set=words_set)
                        lsh.insert(str(counter), hash)
                        counter += 1
    return lsh


def get_current_time_string(format='%Y-%m-%d %H:%M:%S'):
    return datetime.datetime.now().strftime(format)


def clean_text(text):
    text = text.replace("\\\\", "\\").replace("$$", "$").replace("\n", "")
    return text


if __name__ == '__main__':
    # 加载数据
    train_data_prompt_dict = {}
    test_data_prompt_list = []
    output_train_data_prompt_list = []

    train_data_list = load_train_data(param_path_list=CONSTANT__TRAIN_PATH_LIST)
    test_data_dict = load_test_data(param_input_path=CONSTANT__TEST_PATH)
    # print("train_data_list: ", train_data_list)
    # print("test_data_dict: ", test_data_dict)

    # 测试数据prompt保存为list
    for key_path, value_json_list in test_data_dict.items():
        print("key_path: ", key_path)
        for i in range(len(value_json_list)):
            test_data_prompt_string = clean_text(value_json_list[i]['question'])
            test_data_prompt_list.append(test_data_prompt_string)

    # 遍历训练数据，过滤掉包含测试数据的语料
    for i in train_data_list:
        output_train_data_prompt_dict = {}
        for k_path, v_json_list in i.items():
            result_list = []
            if '题拍拍' in k_path:
                for line in v_json_list:
                    for c in line['conversations']:
                        question_prompt = clean_text(c['value'])
                        # print("question_prompt: ", question_prompt)
                        if question_prompt not in test_data_prompt_list:
                            result_list.append(line)
            else:
                for value_json in v_json_list:
                    question_prompt = clean_text(value_json['prompt'])
                    # print("question_prompt:", question_prompt)
                    if question_prompt not in test_data_prompt_list:
                        result_list.append(value_json)
            output_train_data_prompt_dict[k_path] = result_list
        output_train_data_prompt_list.append(output_train_data_prompt_dict)

    print('[{}]开始数据校验...'.format(get_current_time_string()))
    print('original data: ')
    for i in train_data_list:
        for key, value in i.items():
            print('file: {}, data length: {}'.format(key, len(value)))

    print('processed data: ')
    for i in output_train_data_prompt_list:
        for key, value in i.items():
            print('file: {}, data length: {}'.format(key, len(value)))

    print('[{}]保存过滤完成的数据...'.format(get_current_time_string()))
    for i in output_train_data_prompt_list:
        for key, value in i.items():
            json_data = json.dumps(value, ensure_ascii=False)

            output_path = '{}{}/{}'.format(CONSTANT__OUTPUT_PATH_DIR, key.split('/')[-2], key.split('/')[-1])
            # print("output_path: ", output_path)
            # print("json_data: ", json_data)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_data)

    print('done.')
