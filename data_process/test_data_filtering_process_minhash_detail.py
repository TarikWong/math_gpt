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
    '/mnt/pfs/zitao_team/big_model/wangtuo_data/train_data/beautiful_analysis.jsonl',
    # '/mnt/pfs/zitao_team/big_model/processed_data/jiaoyanyun_data/tq_tmp/glm_training/peiyou_paper_q2a.jsonl',
    # '/mnt/pfs/zitao_team/big_model/processed_data/jiaoyanyun_data/tq_tmp/glm_training/cloud_paper_q2a.jsonl',
    # '/mnt/pfs/zitao_team/big_model/processed_data/jiaoyanyun_data/tq_tmp/glm_training/tipaipai_q2a.jsonl',
    # '/Users/tuo/train_data/',
]

## 输出文件路径
CONSTANT__OUTPUT_PATH_DIR = '/mnt/pfs/zitao_team/big_model/wangtuo_data/output_data_details/'
## 输出diff文件路径
CONSTANT__OUTPUT_DIFF_PATH_DIR = '/mnt/pfs/zitao_team/big_model/wangtuo_data/output_data_details/diff.txt'


# 加载指定目录下的所有文件，返回dict，key为文件全路径，value为json列表
def load_data_from_specific_dir(param_path_list):
    result = {}
    # input_path = Path(param_path)
    # files = [file.name for file in input_path.rglob("*.*")]

    # 遍历所有文件
    for file in param_path_list:
        print("file: ", file)
        json_list = []
        with open(file, 'r', encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                json_list.append(data)
        result[file] = json_list
    return result


# 加载不同目录下的文件，返回列表->dict->key为文件全路径，value为json列表
def load_train_data(param_path_list):
    return [load_data_from_specific_dir(param_path_list)]
    # return [load_data_from_specific_dir(path) for path in param_path_list]


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


# create lsh index
def minhash_lsh_initialize(param_path):
    original_test_data_dict = {}
    lsh = MinHashLSH(threshold=0.8, num_perm=128)
    data_dict = load_test_data(param_input_path=param_path)
    print('[{}]测试语料加载完成.'.format(get_current_time_string()))

    counter = 0
    for key_path, value_json_list in data_dict.items():
        for i in range(len(value_json_list)):
            text = clean_text(value_json_list[i]['question'])
            cut_text_set = text2words(param_text=text)
            text_minhash = words2hash(param_set=cut_text_set)
            lsh.insert(str(counter), text_minhash)
            original_test_data_dict[str(counter)] = value_json_list[i]['question']
            counter += 1
    return lsh, original_test_data_dict


def get_current_time_string(format='%Y-%m-%d %H:%M:%S'):
    return datetime.datetime.now().strftime(format)


def clean_text(text):
    text = text.replace("\\\\", "\\").replace("$$", "$").replace("\n", "")
    return text


if __name__ == '__main__':
    print('[{}]任务开始...'.format(get_current_time_string()))
    result_list = []
    diff_list = []
    lsh, original_test_data_dict = minhash_lsh_initialize(param_path=CONSTANT__TEST_PATH)  # 加载训练集语料初始化MinHashLSH
    print('[{}]MinHashLSH初始化完成.'.format(get_current_time_string()))
    train_data_list = load_train_data(param_path_list=CONSTANT__TRAIN_PATH_LIST)  # 加载测试数据
    print('[{}]训练语料加载完成.'.format(get_current_time_string()))

    for data in train_data_list:
        result_dict = {}
        for path, json_list in data.items():
            new_json_list = []
            if 'tipaipai' not in path:
                for value_json in json_list:
                    prompt_text = clean_text(value_json['question'])
                    words_set = set(text2words(param_text=prompt_text))
                    hash = words2hash(param_set=words_set)
                    query_reault_list = lsh.query(hash)
                    if len(query_reault_list) == 0:
                        new_json_list.append(value_json)
                    else:
                        for minhash_index in query_reault_list:
                            diff_list.append("train:{},   test:{}".format(prompt_text, original_test_data_dict[minhash_index]))
                            # print("train:{},   test:{}".format(prompt_text, original_test_data_dict[minhash_index]))
            else:
                for value_json in json_list:
                    flag = 0
                    for c in value_json['conversations']:
                        prompt_text = clean_text(c['value'])
                        words_set = set(text2words(param_text=prompt_text))
                        hash = words2hash(param_set=words_set)
                        if len(lsh.query(hash)) > 0:
                            flag += 1
                    if flag == 0:
                        new_json_list.append(value_json)
            result_dict[path] = new_json_list
        result_list.append(result_dict)

    print('[{}]开始数据校验...'.format(get_current_time_string()))
    print('original data: ')
    for i in train_data_list:
        for key, value in i.items():
            print('file: {}, data length: {}'.format(key, len(value)))

    print('processed data: ')
    for i in result_list:
        for key, value in i.items():
            print('file: {}, data length: {}'.format(key, len(value)))

    print('[{}]保存过滤完成的数据...'.format(get_current_time_string()))
    for i in result_list:
        for key, value in i.items():
            json_data = json.dumps(value, ensure_ascii=False)

            output_path = '{}{}/{}'.format(CONSTANT__OUTPUT_PATH_DIR, key.split('/')[-2], key.split('/')[-1])
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_data)

    print('[{}]保存相似度大于阈值的数据...'.format(get_current_time_string()))
    with open(CONSTANT__OUTPUT_DIFF_PATH_DIR, 'w', encoding='utf-8') as f:
        for i in diff_list:
            f.write(i)
            f.write('\n')

    print('[{}]任务完成!'.format(get_current_time_string()))
