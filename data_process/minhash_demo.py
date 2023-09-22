# -*- coding: utf-8 -*-
# @Time : 2023/9/21 3:40 下午
# @Author : tuo.wang
# @Version : 
# @Function :
from datasketch import MinHash, MinHashLSH
import jieba


def text2words(param_text):
    return list(jieba.cut(param_text))


def words2hash(param_set, param_num_perm=128):
    m = MinHash(num_perm=param_num_perm)
    for d in param_set:
        m.update(d.encode('utf8'))
    return m


def clean_text(text):
    text = text.replace("\\\\", "\\").replace("$$", "$").replace("\n", "")
    return text


if __name__ == '__main__':
    # 有无引导语句相似度阈值0.6左右
    text1 = '''In your role as a mathematical computation assistant, please answer the following math questions:: 已知太阳光从太阳表面射出后大约需要$8$分钟才能到达地球，光速约为$300000$千米/秒，则地球到太>阳的距离大约为 $\\underline{ }$ 亿千米。'''
    text2 = '''In your role as a mathematical computation assistant, please answer the following math questions:: 已知太阳光从太阳表面射出后大约需要$8$分钟才能到达地球，光速约为$30$千米/秒，则地球到太>阳的距离大约为 $\\underline{ }$ 亿千米。'''
    text3 = '''In your role as a mathematical computation assistant, please answer the following math questions:: 已知太阳光从月亮表面射出后大约需要$165$分钟才能到达火星，光速约为$3000000$千米/秒，则地球到太>阳的距离大约为 $\\underline{ }$ 亿千米。'''


    # 其中一些数字、字符不一样，相似度0.88左右；删除一段话，相似度0.52左右
    # text1 = '''Several sets of prime numbers, such as $\\left\\{7,83,421,659\\right\\}$ use each of the nine nonzero digits exactly once. What is the smallest possible sum such a set of primes could have?（   ）A. $193$B. $207$C. $225$D. $252$E. $447$'''
    # text2 = '''Several sets of prime numbers, such as $\\left\\{789,8323,42112,659231\\right\\}$ use each of the nine nonzero digits exactly once. What is the largest possible sum such a set of primes could have?（   ）A. $193$B. $207$C. $225$D. $252$E. $447$'''
    # text3 = '''What is the smallest possible sum such a set of primes could have?（   ）A. $193$B. $207$C. $225$D. $252$E. $447$'''

    m1 = words2hash(text2words(clean_text(text1)))
    m2 = words2hash(text2words(clean_text(text2)))
    m3 = words2hash(text2words(clean_text(text3)))

    # Create LSH index
    threshold = 0.87
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    lsh.insert("m2", m2)
    lsh.insert("m3", m3)
    result = lsh.query(m1)
    print("Approximate neighbours with Jaccard similarity > {} : ".format(threshold), result)
