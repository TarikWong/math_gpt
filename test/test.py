import pandas as pd
import json
from typing import *
from sklearn.model_selection import train_test_split
import argparse


def argparse_check(*args):
    if args.model_name == None:
        print("model_name为空")
        exit(2)




if __name__ == "__main__":
    # a = [i for i in range(1, 10)]
    # train, test = train_test_split(a, test_size=0.1, random_state=123)
    # print(train)
    # print(test)

    parser = argparse.ArgumentParser(description="Generate responses using trained models", argument_default=None)
    parser.add_argument("--model_name", type=str, help="模型名称")

    args = parser.parse_args()
    argparse_check(args)
    print("model_name: ", args.model_name)
