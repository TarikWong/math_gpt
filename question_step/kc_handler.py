# -*- coding: utf-8 -*-
# @Time : 2023/10/19 6:19 下午
# @Author : tuo.wang
# @Version : 
# @Function :
import pandas as pd
import json
import os
from collections import defaultdict
from typing import *
from config import config_dict

config = config_dict["test"]
kc_dir = config["kc_dir"]
kc_file = config["kc_file"]


class Knowledge:
    sheet_name = [
        "数",
        "式",
        "方程与不等式",
        "函数",
        "几何图形初步",
        "三角形",
        "四边形",
        "圆",
        "几何变换",
        "统计与概率",
    ]
    sheet_name = ["副本" + item for item in sheet_name]

    def __init__(
            self, sheet_list=None, lask_kc_file_name: str = "kc_extend_mapping.json"
    ) -> None:
        self.kc_file = "{}/{}".format(kc_dir, kc_file)
        self.last_kc_file_name = lask_kc_file_name
        if sheet_list is not None:
            self.sheet_name = sheet_list
        self._mapping = self.map_sub_level2kc()

    def map_sub_level2kc(self) -> Dict[str, List]:
        """根据教研老师整理好的文本,构造次末级到末级的映射."""
        kc_details_json = defaultdict(list)
        for sheet in self.sheet_name:
            tmp = defaultdict(list)
            kc_data = pd.read_excel(self.kc_file, sheet_name=sheet, names=["次末级", "末级"])
            kc_data.fillna(method="pad", inplace=True)
            for _, (a, b) in kc_data.iterrows():
                tmp[a].append(b)
            tmp = self.rebuild(kc_details_json, tmp)
            kc_details_json.update(tmp)
        kc_details_json = {k: list(set(v)) for k, v in kc_details_json.items()}
        self.to_json_file(kc_details_json)
        return kc_details_json

    def to_json_file(self, mapping_dict: Dict[str, List[str]]):
        with open(
                os.path.join(kc_dir, self.last_kc_file_name), encoding="utf-8", mode="w"
        ) as f:
            json.dump(
                mapping_dict,
                f,
                ensure_ascii=False,
                indent=2,
            )

    def rebuild(
            self, original: Dict[str, List], new: Dict[str, List]
    ) -> Dict[str, List]:
        """处理次末级知识点映射关系中的次末级知识点."""
        for key, value in new.items():
            tmp = []
            for item in value:
                if item in original:
                    tmp.extend(original.get(item))
                else:
                    tmp.append(item)
            new[key] = tmp
        return new

    @property
    def mapping(self) -> Dict[str, List[str]]:
        """次末级知识点到末级知识点的映射关系."""
        return self._mapping

    @property
    def sub_level_kc(self) -> List[str]:
        """次末级知识点列表."""
        return list(self.mapping.keys())

    @property
    def last_level_kc(self) -> List[str]:
        """末级知识点列表."""
        kc_list = []
        for _, val in self.mapping.items():
            kc_list.extend(val)
        return kc_list

    @property
    def sub_level_kc_cnt(self) -> int:
        """次末级知识点数量."""
        return len(self.sub_level_kc)

    @property
    def last_level_kc_cnt(self) -> str:
        """末级知识点数量."""
        return len(self.last_level_kc)


if __name__ == "__main__":
    kc = Knowledge()
    print(kc.sub_level_kc_cnt)
    print(kc.last_level_kc_cnt)
    print(kc.last_level_kc)
