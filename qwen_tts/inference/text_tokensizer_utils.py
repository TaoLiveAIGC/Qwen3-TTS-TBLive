import json
import os
import random
import shutil
from typing import List
import torch
from transformers import PreTrainedTokenizer

# 添加结束符号与第三等级标签符号
DEFAULT_SPECIAL_TOKENS=[
    "[energy_low]",  "[energy_mid]",  "[energy_high]",
    "[pitch_low]",   "[pitch_mid]",   "[pitch_high]",
    "[speed_low]",   "[speed_mid]",   "[speed_high]",
    "[energy_d1]", "[energy_d2]", "[energy_e1]", "[energy_e2]",
    "[pitch_rate_d1]",  "[pitch_rate_d2]",  "[pitch_rate_e1]",  "[pitch_rate_e2]",
    "[speed_d1]",  "[speed_d2]",  "[speed_e1]",  "[speed_e2]",
    "[speed_d3]", "[speed_e3]", "[pitch_rate_d3]", "[pitch_rate_e3]", "[energy_d3]", "[energy_e3]",
    "<end_ins>",
]

def add_special_tokens_to_tokenizer(
    tokenizer: PreTrainedTokenizer,
    special_tokens: List[str] = DEFAULT_SPECIAL_TOKENS,
) -> int:
    """
    以 special token 方式添加 token 到 tokenizer
    
    Args:
        special_tokens: 要添加的特殊 token 列表
        tokenizer: 预训练的 tokenizer
    
    Returns:
        int: 实际添加的新 token 数量
    """
    # 过滤已存在的 token
    existing_tokens = set(tokenizer.get_vocab().keys())
    new_tokens = [token for token in special_tokens if token not in existing_tokens]
    
    if not new_tokens:
        print("所有 token 已存在于词表中，无需添加")
        return 0
    
    # 使用 add_special_tokens 添加
    # 需要将列表包装为 additional_special_tokens
    num_added = tokenizer.add_special_tokens({
        "additional_special_tokens": new_tokens
    })
    
    print(f"成功添加 {num_added} 个 special token: {new_tokens}")
    return num_added

