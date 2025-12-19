"""
Preprocess the MedQA dataset to parquet format
Adapted from verl/examples/data_preprocess/gsm8k.py
"""

import argparse
import os
import json
import pandas as pd
from pathlib import Path

def load_medqa_dataset(local_dataset_path):
    """
    加载 MedQA 数据集，支持 JSONL 格式
    """
    samples = []
    
    # 读取 JSONL 文件
    with open(local_dataset_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if line.strip():
                try:
                    data = json.loads(line)
                    data['line_number'] = line_num  # 添加行号用于标识
                    samples.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    continue
    
    return samples

def build_prompt(question_data):
    """
    构建 MedQA 问题的对话格式
    """
    question = question_data['question']
    options = question_data['options']
    
    # 构建选项文本 - 根据实际类型处理
    if isinstance(options, dict):
        # 如果是字典格式（如 {'A': '内容1', 'B': '内容2'}）
        options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
    elif isinstance(options, list):
        # 如果是列表格式（如 ['内容1', '内容2', '内容3']）
        # 我们需要为列表中的每个选项分配字母标签
        option_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G'][:len(options)]
        options_text = "\n".join([f"{letter}: {option}" for letter, option in zip(option_letters, options)])
    else:
        # 其他未知格式，转换为字符串
        options_text = str(options)
    
    prompt_content = f"{question}\n\n选项:\n{options_text}\n\n请只输出答案选项字母（如 A、B、C、D）。"
    
    return [
        {
            "role": "user",
            "content": prompt_content
        }
    ]

def process_medqa_sample(example, idx, split):
    """
    处理单个 MedQA 样本，转换为 VERL 格式
    """
    # 构建提示词
    prompt = build_prompt(example)
    
    # 构建 VERL 数据格式

    data = {
        "data_source": "medqa",
        "prompt": prompt,
        "ability": "medical_qa",
        "reward_model": {
            "style": "multiple_choice",
            "ground_truth": example['answer_idx'],  # 正确答案选项，如 "A", "B"
            "full_answer": example['answer'],       # 完整答案文本
            "options": example['options']          # 所有选项
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "meta_info": example.get('meta_info', ''),
            "line_number": example.get('line_number', idx)
        }
    }
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MedQA dataset for VERL")
    parser.add_argument("--local_dataset_path", required=True, 
                       help="Path to the MedQA JSONL file")
    parser.add_argument("--local_save_dir", default="./data/medqa_processed",
                       help="Directory to save processed Parquet files")
    parser.add_argument("--test_split_ratio", type=float, default=1.0,
                       help="Ratio of data to use as test set (1.0 means all data for testing)")
    
    args = parser.parse_args()
    
    # 加载原始数据
    print("Loading MedQA dataset...")
    raw_data = load_medqa_dataset(args.local_dataset_path)
    print(f"Loaded {len(raw_data)} samples")
    
    # 创建保存目录
    save_dir = Path(args.local_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理数据
    processed_samples = []
    for idx, sample in enumerate(raw_data):
        # 这里我们假设所有数据都用于测试，因为这是评估任务
        processed_sample = process_medqa_sample(sample, idx, "test")
        processed_samples.append(processed_sample)
    
    # 转换为 DataFrame 并保存
    df = pd.DataFrame(processed_samples)
    
    # 保存为 Parquet 文件
    test_file = save_dir / "UStest.parquet"
    df.to_parquet(test_file, index=False)
    
    print(f"Preprocessing completed!")
    print(f"Test set: {len(df)} samples")
    print(f"Saved to: {test_file}")
    
    # 显示第一个样本作为验证
    print("\nFirst sample preview:")
    first_sample = processed_samples[0]
    print(f"Prompt: {first_sample['prompt'][0]['content'][:200]}...")
    print(f"Ground truth: {first_sample['reward_model']['ground_truth']}")
    print(f"Ability: {first_sample['ability']}")