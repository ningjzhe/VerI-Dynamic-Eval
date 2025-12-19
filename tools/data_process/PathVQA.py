"""
Preprocess the PathVQA dataset to parquet format
Adapted from MedQA preprocessing script for VERL framework
"""

import argparse
import os
import json
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

def load_pathvqa_data(data_root_dir: str, split: str = "test"):
    """
    加载PathVQA数据集指定分割的数据
    """
    samples = []
    
    # 构建QA文件路径
    qa_file = Path(data_root_dir)/f"{split}_qa.pkl"
    
    if not qa_file.exists():
        print(f"Error: QA file not found: {qa_file}")
        return samples
    
    # 加载QA数据
    try:
        with open(qa_file, 'rb') as f:
            qa_data = pickle.load(f)
        print(f"Loaded {len(qa_data)} QA pairs for {split} split")
    except Exception as e:
        print(f"Error loading {qa_file}: {e}")
        return samples
    
    # 加载图像ID到索引的映射
    img_map_file = Path(data_root_dir) / f"{split}_img_id2idx.pkl"
    img_id2idx = {}
    if img_map_file.exists():
        try:
            with open(img_map_file, 'rb') as f:
                img_id2idx = pickle.load(f)
        except Exception as e:
            print(f"Error loading image mapping {img_map_file}: {e}")
    
    # 转换数据格式
    for i, qa_item in enumerate(qa_data): # 直接遍历列表元素
    # 直接从字典元素中获取 qid
        qid = qa_item.get('qid', f'default_id_{i}') # 使用get方法更安全
        question = qa_item.get('question', '')
        answer = qa_item.get('answer', '')
        image_id = qa_item.get('image', '')
        # 构建图像路径
        image_path = f"images/{split}/{image_id}.jpg"  # 假设图像格式为JPG
        # 创建样本
        sample = {
            'qid': qid,
            'question': question,
            'answer': answer,
            'image_id': image_id,
            'image_path': image_path,
            'split': split,
            'index': i
        }
        samples.append(sample)
    
    return samples

def build_multimodal_prompt(question_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    构建PathVQA多模态问题的对话格式
    包含图像和文本问题
    """
    question = question_data['question']
    image_path = question_data['image_path']
    
    # 构建多模态提示内容
    # VERL框架可能支持多模态输入，格式参考LLaVA等多模态模型
    prompt_content = [
        {
            "type": "image",
            "image": image_path  # 图像路径，VERL会在评估时加载
        },
        {
            "type": "text", 
            "text": f"{question}\n\n请给出准确的答案。"
        }
    ]
    
    return [
        {
            "role": "user",
            "content": prompt_content
        }
    ]

def process_pathvqa_sample(example: Dict[str, Any], idx: int, split: str) -> Dict[str, Any]:
    """
    处理单个PathVQA样本，转换为VERL多模态格式
    """
    # 构建多模态提示词
    prompt = build_multimodal_prompt(example)
    
    # 构建VERL多模态数据格式
    data = {
        "data_source": "pathvqa",
        "prompt": prompt,
        "image_path": example['image_path'],  # 图像路径字段
        "ability": "pathology_vqa",
        "reward_model": {
            "style": "open_ended",  # PathVQA主要是开放性问题
            "ground_truth": example['answer'],  # 正确答案文本
            "question": example['question'],
            "image_id": example['image_id'],
            "question_type": "open_ended"  # 可以添加问题类型信息
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "qid": example.get('qid', ''),
            "image_id": example.get('image_id', '')
        }
    }
    return data

def process_all_splits(data_root_dir: str, output_dir: str):
    """
    处理所有数据分割（train, val, test）
    """
    splits = ["test"]
    
    for split in splits:
        print(f"\nProcessing {split} split...")
        
        # 加载数据
        raw_data = load_pathvqa_data(data_root_dir, split)
        if not raw_data:
            print(f"No data found for {split} split, skipping...")
            continue
        
        # 处理数据
        processed_samples = []
        for idx, sample in enumerate(raw_data):
            processed_sample = process_pathvqa_sample(sample, idx, split)
            processed_samples.append(processed_sample)
        
        # 转换为DataFrame并保存
        df = pd.DataFrame(processed_samples)
        
        # 确保输出目录存在
        split_output_dir = Path(output_dir) / "pathvqa_processed"
        split_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存为Parquet文件
        output_file = split_output_dir / f"{split}.parquet"
        df.to_parquet(output_file, index=False)
        
        print(f"{split} split processing completed!")
        print(f"{split} set: {len(df)} samples")
        print(f"Saved to: {output_file}")
        
        # 显示第一个样本作为验证
        if processed_samples:
            first_sample = processed_samples[0]
            print(f"First sample preview:")
            print(f"Image path: {first_sample['image_path']}")
            print(f"Question: {first_sample['reward_model']['question'][:100]}...")
            print(f"Ground truth: {first_sample['reward_model']['ground_truth']}")
            print(f"Ability: {first_sample['ability']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PathVQA dataset for VERL")
    parser.add_argument("--data_root_dir", required=True, 
                       help="Root directory containing PathVQA data (with images/ and qas/ subdirectories)")
    parser.add_argument("--output_dir", default="./data/processed",
                       help="Directory to save processed Parquet files")
    
    args = parser.parse_args()
    
    # 处理所有数据分割
    process_all_splits(args.data_root_dir, args.output_dir)
    
    print(f"\nPathVQA preprocessing completed!")
    print(f"All splits saved to: {args.output_dir}/pathvqa_processed/")