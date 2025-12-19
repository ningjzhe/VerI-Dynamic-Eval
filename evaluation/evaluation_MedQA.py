import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

def evaluate_answers(answer_path, ground_truth_path):
    # 加载预测答案（处理可能的格式问题）
    answers_df = pd.read_csv(answer_path, dtype={'answer': str}).astype({'index': 'int64'})
    
    # 加载真实答案（优化Parquet读取）
    gt_df = pd.read_parquet(ground_truth_path, engine='pyarrow')
    
    # 数据清洗与格式统一

    answers_df['clean_answer'] = answers_df['answer'].str.strip().str.upper()
    # 评估核心逻辑
    results = []
    #for idx, row in tqdm(gt_df.iterrows(), total=len(gt_df), desc="Evaluating Answers"):
    for idx, row in tqdm(gt_df.iterrows(), total=len(gt_df), desc="Evaluating Answers"):
        try:
            pred_clean = answers_df.loc[answers_df['index'] == idx, 'clean_answer'].iloc[0]
            right_answer = gt_df.loc[idx, 'reward_model']['ground_truth'].strip().upper()
            if pred_clean==right_answer:
                is_correct = True
            else:
                is_correct = False
            results.append({
                'index': idx,
                'predicted': pred_clean,
                'ground_truth': right_answer,
                'correct': is_correct
            })
            
        except Exception as e:
            print(f"处理索引{idx}时出错：{str(e)}")
            continue
    
    # 生成评估报告
    results_df = pd.DataFrame(results)
    accuracy = results_df['correct'].mean() * 100
    
    print("===== 评估报告 =====")
    print(f"总样本数: {len(results_df)}")
    print(f"整体准确率: {accuracy:.2f}%")
    print("\n===== 错误分析 =====")
    print(f"错误样本数: {len(results_df[~results_df['correct']])}")
    print("错误类型分布:")
    print(results_df[~results_df['correct']]['predicted'].value_counts())
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer-path", type=str, required=True)
    parser.add_argument("--ground-truth-path", type=str, required=True)
    args = parser.parse_args()
    
    evaluate_answers(
        answer_path=args.answer_path,
        ground_truth_path=args.ground_truth_path
    )