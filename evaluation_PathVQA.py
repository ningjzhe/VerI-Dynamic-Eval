import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

import pandas as pd
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import nltk

def calculate_f1(pred_tokens, gt_tokens):
    """计算F1分数（词袋级别）"""
    common_tokens = set(pred_tokens) & set(gt_tokens)
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def evaluate_answers(answer_path, ground_truth_path):
    # 加载预测答案和真实答案
    answers_df = pd.read_csv(answer_path, dtype={'answer': str}).astype({'index': 'int64'})
    gt_df = pd.read_parquet(ground_truth_path, engine='pyarrow')
    
    # 数据清洗
    answers_df['clean_answer'] = answers_df['answer'].str.strip().str.upper()
    
    # 存储分类型结果
    results_yn = []  # 用于yn问题
    results_open = []  # 用于open_ended问题
    
    for idx, row in tqdm(gt_df.iterrows(), total=len(gt_df), desc="Evaluating Answers"):
        try:
            pred_row = answers_df.loc[answers_df['index'] == idx]
            if pred_row.empty:
                continue
            pred_clean = pred_row['clean_answer'].iloc[0]
            question_type = pred_row['question_type'].iloc[0]  # 获取问题类型
            right_answer = gt_df.loc[idx, 'reward_model']['ground_truth'].strip().upper()
            
            # 根据类型处理
            if question_type == 'yn':
                is_correct = (pred_clean == right_answer)
                results_yn.append({
                    'index': idx,
                    'predicted': pred_clean,
                    'ground_truth': right_answer,
                    'correct': is_correct
                })
            elif question_type == 'open_ended':
                # 计算开放性问题指标
                pred_tokens = word_tokenize(pred_clean.lower())
                gt_tokens = word_tokenize(right_answer.lower())
                exact_match = 1 if pred_clean == right_answer else 0
                f1 = calculate_f1(pred_tokens, gt_tokens)
                bleu1 = sentence_bleu([gt_tokens], pred_tokens, weights=(1, 0, 0, 0))
                bleu2 = sentence_bleu([gt_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0))
                bleu3 = sentence_bleu([gt_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0))
                
                results_open.append({
                    'index': idx,
                    'predicted': pred_clean,
                    'ground_truth': right_answer,
                    'exact_match': exact_match,
                    'f1': f1,
                    'bleu1': bleu1,
                    'bleu2': bleu2,
                    'bleu3': bleu3
                })
        except Exception as e:
            print(f"处理索引{idx}时出错：{str(e)}")
            continue
    
    # 生成评估报告
    print("===== 评估报告 =====")
    
    # 处理yn问题
    if results_yn:
        df_yn = pd.DataFrame(results_yn)
        yn_accuracy = df_yn['correct'].mean() * 100
        print(f"[Yes/No 问题] 样本数: {len(df_yn)}, 准确率: {yn_accuracy:.2f}%")
    else:
        print("[Yes/No 问题] 无样本")
    
    # 处理开放性问题
    if results_open:
        df_open = pd.DataFrame(results_open)
        exact_match_rate = df_open['exact_match'].mean() * 100
        avg_f1 = df_open['f1'].mean() * 100
        avg_bleu1 = df_open['bleu1'].mean() * 100
        avg_bleu2 = df_open['bleu2'].mean() * 100
        avg_bleu3 = df_open['bleu3'].mean() * 100
        
        print(f"[开放性问题] 样本数: {len(df_open)}")
        print(f"  - 精确匹配 (Exact Match): {exact_match_rate:.2f}%")
        print(f"  - 宏平均 F1: {avg_f1:.2f}%")
        print(f"  - BLEU-1: {avg_bleu1:.2f}%")
        print(f"  - BLEU-2: {avg_bleu2:.2f}%")
        print(f"  - BLEU-3: {avg_bleu3:.2f}%")
    else:
        print("[开放性问题] 无样本")
    
    # 错误分析
    print("\n===== 错误分析 =====")
    if results_yn:
        wrong_yn = df_yn[~df_yn['correct']]
        print(f"Yes/No 问题错误数: {len(wrong_yn)}")
        if not wrong_yn.empty:
            print("常见错误预测:")
            print(wrong_yn['predicted'].value_counts().head())
    if results_open:
        wrong_open = df_open[df_open['exact_match'] == 0]
        print(f"开放性问题错误数 (精确匹配): {len(wrong_open)}")
        if not wrong_open.empty:
            print("常见错误预测（前5）:")
            print(wrong_open['predicted'].value_counts().head())

# 调用函数示例
# evaluate_answers("path/to/predictions.csv", "path/to/ground_truth.parquet")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer-path", type=str, required=True)
    parser.add_argument("--ground-truth-path", type=str, required=True)
    args = parser.parse_args()
    
    evaluate_answers(
        answer_path=args.answer_path,
        ground_truth_path=args.ground_truth_path
    )