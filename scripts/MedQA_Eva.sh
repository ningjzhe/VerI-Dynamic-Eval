#!/bin/bash

# 参数配置
MODEL_PATH="/root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct"
PARQUET_PATH="/root/autodl-tmp/data/processed/processed_data/medqa_processed/mainlandtest.parquet"
ANSWER_PATH="/root/autodl-tmp/results/generated_answers.csv"
GROUND_TRUTH_PATH="/root/autodl-tmp/data/processed/processed_data/medqa_processed/mainlandtest.parquet"
# 执行评估
python /root/autodl-tmp/evaluation/evaluation_MedQA.py \
    --answer-path $ANSWER_PATH \
    --ground-truth-path $GROUND_TRUTH_PATH 