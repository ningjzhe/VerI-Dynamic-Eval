#!/bin/bash

# 参数配置
MODEL_PATH="/root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct"
PARQUET_PATH="/root/autodl-tmp/data/processed/processed_data/medqa_processed/mainlandtest.parquet"
ANSWER_PATH="/root/autodl-tmp/results/generated_answers.csv"
GROUND_TRUTH_PATH="/root/autodl-tmp/data/processed/processed_data/medqa_processed/mainlandtest.parquet"

# 生成答案
python /root/autodl-tmp/evaluation/generation_MedQA.py \
    --model-path $MODEL_PATH \
    --parquet-path $PARQUET_PATH \
    --output-path $ANSWER_PATH \
    --temperature 0.0
