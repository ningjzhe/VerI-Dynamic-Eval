"""
Main Module for Dynamic Query Task (Q2)
This module integrates all components to run the dynamic query inference task.
"""

import os
import sys
import json
import argparse
import pandas as pd
from typing import Dict, Any
from pathlib import Path
import torch
# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from tools.dynamic_query.dynamic_query import DynamicQueryProcessor, create_case_study_report
from tools.dynamic_query.strong_model_api import create_strong_model_api
from tools.dynamic_query.analyzer import DynamicQueryAnalyzer


def load_test_data(data_path: str, num_samples: int = 100) -> pd.DataFrame:
    """
    加载测试数据
    
    Args:
        data_path: 数据文件路径
        num_samples: 采样数量
        
    Returns:
        测试数据DataFrame
    """
    # 读取数据
    df = pd.read_parquet(data_path)
    
    # 如果指定了采样数量，进行采样
    if num_samples > 0 and len(df) > num_samples:
        df = df.sample(n=num_samples, random_state=42)
    
    return df


def run_dynamic_query_inference(config: Dict[str, Any]):
    """
    运行动态查询推理任务
    
    Args:
        config: 配置字典
    """
    # 加载测试数据
    print("Loading test data...")
    test_data = load_test_data(config["data_path"], config["num_samples"])
    print(f"Loaded {len(test_data)} test samples")
    
    # 创建强模型API
    print("Initializing strong model API...")
    strong_model_api = create_strong_model_api(
        config["api_type"],
        api_key=config.get("api_key", ""),
        model_name=config.get("model_name", "qwen-plus")
    )
    # 创建基础模型API（用于生成初始响应）
    print("Initializing base model API...")
    base_model_api = create_strong_model_api(
        config.get("base_model_api_type", "local"),
        model_path=config.get("base_model_path", "/path/to/your/qwen2.5-7b-instruct"),
        device=config.get("base_model_device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    
    # 创建动态查询处理器
    print("Initializing dynamic query processor...")
    processor = DynamicQueryProcessor(
        strong_model_api,
        base_model_api,
        max_queries_per_session=config["max_queries_per_session"]
    )
    
    # 创建分析器
    print("Initializing analyzer...")
    analyzer = DynamicQueryAnalyzer(config["results_file"])
    
    # 处理每个测试样本
    print("Processing test samples...")
    for idx, row in test_data.iterrows():
        question_id = f"q{idx:03d}"
        original_question = row["prompt"][0]["content"]  # 假设这是问题内容
        ground_truth = row["reward_model"]["ground_truth"]  # 正确答案
        # 按空行分割内容，保留问题部分和选项部分
        parts = original_question.split('\n\n')  # 根据空行分割
        clean_question = '\n\n'.join(parts[:2])   # 保留前两部分（问题和选项）

        # 调用基础模型生成初始响应
        try:
            prompt = f"""
            【严格格式要求】
            1. **输出内容**：仅返回选项字母（A/B/C/D/E）或<query>标签内容，禁止任何前缀、后缀或解释。
        
            2. **格式示例**：
            - 正确输出：A 
            - 正确输出： <query>需要查询的内容</query>
            - 错误输出：Assistant: A（需删除前缀）

            3. **原始问题与选项**：
            - 原始问题: {clean_question}
            - 选项: A.选项A B.选项B ... E.选项E
            4. **回答要求**：
            - 如果有足够信息回答问题，直接输出正确选项的字母（A/B/C/D/E）。
            - 如果信息不足，请生成一个新的查询，格式为：<query>需要查询的内容</query>，其中“需要查询的内容”应具体
            请严格执行上述要求，确保输出符合格式约束。"""
            print(prompt)
            initial_response = base_model_api.call(prompt)
            print(f"Initial response from base model: {initial_response}")
        except Exception as e:
            print(f"Error calling base model: {e}")
        
        # 处理带动态查询的推理
        final_answer = processor.process_inference_with_dynamic_queries(
            question_id, clean_question, initial_response, ground_truth
        )
        
        # 获取会话统计信息
        session_stats = processor.get_session_stats(question_id)
        
        # 添加结果到分析器
        result_data = {
            "question_id": question_id,
            "original_question": clean_question,
            "initial_response": initial_response,
            "final_answer": final_answer,
            "ground_truth": ground_truth,
            "query_count": session_stats["total_queries"],
            "is_correct": session_stats["is_correct"]
        }
        analyzer.add_result(result_data)
        
        # 打印进度
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx+1}/{len(test_data)} samples")
    
    # 保存结果
    print("Saving results...")
    analyzer.save_results()
    
    # 生成报告
    print("Generating analysis report...")
    report = analyzer.generate_summary_report()
    with open(config["report_file"], "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to {config["report_file"]}")
    
    # 生成案例研究报告
    print("Generating case study reports...")
    generate_case_studies(processor, config["case_study_dir"], config["num_case_studies"])
    
    # 生成图表
    print("Generating charts...")
    analyzer.plot_query_distribution(config["query_distribution_plot"])
    analyzer.plot_accuracy_comparison(config["accuracy_comparison_plot"])
    
    print("Dynamic query inference task completed!")

def generate_case_studies(processor: DynamicQueryProcessor, output_dir: str, num_cases: int = 3):
    """
    生成案例研究报告
    
    Args:
        processor: 动态查询处理器
        output_dir: 输出目录
        num_cases: 案例数量
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 选择一些有查询的会话作为案例
    case_sessions = [sid for sid, session in processor.sessions.items() if session.total_query_count > 0][:num_cases]
    
    for i, session_id in enumerate(case_sessions, 1):
        report = create_case_study_report(session_id, processor)
        report_file = Path(output_dir) / f"case_study_{i:02d}.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Case study {i} saved to {report_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Run Dynamic Query Inference Task (Q2)")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # 运行动态查询推理任务
    run_dynamic_query_inference(config)


if __name__ == "__main__":
    main()
