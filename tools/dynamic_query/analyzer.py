"""
Analyzer for Dynamic Query Results
This module provides functionality to analyze and visualize the results of dynamic querying.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import json
from collections import defaultdict
import os


class DynamicQueryAnalyzer:
    """动态查询结果分析器"""
    
    def __init__(self, results_file: str = "dynamic_query_results.csv"):
        """
        初始化分析器
        
        Args:
            results_file: 结果文件路径
        """
        self.results_file = results_file
        self.results_df = None
        self.load_results()
    
    def load_results(self):
        """加载结果数据"""
        if os.path.exists(self.results_file):
            self.results_df = pd.read_csv(self.results_file)
            print(f"Loaded {len(self.results_df)} results from {self.results_file}")
        else:
            print(f"Results file {self.results_file} not found. Starting with empty dataset.")
            self.results_df = pd.DataFrame()
    
    def add_result(self, result_data: Dict[str, Any]):
        """
        添加新的结果数据
        
        Args:
            result_data: 结果数据字典
        """
        new_row = pd.DataFrame([result_data])
        self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
    
    def save_results(self):
        """保存结果数据"""
        if not self.results_df.empty:
            self.results_df.to_csv(self.results_file, index=False)
            print(f"Saved {len(self.results_df)} results to {self.results_file}")
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """
        计算统计数据
        
        Returns:
            统计数据字典
        """
        if self.results_df.empty:
            return {}
        
        stats = {
            "total_questions": len(self.results_df),
            "questions_with_queries": len(self.results_df[self.results_df["query_count"] > 0]),
            "questions_without_queries": len(self.results_df[self.results_df["query_count"] == 0]),
            "avg_query_count": self.results_df["query_count"].mean(),
            "max_query_count": self.results_df["query_count"].max(),
            "min_query_count": self.results_df["query_count"].min(),
            "accuracy_with_queries": self.results_df[self.results_df["query_count"] > 0]["is_correct"].mean() if len(self.results_df[self.results_df["query_count"] > 0]) > 0 else 0,
            "accuracy_without_queries": self.results_df[self.results_df["query_count"] == 0]["is_correct"].mean() if len(self.results_df[self.results_df["query_count"] == 0]) > 0 else 0,
            "overall_accuracy": self.results_df["is_correct"].mean()
        }
        
        return stats
    
    def compare_accuracies(self) -> Dict[str, float]:
        """
        比较有查询和无查询情况下的准确率
        
        Returns:
            准确率比较字典
        """
        stats = self.calculate_statistics()
        
        return {
            "accuracy_with_queries": stats["accuracy_with_queries"],
            "accuracy_without_queries": stats["accuracy_without_queries"],
            "accuracy_improvement": stats["accuracy_with_queries"] - stats["accuracy_without_queries"] if stats["accuracy_without_queries"] > 0 else 0
        }
    
    def plot_query_distribution(self, save_path: str = "query_distribution.png"):
        """
        绘制查询分布图
        
        Args:
            save_path: 保存路径
        """
        if self.results_df.empty:
            print("No data to plot")
            return
        
        plt.figure(figsize=(10, 6))
        query_counts = self.results_df["query_count"].value_counts().sort_index()
        plt.bar(query_counts.index, query_counts.values)
        plt.xlabel("Number of Queries")
        plt.ylabel("Frequency")
        plt.title("Distribution of Queries per Question")
        plt.xticks(range(int(self.results_df["query_count"].max()) + 1))
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Query distribution plot saved to {save_path}")
    
    def plot_accuracy_comparison(self, save_path: str = "accuracy_comparison.png"):
        """
        绘制准确率对比图
        
        Args:
            save_path: 保存路径
        """
        comparison = self.compare_accuracies()
        
        if not comparison:
            print("No data to plot")
            return
        
        labels = ['With Queries', 'Without Queries']
        accuracies = [comparison["accuracy_with_queries"], comparison["accuracy_without_queries"]]
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(labels, accuracies, color=['skyblue', 'lightcoral'])
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison: With vs Without Dynamic Queries')
        plt.ylim(0, 1)
        
        # 在柱状图上添加数值标签
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                     f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Accuracy comparison plot saved to {save_path}")
    
    def generate_summary_report(self) -> str:
        """
        生成摘要报告
        
        Returns:
            报告文本
        """
        stats = self.calculate_statistics()
        comparison = self.compare_accuracies()
        
        if not stats:
            return "No data available for report generation."
        
        report = f"""
# Dynamic Query Analysis Report

## Overall Statistics
- Total Questions: {stats['total_questions']}
- Questions with Queries: {stats['questions_with_queries']} ({stats['questions_with_queries']/stats['total_questions']*100:.1f}%)
- Questions without Queries: {stats['questions_without_queries']} ({stats['questions_without_queries']/stats['total_questions']*100:.1f}%)
- Average Query Count: {stats['avg_query_count']:.2f}
- Max Query Count: {stats['max_query_count']}
- Min Query Count: {stats['min_query_count']}

## Accuracy Analysis
- Overall Accuracy: {stats['overall_accuracy']:.3f}
- Accuracy with Queries: {stats['accuracy_with_queries']:.3f}
- Accuracy without Queries: {stats['accuracy_without_queries']:.3f}
- Accuracy Improvement: {comparison['accuracy_improvement']:.3f}
"""
        
        return report
    
    def export_detailed_results(self, export_path: str = "detailed_results.json"):
        """
        导出详细结果
        
        Args:
            export_path: 导出路径
        """
        if self.results_df.empty:
            print("No data to export")
            return
        
        # 转换为字典列表
        detailed_results = self.results_df.to_dict('records')
        
        # 保存为JSON文件
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        print(f"Detailed results exported to {export_path}")


def create_sample_analysis_data(analyzer: DynamicQueryAnalyzer):
    """
    创建示例分析数据（用于演示）
    
    Args:
        analyzer: 分析器实例
    """
    # 创建示例数据
    sample_data = [
        {"question_id": "q001", "query_count": 2, "is_correct": True, "original_question": "卧位腰椎穿刺，脑脊液压力正常值是（　　）。"},
        {"question_id": "q002", "query_count": 0, "is_correct": False, "original_question": "另一种医学问题..."},
        {"question_id": "q003", "query_count": 1, "is_correct": True, "original_question": "第三个问题..."},
        {"question_id": "q004", "query_count": 3, "is_correct": True, "original_question": "第四个问题..."},
        {"question_id": "q005", "query_count": 0, "is_correct": False, "original_question": "第五个问题..."},
    ]
    
    for data in sample_data:
        analyzer.add_result(data)
    
    analyzer.save_results()


# 示例用法
if __name__ == "__main__":
    # 创建分析器
    analyzer = DynamicQueryAnalyzer("sample_results.csv")
    
    # 创建示例数据
    create_sample_analysis_data(analyzer)
    
    # 生成报告
    report = analyzer.generate_summary_report()
    print(report)
    
    # 绘制图表
    analyzer.plot_query_distribution("sample_query_distribution.png")
    analyzer.plot_accuracy_comparison("sample_accuracy_comparison.png")
    
    # 导出详细结果
    analyzer.export_detailed_results("sample_detailed_results.json")
