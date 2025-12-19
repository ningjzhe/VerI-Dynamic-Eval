import re
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import pandas as pd


@dataclass
class QueryRecord:
    """记录单次查询的信息"""
    query_id: str
    original_question: str
    query_content: str
    response: str
    timestamp: float
    is_successful: bool


@dataclass
class InferenceSession:
    """记录单次推理会话的信息"""
    session_id: str
    original_question: str
    final_answer: str = ""
    queries_made: List[QueryRecord] = field(default_factory=list)
    total_query_count: int = 0
    is_correct: bool = False


class DynamicQueryProcessor:
    """动态查询处理器"""
    
    def __init__(self, strong_model_api, base_model_api,max_queries_per_session: int = 5):
        """
        初始化动态查询处理器
        
        Args:
            strong_model_api: 强模型API接口
            max_queries_per_session: 每个推理会话的最大查询次数
        """
        self.strong_model_api = strong_model_api
        self.base_model_api=base_model_api
        self.max_queries_per_session = max_queries_per_session
        self.sessions: Dict[str, InferenceSession] = {}
        self.query_counter = 0
        
    def extract_query_from_response(self, response: str) -> Optional[str]:
        """
        从模型响应中提取查询内容
        
        Args:
            response: 模型的响应文本
            
        Returns:
            提取的查询内容，如果没有找到则返回None
        """
        # 查找 <query> 标签内的内容
        query_pattern = r"<query>(.*?)</query>"
        match = re.search(query_pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def format_query_for_strong_model(self,query_content: str) -> str:
        """
        格式化查询内容以发送给强模型
        
        Args:
            query_content: 提取的查询内容
            
        Returns:
            格式化后的查询内容
        """
        formatted_query = f"""

为了回答问题，我需要获取以下信息：
"{query_content}"

请提供相关信息。"""
        
        return formatted_query
    
    def process_inference_with_dynamic_queries(self, 
                                             session_id: str,
                                             original_question: str,
                                             initial_response: str,
                                             ground_truth: str = None) -> str:
        """
        处理带动态查询的推理过程
        
        Args:
            session_id: 推理会话ID
            original_question: 原始问题
            initial_response: 初始模型响应
            ground_truth: 正确答案（用于评估）
            
        Returns:
            最终答案
        """
        # 创建新的推理会话
        session = InferenceSession(session_id=session_id, original_question=original_question)
        self.sessions[session_id] = session
        
        # 检查初始响应是否包含查询
        current_response = initial_response
        query_count = 0
        
        while query_count < self.max_queries_per_session:
            # 提取查询内容
            query_content = self.extract_query_from_response(current_response)
            cleaned_current_response = self.extract_final_answer(current_response)
            # 如果没有查询内容，认为这是最终答案
            if not query_content:
                break
            
            # 记录查询
            query_record = QueryRecord(
                query_id=f"{session_id}_query_{query_count}",
                original_question=original_question,
                query_content=query_content,
                response="",
                timestamp=pd.Timestamp.now().timestamp(),
                is_successful=False
            )
            
            # 发送查询到强模型
            formatted_query = self.format_query_for_strong_model(query_content)
            try:
                strong_model_response = self.strong_model_api.call(formatted_query)
                query_record.response = strong_model_response
                print(f"Strong model response: {strong_model_response}")
                query_record.is_successful = True
                
                # 将强模型的响应附加到当前响应后面，供模型继续推理
                current_response = f"{cleaned_current_response}\n\n强模型反馈: {strong_model_response}"
                follow_up_prompt = f"""
                【严格格式要求】
                1. **输出内容**：仅返回选项字母（A/B/C/D/E）或<query>标签内容，禁止任何前缀、后缀或解释。
        
                2. **格式示例**：
                    - 正确输出：A
                    - 正确输出：<query>需要查询的内容</query>
                    - 错误输出：Assistant: A（需删除前缀）

                3. **原始问题与选项**：
                    - 原始问题: {original_question}
                    - 选项: A.选项A B.选项B ... E.选项E
                    - 信息： {current_response}
                4. **回答要求**：
                    - 如果有足够信息回答问题，直接输出正确选项的字母（A/B/C/D/E）。
                    - 如果信息不足，请生成一个新的查询，格式为：<query>需要查询的内容</query>，其中“需要查询的内容”应具体
                请严格执行上述要求，确保输出符合格式约束。"""
                current_response = current_response+self.base_model_api.call(follow_up_prompt)
                print(f"Response after strong model feedback: {current_response}")
            except Exception as e:
                print(f"Error calling strong model: {e}")
                query_record.response = f"Error: {str(e)}"
            # 记录查询
            session.queries_made.append(query_record)
            query_count += 1
            session.total_query_count = query_count
            
            # 如果达到了最大查询次数，停止查询
            if query_count >= self.max_queries_per_session:
                break
        
        # 提取最终答案（去除查询标记）
        final_answer = self.extract_final_answer(current_response)[-1]
        session.final_answer = final_answer
        
        # 如果提供了正确答案，评估结果
        if ground_truth:
            session.is_correct = self.evaluate_answer(final_answer, ground_truth)
            
        return final_answer
    
    def extract_final_answer(self, response: str) -> str:
        """
        从响应中提取最终答案（去除查询标记）
        
        Args:
            response: 包含查询和答案的完整响应
            
        Returns:
            最终答案
        """
        # 移除所有查询标记
        cleaned_response = re.sub(r"<query>", "", response, flags=re.DOTALL)
        cleaned_response = re.sub(r"</query>", "", cleaned_response, flags=re.DOTALL)
        # 清理多余的空白字符
        cleaned_response = cleaned_response.strip()
        
        return cleaned_response
    
    def evaluate_answer(self, predicted: str, ground_truth: str) -> bool:
        """
        评估预测答案是否正确
        
        Args:
            predicted: 预测答案
            ground_truth: 正确答案
            
        Returns:
            是否正确
        """
        # 简单的字符串匹配（可以根据需要改进）
        return predicted.strip().lower() == ground_truth.strip().lower()
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        获取特定会话的统计信息
        
        Args:
            session_id: 会话ID
            
        Returns:
            统计信息字典
        """
        if session_id not in self.sessions:
            return {}
            
        session = self.sessions[session_id]
        return {
            "session_id": session_id,
            "total_queries": session.total_query_count,
            "is_correct": session.is_correct,
            "queries": [{
                "query_content": q.query_content,
                "response": q.response,
                "successful": q.is_successful
            } for q in session.queries_made]
        }
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """
        获取整体统计信息
        
        Returns:
            整体统计信息字典
        """
        if not self.sessions:
            return {}
            
        total_sessions = len(self.sessions)
        correct_sessions = sum(1 for s in self.sessions.values() if s.is_correct)
        total_queries = sum(s.total_query_count for s in self.sessions.values())
        
        return {
            "total_sessions": total_sessions,
            "correct_sessions": correct_sessions,
            "accuracy": correct_sessions / total_sessions if total_sessions > 0 else 0,
            "total_queries": total_queries,
            "avg_queries_per_session": total_queries / total_sessions if total_sessions > 0 else 0,
            "sessions_with_queries": sum(1 for s in self.sessions.values() if s.total_query_count > 0)
        }



def create_case_study_report(session_id: str, processor: DynamicQueryProcessor) -> str:
    """
    创建案例研究报告
    
    Args:
        session_id: 会话ID
        processor: 动态查询处理器
        
    Returns:
        案例研究报告文本
    """
    stats = processor.get_session_stats(session_id)
    if not stats:
        return "No session found"
        
    session = processor.sessions[session_id]
    
    report = f"""
# Case Study Report: {session_id}

## Original Question
{session.original_question}

## Final Answer
{session.final_answer}

## Evaluation
Correct: {stats['is_correct']}

## Queries Made
Total Queries: {stats['total_queries']}
"""
    
    for i, query in enumerate(stats['queries'], 1):
        report += f"""
### Query {i}
- Content: {query['query_content']}
- Response: {query['response']}
- Successful: {query['successful']}
"""
    
    return report
