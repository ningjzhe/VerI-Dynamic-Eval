"""
Strong Model API Interface for Dynamic Query
This module provides interfaces to communicate with stronger models 
"""

import requests
import json
import time
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
class MultiStopCriteria(StoppingCriteria):
    def __init__(self, stop_sequences, tokenizer, device):
        super().__init__()
        # 将字符串序列转换为token ID列表
        self.stop_sequences = [
            tokenizer.encode(seq, add_special_tokens=False) 
            for seq in stop_sequences
        ]
        self.device = device

    def __call__(self, input_ids, scores, **kwargs):
        # 获取当前生成的token序列
        current_tokens = input_ids[0].tolist()
        
        # 检查每个停止序列
        for stop_seq in self.stop_sequences:
            # 确保有足够长度的token可供匹配
            if len(current_tokens) >= len(stop_seq):
                # 检查末尾是否匹配
                if current_tokens[-len(stop_seq):] == stop_seq:
                    return True
        return False

class StrongModelAPI(ABC):
    """强模型API抽象基类"""
    
    @abstractmethod
    def call(self, query: str, **kwargs) -> str:
        """
        调用强模型API
        
        Args:
            query: 查询内容
            **kwargs: 其他参数
            
        Returns:
            模型响应
        """
        pass

class LocalModelAPI(StrongModelAPI):
    """本地模型API实现"""
    
    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化本地模型API
        
        Args:
            model_path: 本地模型路径
            device: 运行设备 ("cuda" 或 "cpu")
        """
        self.model_path = model_path
        self.device = device
        
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.model.eval()
        print(f"Model loaded successfully on {device}")
        
    def call(self, query: str, **kwargs) -> str:
        """
        调用本地模型生成响应
        
        Args:
            query: 查询内容（问题）
            **kwargs: 其他参数
            
        Returns:
             模型响应
        """
        try:
            # 编码输入
            inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
            # 生成响应
            eos_token_id = self.tokenizer.eos_token_id
            attention_mask = inputs.attention_mask
            print("Generating response from local model...")
            stopping_criteria = StoppingCriteriaList([
                MultiStopCriteria(
                    stop_sequences=['答案是A', '答案是B', '答案是C', '答案是D', '答案是E'],  # 多个停止序列
                    tokenizer=self.tokenizer,
                    device=self.device
                )
            ])

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=kwargs.get("max_tokens", 100),
                    do_sample=True,
                    eos_token_id=eos_token_id,  # 关键参数
                    top_p=0.9,
                    temperature=0.1,
                    attention_mask=attention_mask,
                    stopping_criteria=stopping_criteria
                )
             # 解码输出
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
             # 如果模型会重复输入，需要去掉输入部分
            if response.startswith(query):
                response = response[len(query):].strip()
                
            return response
        except Exception as e:
            raise Exception(f"Error calling local model: {e}")

class OpenAIAPI(StrongModelAPI):
    
    def __init__(self, api_key: str, model_name: str = "qwen-max", base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        """
        初始化OpenAI API
        
        Args:
            api_key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def call(self, query: str, **kwargs) -> str:
        """
        调用OpenAI API
        
        Args:
            query: 查询内容
            **kwargs: 其他参数
            
        Returns:
            模型响应
        """
        url = f"{self.base_url}/chat/completions"
        
        # 构建请求参数
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ],
            "max_tokens": kwargs.get("max_tokens", 300),
            "temperature": kwargs.get("temperature", 0.3),
            "top_p": kwargs.get("top_p", 0.9)
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                raise Exception(f"Unexpected API response format: {result}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {e}")
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to decode API response: {e}")


def create_strong_model_api(api_type: str, **kwargs) -> StrongModelAPI:
    """
    创建强模型API实例
    
    Args:
        api_type: API类型 ("bailian", "openai", "mock")
        **kwargs: API特定参数
        
    Returns:
        强模型API实例
    """
    if api_type == "openai":
        return OpenAIAPI(
            api_key=kwargs.get("api_key", ""),
            model_name=kwargs.get("model_name", "qwen-max")
        )
    elif api_type == "local":
        return LocalModelAPI(
            model_path=kwargs.get("model_path", ""),
            device=kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            )
    else:
        raise ValueError(f"Unsupported API type: {api_type}")


# 示例用法
if __name__ == "__main__":
    # 创建模拟API进行测试
    api = create_strong_model_api("openai")
    
    # 测试调用
    query = "卧位腰椎穿刺时脑脊液的正常压力范围是多少？"
    try:
        response = api.call(query)
        print(f"Query: {query}")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
