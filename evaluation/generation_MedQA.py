import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def generate_answers(model_path, parquet_path, output_path, temperature=0.0, max_length=512):
    # 加载预训练模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 读取parquet文件
    df = pd.read_parquet(parquet_path)
    
    answers = []
    failed_indices = []  # 记录无法解析的样本
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating Answers"):
        try:
            # 提取问题内容
            question = row['prompt'][0]['content']
            # 构造Prompt模板（根据实际需求调整）
            prompt_template = f"题目：{question},你的回答只能是A、B、C、D，E中的一个字母，不要添加任何解释。"
            
            # Tokenize输入
            inputs = tokenizer(prompt_template, return_tensors="pt").to(device)
            
            # 生成回答
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    temperature=temperature,
                    do_sample=False  # 禁用随机采样确保确定性
                )
            
            # 解析回答
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = generated_text.strip()[-1]  # 假设答案是最后一个单词
            # 验证答案有效性
            if answer in ['A', 'B', 'C', 'D', 'E']:
                answers.append({"index": idx, "answer": answer})
            else:
                failed_indices.append(idx)
                
        except Exception as e:
            print(f"处理索引 {idx} 时发生错误: {str(e)}")
            failed_indices.append(idx)
    
    # 保存答案文件
    answers_df = pd.DataFrame(answers)
    answers_df.to_csv(output_path, index=False)
    
    # 输出失败统计
    print(f"\n生成完成！共处理 {len(df)} 题")
    print(f"成功生成 {len(answers)} 题")
    print(f"失败 {len(failed_indices)} 题（索引：{failed_indices})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--parquet-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, default="answers.csv")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    
    generate_answers(
        model_path=args.model_path,
        parquet_path=args.parquet_path,
        output_path=args.output_path,
        temperature=args.temperature
    )