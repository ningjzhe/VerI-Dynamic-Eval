import argparse
import pandas as pd
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
def generate_answers(model_path, parquet_path, output_path, temperature=0.0, max_length=512):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, dtype="auto")
    processor = AutoProcessor.from_pretrained(model_path,use_fast=False)
    df = pd.read_parquet(parquet_path)
    i=0
    answers = []
    failed_count = 0
    for row in tqdm(df.to_dict('records'), total=len(df), desc="Generating Answers"):
        try:
            # 提取问题内容
            question = row['reward_model']['question']
            image_id = row['reward_model']['image_id']
            image_path = f"/root/autodl-tmp/data/path-vqa/pvqa/images/test/{image_id}.jpg"
            answer = row['reward_model']['ground_truth']
            # 构造Prompt模板（根据实际需求调整）
            if answer=='yes' or answer=='no':
                prompt_template = f"题目：{question},你的回答只能是yes或者no，不要添加任何解释。"
                question_type = 'yn'
                max_new_tokens = 1
            else:
                prompt_template = f"question：{question},please provide a brief and accurate answer(less than 8 words) for the question based on the image."
                question_type = 'open_ended'
                max_new_tokens = 32
            messages = [{"role": "user","content": [{"type": "image","image": image_path,},{"type": "text", "text": prompt_template},],}]
            # Tokenize输入
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt")
            inputs = inputs.to(model.device)
            # 解析回答
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            generated_text = output_text[0]
            # 保存结果
            answer = generated_text.strip()  
            # 验证答案有效性
            answers.append({"index": i, "answer": answer, "question_type": question_type})
            i+=1
        except Exception as e:
            print(f"处理索引 {i} 时发生错误: {str(e)}")
            failed_count += 1
            i+=1
     # 保存答案文件
    answers_df = pd.DataFrame(answers)
    answers_df.to_csv(output_path, index=False)
    
    # 输出失败统计
    print(f"\n生成完成！共处理 {len(df)} 题")
    print(f"成功生成 {len(answers)} 题")
    print(f"失败 {failed_count} 题")
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