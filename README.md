# Dynamic-Eval

一个用于研究“动态查询”（Dynamic Query）在问答（MedQA / PathVQA）中的评估与实验框架，包含数据预处理、答案生成与基于“强模型 + 基础模型”的动态查询流水线与分析工具。

**主要功能**
- 数据预处理：将原始数据转换为 parquet / VERL 风格数据（见 tools/data_process）。
- 生成模块：使用本地/预训练模型生成选择题答案（见 evaluation/generation_MedQA.py, evaluation/generation_PathVQA.py）。
- 评估模块：比较模型输出与 ground-truth 并输出统计与错误分析（见 evaluation/evaluation_MedQA.py, evaluation/evaluation_PathVQA.py）。
- 动态查询流水线：基于基础模型生成初始回答，按需向强模型发起检索式或补充查询，再由基础模型整合形成最终答案（见 tools/dynamic_query）。
- 结果分析：可生成统计报告、柱状图与案例研究（见 tools/dynamic_query/analyzer.py）。

项目结构（概要）
- evaluation/: 生成与评估脚本（MedQA / PathVQA）。
- generation_*.py / evaluation_*.py: 各任务的命令行入口，用于生成与评估答案。
- tools/data_process/: 数据预处理脚本（MedQA、PathVQA）。
- tools/dynamic_query/: 动态查询核心实现、强模型适配与结果分析。
- scripts/: 便捷的 shell 运行脚本（Linux/WSL）。

快速开始

1) 环境（推荐）

```bash
# 建议使用 Python 3.8+，并安装以下依赖
pip install pandas numpy pyarrow tqdm transformers torch matplotlib seaborn requests
```

2) 数据预处理（示例，MedQA）

```bash
python tools/data_process/MedQA.py --local_dataset_path path/to/medqa.jsonl --local_save_dir ./data/medqa_processed
```

3) 使用模型生成答案（示例，MedQA）

```bash
python evaluation/generation_MedQA.py \
	--model-path /path/to/local/model \
	--parquet-path ./data/medqa_processed/UStest.parquet \
	--output-path answers.csv \
	--temperature 0.0
```

4) 评估生成结果

```bash
python evaluation/evaluation_MedQA.py --answer-path answers.csv --ground-truth-path ./data/medqa_processed/UStest.parquet
```

5) 运行动态查询主流程

```bash
python tools/dynamic_query/main.py --config tools/dynamic_query/config_template.json
```

配置说明
- 配置模板位于 tools/dynamic_query/config_template.json，主要字段包括 `data_path`, `num_samples`, `api_type`, `api_key`, `model_name`, `base_model_api_type`, `base_model_path`, `base_model_device`, `max_queries_per_session`, `results_file`, `report_file`, `case_study_dir` 等。

脚本
- scripts/ 目录包含若干便于批量运行的 shell 脚本：`Dynamic_query.sh`, `MedQA_Eva.sh`, `MedQA_Gen.sh`, `PathVQA_Eva.sh`, `PathVQA_Gen.sh`。在 Windows 环境请通过 WSL、Git Bash 或等价环境运行这些脚本。

注意事项
- 大部分模型调用（local / transformers）依赖 GPU 与匹配的 CUDA；在无 GPU 情况下请使用 `base_model_device=cpu` 并注意速度较慢。
- config 中可能含有示例 API key，请务必替换为你自己的凭据并妥善保管。
- 本仓库示例代码以研究与实验为主，若要生产化使用，请补充异常处理、单元测试与配置管理。

许可
- 见项目根目录的 LICENSE 文件。



