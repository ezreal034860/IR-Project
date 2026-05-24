# IR-Project

基于《马克思主义基本原理》教材构建的检索增强生成（RAG）项目。当前版本已改为使用 OpenAI Embedding API 生成向量表示，并通过 FAISS 进行语义检索。

## 项目功能

- 从教材文本中自动切分 chunk
- 调用 OpenAI embedding 模型为 chunk 建立向量索引
- 对用户问题进行 embedding 并检索最相关片段
- 支持抽取式回答
- 支持可选的 LLM 生成回答
- 支持检索与生成评测

## 技术路线

1. 先把原始教材文本清洗并切分为多个段落块。
2. 对每个 chunk 调用 OpenAI embedding 接口，得到向量。
3. 使用 FAISS 建立向量索引并持久化到本地。
4. 查询时同样调用 embedding 接口，把问题向量化后做相似度检索。
5. 将检索到的片段交给抽取式回答器或 LLM 回答器生成结果。

## 目录结构

```text
IR-Project/
├─ config.json
├─ main.py
├─ requirements.txt
├─ data/
├─ indices/
├─ results/
├─ scripts/
│  └─ extract_doc.py
└─ src/
   ├─ config.py
   ├─ evaluate.py
   ├─ generator.py
   ├─ indexer.py
   ├─ preprocess.py
   ├─ rag.py
   └─ retriever.py
```

## 环境要求

- Python 3.10+
- Windows
- 已安装 Microsoft Word（仅在使用 `.doc` 文档提取脚本时需要）
- 可访问 OpenAI API

## 安装依赖

```bash
pip install -r requirements.txt
```

如果你要使用 LLM 回答，还需要安装带 `langchain-openai` 的环境，`requirements.txt` 已经包含。

## 配置

项目不再从环境变量读取 API Key 和模型名，而是统一从 `config.json` 读取。

配置文件示例：

```json
{
  "api_key": "PASTE_YOUR_OPENAI_API_KEY_HERE",
  "embedding_model": "text-embedding-3-small",
  "llm_model": "gpt-4o-mini"
}
```

说明：

- `api_key`：OpenAI API Key
- `embedding_model`：用于构建和查询向量索引的 embedding 模型
- `llm_model`：当启用 `--llm` 时用于生成答案的模型

## 数据准备

项目默认读取：

- 原始文本：`data/raw_text.txt`
- 评测集：`data/eval_qa.json`

如果你手头是 `.doc` 文档，可以先用：

```bash
python scripts/extract_doc.py "你的教材.doc" -o data/raw_text.txt
```

然后再执行切分和建库。

## 使用方法

### 1. 构建索引

```bash
python main.py build
```

这一步会：

- 读取 `data/raw_text.txt`
- 自动切分 chunk
- 调用 OpenAI embedding API
- 生成并保存 FAISS 索引到 `indices/`

### 2. 检索问答

```bash
python main.py query "什么是马克思主义的基本特征？"
```

常用参数：

- `--top-k`：返回的检索片段数量
- `--llm`：使用 LLM 生成答案，不加则使用抽取式回答
- `--show-citations`：显示引用的 chunk id

示例：

```bash
python main.py query "如何理解剩余价值？" --top-k 5 --show-citations
python main.py query "实践和认识的关系是什么？" --llm
```

### 3. 评测

```bash
python main.py evaluate
```

输出内容包括：

- 检索指标：`precision@k`、`recall@k`、`mrr`、`ndcg@k`、`hit@k`
- 生成指标：`char_f1`、`keyword_coverage`

### 4. 演示模式

```bash
python main.py demo
```

会自动：

- 如果没有索引，则先构建索引
- 运行几条示例问题
- 输出评测摘要

## 输出文件

- `indices/`：索引文件目录
- `results/eval_report.json`：评测结果
- `data/chunks.json`：切分后的 chunk 数据

## 实现说明

- 当前检索不再使用 BM25 或 TF-IDF。
- 向量检索完全依赖 OpenAI embedding + FAISS。
- `indices/meta.json` 会保存当前索引所用的 embedding 模型名。
- 抽取式回答会直接拼接检索到的原文片段，适合验证检索效果。
- `--llm` 模式下会调用配置文件里指定的 LLM 模型生成答案。

## 常见问题

### 1. 提示找不到 `config.json`

请确保项目根目录下存在 `config.json`，并填写有效的 `api_key`。

### 2. 提示 OpenAI API 调用失败

通常是以下原因之一：

- `api_key` 无效
- 账号没有对应模型权限
- 网络不可达

### 3. 索引构建很慢

因为需要对所有 chunk 调用 embedding API。教材越长，耗时越久。

### 4. 为什么没有 BM25 选项了

因为当前版本已经改成纯 embedding 语义检索，不再混合 BM25。

## 许可证

未单独指定许可证，默认按课程项目使用。
