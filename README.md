# IR-Project

基于《马克思主义基本原理》教材的检索增强问答系统，支持：
- 混合检索：BM25 + dense embedding + RRF 融合
- OpenAI 兼容中转站 API
- 可选 LLM 生成回答
- 检索评测与生成评测

## 目录
- `main.py`：命令行入口
- `src/`：核心实现
- `data/`：原始文本、切块和评测集
- `indices/`：构建后的索引
- `results/`：评测结果
- `config.json`：API 与模型配置

## 配置
先编辑项目根目录的 `config.json`：

```json
{
  "api_key": "你的API_KEY",
  "base_url": "你的中转站地址",
  "embedding_model": "你的embedding模型名",
  "llm_model": "你的对话模型名"
}
```

示例：

```json
{
  "api_key": "sk-xxx",
  "base_url": "https://your-proxy.example.com/v1",
  "embedding_model": "Qwen/Qwen3-Embedding-4B",
  "llm_model": "gpt-4o-mini"
}
```

## 安装

```bash
pip install -r requirements.txt
```

## 使用

### 1. 构建索引

```bash
python main.py build
```

### 2. 查询

```bash
python main.py query "什么是马克思主义的基本特征？"
```

可选参数：
- `--top-k`：返回前几段，默认 `5`
- `--mode`：`bm25` / `dense` / `hybrid`，默认 `hybrid`
- `--llm`：启用 LLM 生成回答
- `--show-citations`：输出引用 chunk id

示例：

```bash
python main.py query "如何理解剩余价值？" --mode hybrid --llm
```

### 3. 评测

```bash
python main.py evaluate
```

会输出检索指标和生成指标，并将结果写入 `results/eval_report.json`。

### 4. 演示

```bash
python main.py demo
```

## 索引说明
- 建库时会同时保存 BM25、TF-IDF 组件、embedding 向量和 FAISS 索引。
- dense 检索使用 `config.json` 中的 `embedding_model`。
- 如果你更换了 embedding 模型，建议重新执行 `python main.py build` 生成新索引。

## 常见问题

### 1. 查询时报维度错误
通常是旧索引和当前 embedding 模型不一致。删除 `indices/` 后重新 `build` 即可。

### 2. LLM 不返回结果
检查 `config.json` 中的 `llm_model` 和 `base_url` 是否可用，且中转站支持该模型。

### 3. embedding 调用失败
检查 `api_key`、`base_url`、`embedding_model` 是否填写正确。

## 说明
该项目默认保留混合检索方式，不会只依赖单一 dense 检索。
