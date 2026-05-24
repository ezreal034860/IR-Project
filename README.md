
# Hybrid RAG for Chinese Textbook QA

一个基于中文教材的 Hybrid RAG（Retrieval-Augmented Generation）问答系统，支持：

- BM25 稀疏检索
- TF-IDF + FAISS 稠密检索
- RRF 混合召回
- 抽取式回答生成
- 可选 LLM 增强生成
- 自动化检索评估

适用于：

- 中文教材问答
- 本地知识库
- RAG 系统学习
- IR（信息检索）实验
- NLP 课程项目

---

# 系统架构

```mermaid
flowchart LR
  A[Word 教材 .doc] --> B[文本提取]
  B --> C[分块 + 元数据]
  C --> D1[BM25 稀疏索引]
  C --> D2[TF-IDF + FAISS 稠密索引]
  D1 --> E[混合检索 RRF]
  D2 --> E
  E --> F[答案生成]
  F --> G[评估报告]
````

---

# 功能模块

## 文档处理

* 从 `.doc` 教材提取文本
* 自动按章节与长度进行分块
* 当前数据集约 `623` 个 chunk

---

## 稀疏检索（Sparse Retrieval）

使用：

* BM25
* 中文字符 / 二元组分词

特点：

* 中文术语匹配效果好
* 对教材专有名词召回稳定

---

## 稠密检索（Dense Retrieval）

使用：

* TF-IDF 向量化
* FAISS 余弦相似度检索

特点：

* 可支持语义相似召回
* 检索速度快

---

## 混合检索（Hybrid Retrieval）

采用：

* BM25
* Dense Retrieval
* RRF（Reciprocal Rank Fusion）

进行融合排序。

优势：

* 综合关键词匹配与语义召回能力
* 提高整体鲁棒性

---

## 生成模块

默认：

* 抽取式生成（直接返回相关片段）

可选：

* 配置 `OPENAI_API_KEY`
* 启用 LLM 摘要式回答

---

## 评估模块

内置：

* 20 条 QA 测试集
* 自动评估多种检索方式

支持指标：

* Precision@K
* Recall@K
* MRR
* nDCG@K
* Hit@K

---

# 项目结构

```text
proj/
├── data/
│   ├── raw_text.txt          # 已提取的教材全文
│   ├── chunks.json           # 分块结果
│   └── eval_qa.json          # 评估问答集（20题）
│
├── indices/                  # 检索索引
│
├── results/
│   └── eval_report.json      # 评估报告
│
├── src/                      # 核心模块
│
├── scripts/
│   └── extract_doc.py
│
└── main.py                   # 命令行入口
```

---

# 快速开始

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

---

## 2. 构建索引

```bash
python main.py build
```

---

## 3. 提问检索

```bash
python main.py query "什么是矛盾？" --show-citations
```

---

## 4. 运行完整评估

```bash
python main.py evaluate
```

---

## 5. 一键演示

```bash
python main.py demo
```

---

## 6. 启用 LLM 生成

配置环境变量：

```bash
export OPENAI_API_KEY=your_api_key
```

Windows：

```powershell
set OPENAI_API_KEY=your_api_key
```

然后：

```bash
python main.py query "剩余价值是什么？" --llm
```

---

# 评估结果（20 题测试集）

| 检索方式   | Precision@5 | Recall@5 | MRR   | nDCG@5 | Hit@5 |
| ------ | ----------- | -------- | ----- | ------ | ----- |
| BM25   | 0.770       | 0.237    | 0.933 | 0.796  | 1.000 |
| Dense  | 0.730       | 0.231    | 0.833 | 0.740  | 0.950 |
| Hybrid | 0.770       | 0.237    | 0.875 | 0.781  | 1.000 |

---

# 生成质量

## 抽取式生成

* 关键词覆盖率：`84.8%`
* 字符 F1：约 `0.10`

说明：

由于抽取式生成返回较长原文片段，因此与短参考答案的字面重叠较低，属于预期现象。

---

# 实验结论

* 三种方式 `Hit@5` 均 ≥ `95%`
* Top-5 检索基本能够命中相关内容
* BM25 在 MRR 上略优
* 中文教材中的术语匹配效果较强
* Hybrid Retrieval 在 Precision 与 Hit 上更均衡
* 抽取式生成覆盖率高
* 若结合 LLM，可进一步生成流畅摘要式答案

---

# 评估设计说明

`data/eval_qa.json` 包含：

* 20 道典型教材问题
* 覆盖多个核心章节

例如：

* 物质与意识
* 矛盾
* 剩余价值
* 科学社会主义

相关性标注规则：

> 若文档块包含 ≥2 个标注关键词，则视为相关文档。

据此计算：

* P@K
* R@K
* MRR
* nDCG@K

---

# 后续改进方向

可进一步：

* 扩充评估集
* 人工标注 gold chunk
* 接入 sentence-transformers 中文 embedding
* 使用 BGE / BCE embedding 模型
* 引入 Cross-Encoder reranker
* 支持多轮对话记忆
* 增加 Web UI
* 接入向量数据库（Milvus / FAISS / Chroma）

---

# 技术栈

* Python
* BM25
* TF-IDF
* FAISS
* NumPy
* scikit-learn
* OpenAI API（可选）

---

# License

MIT License

```
```
