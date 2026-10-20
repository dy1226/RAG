# RAG 知识库问答助手

[![Python](https://img.shields.io/badge/Python-3.12-1f2937?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain%201.4%20%2F%20LangGraph%201.2-1f2937?style=flat-square)](https://python.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-1f2937?style=flat-square)](LICENSE)

基于大模型的本地知识库问答系统。支持 PDF、Word、TXT 等多种文档格式的上传，智能体自动检索并基于知识库内容生成准确回答。采用两阶段检索（向量宽召回 + LLM 重排）保证高相关性，支持多轮对话记忆。

## 核心功能

- **文档上传与处理**：支持 PDF / Word(.docx) / TXT / MD / CSV，自动分块切片并写入本地向量库
- **智能检索**：两阶段检索架构
  - 第一阶段：向量相似度宽召回（top-10）
  - 第二阶段：LLM 重排过滤（top-3），确保高相关性
- **多轮对话**：基于 LangGraph checkpointer，按 session_id 持久化上下文，支持追问
- **检索限定**：可指定只在某一份文档内检索，避免多文档间干扰
- **溯源展示**：每条回答附带检索到的原文片段，便于追踪信息来源
- **模型可插拔**：支持火山方舟、DeepSeek、智谱等任意 OpenAI 兼容接口，只改 .env 不动代码

## 快速开始

### 前置要求
- Python 3.12+
- 一个大模型 API Key（火山方舟 / DeepSeek / OpenAI / 智谱等）

### 1. 配置环境

```bash
# 克隆项目
cd llm-data-analyst-agent

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # macOS / Linux
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置大模型

复制 `.env.example` 为 `.env`，填入你的 API 信息：

```bash
cp .env.example .env
# 编辑 .env，填入：
# LLM_BASE_URL=https://your-api-endpoint
# LLM_API_KEY=your_api_key
# LLM_MODEL=your_model_name
```

**常用配置示例：**

**火山方舟（豆包 / DeepSeek）：**
```env
LLM_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
LLM_API_KEY=sk-xxxxxxx
LLM_MODEL=ep-xxxxxxx
```

**DeepSeek 官方：**
```env
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-xxxxxxx
LLM_MODEL=deepseek-chat
```

### 3. 启动服务

**后端：**
```bash
python main.py
```
服务将在 `http://127.0.0.1:8000` 启动。

**前端（另开一个终端）：**
```bash
cd frontend
npm install
npm run dev
```
前端将在 `http://localhost:5173` 启动。

### 4. 使用

1. 打开浏览器访问 `http://localhost:5173`
2. 上传一份 PDF / Word / TXT 文档
3. 在输入框输入问题，点击发送
4. 系统自动在知识库中检索并生成答案
5. 支持追问，助手会记住前面的对话上下文

## API 文档

### RAG 文档管理

**上传文档**
```bash
POST /rag/upload
Content-Type: multipart/form-data

file: <binary>
```

**列出已上传的文档**
```bash
GET /rag/documents
```

返回示例：
```json
{
  "success": true,
  "documents": [
    {"source": "技术方案.pdf", "chunks": 24},
    {"source": "使用说明.docx", "chunks": 18}
  ],
  "total_chunks": 42
}
```

### RAG 问答

**发起提问**
```bash
POST /rag/chat
Content-Type: application/json

{
  "question": "这份文档主要讲了什么？",
  "session_id": "user-123",        # 可选，用于多轮对话记忆
  "source": "技术方案.pdf",         # 可选，限定检索范围
  "use_rerank": true               # 可选，是否启用 LLM 重排（默认 true）
}
```

返回示例：
```json
{
  "success": true,
  "answer": "这份文档介绍了系统的架构设计...",
  "related_docs": [
    "系统包括三个主要模块...",
    "第一个模块负责...",
    "第二个模块负责..."
  ]
}
```

### 监控指标

**查询大模型调用指标**
```bash
GET /metrics
```

返回示例：
```json
{
  "calls": 45,
  "total_tokens": 12450,
  "avg_latency_ms": 1230,
  "p95_latency_ms": 2100,
  "total_cost": 0.0642,
  "failed": 2,
  "alerts": []
}
```

## 技术架构

- **后端**：FastAPI + LangChain/LangGraph
- **前端**：React 19 + Ant Design
- **向量库**：ChromaDB（本地持久化）+ ONNX 嵌入模型
- **会话记忆**：SQLite 持久化 checkpointer
- **检索算法**：两阶段（向量召回 + LLM 重排）

## 项目结构

```
llm-data-analyst-agent/
├── main.py                 # 后端服务入口
├── requirements.txt        # Python 依赖
├── .env.example           # 配置文件模板
├── chroma_db/             # 向量库数据目录
├── models/                # 本地 ONNX 嵌入模型
├── chat_memory.sqlite3    # 对话记录数据库
└── frontend/
    ├── src/
    │   ├── App.jsx       # 主应用
    │   ├── RagPanel.jsx  # 知识库问答面板
    │   ├── App.css       # 样式
    │   └── main.jsx      # 前端入口
    └── package.json      # 前端依赖
```

## 配置说明

### 环境变量（.env）

| 变量名 | 必需 | 说明 |
|------|------|------|
| `LLM_BASE_URL` | ✓ | 大模型 API 端点 |
| `LLM_API_KEY` | ✓ | 大模型 API 密钥 |
| `LLM_MODEL` | ✓ | 模型 ID（如 `ep-xxx` 或 `gpt-4`） |
| `RERANK_MODEL` | - | 检索重排用的模型，不填默认同 LLM_MODEL |
| `LLM_DISABLE_THINKING_JSON` | - | 禁用思考链参数（各厂商字段不同） |
| `CHAT_DB_PATH` | - | 对话记录数据库路径，默认 `chat_memory.sqlite3` |
| `KB_DISABLED` | - | 设为 `1` 禁用知识库（CI 环境用） |
| `ALERT_LATENCY_MS` | - | 单次调用耗时告警阈值 |
| `ALERT_FAIL_RATE` | - | 调用失败率告警阈值 |
| `ALERT_TOKENS` | - | Token 预算告警阈值 |

## 常见问题

### Q: 为什么上传的 PDF 文本提取为空？

A: 扫描版 PDF（图片形式）无法直接文本提取。可以：
- 用 OCR 工具先转文字（如 tesseract / paddle-ocr）
- 改用 Word / TXT 格式上传

### Q: 如何重置知识库？

A: 删除 `chroma_db/` 目录，重启后端自动重建：
```bash
rm -rf chroma_db/
```

### Q: 如何查看对话历史？

A: 对话记录存在 `chat_memory.sqlite3` 中。可用 SQLite 浏览器打开查看，或通过 Python：
```python
import sqlite3
conn = sqlite3.connect("chat_memory.sqlite3")
cursor = conn.cursor()
cursor.execute("SELECT * FROM checkpoints")
for row in cursor:
    print(row)
```

### Q: 能否使用本地大模型（Ollama / LM Studio）？

A: 可以。将 `.env` 改为：
```env
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=llama2
```

## 性能优化建议

1. **加快首次运行**：ONNX 嵌入模型首次加载后会被缓存，后续调用秒级返回
2. **降低成本**：设置 `RERANK_MODEL` 为更轻量的模型（如 gpt-3.5）
3. **提升检索精度**：调整 `RECALL_K` 和 `FINAL_K` 参数（main.py 里）
4. **清理过期对话**：定期清理 `chat_memory.sqlite3` 中的旧记录

## 许可证

MIT

## 反馈与支持

如有问题或建议，欢迎提 Issue 或联系作者。
