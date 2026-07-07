<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.78-009688?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker" alt="Docker">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

<h1 align="center">mnist-symbols-ai</h1>

<p align="center">
  基于神经网络的数学符号手写识别服务<br>
  通过 RESTful API 实时识别 <b>5 种</b>常见数学符号
</p>

<p align="center">
  <a href="README.md">📖 English Docs</a>
</p>

---

## 📖 简介

基于**两层全连接神经网络**的手写数学符号识别服务。模型权重通过 MNIST 风格符号数据集训练得到，提供轻量级 RESTful API，易于集成。

## 🎯 识别能力

| 标签 | 符号 | 名称 |
|:---:|:---:|------|
| 0 | `>` | 大于号 |
| 1 | `<` | 小于号 |
| 2 | `=` | 等于号 |
| 3 | `≥` | 大于等于 |
| 4 | `≤` | 小于等于 |

## 🧠 模型架构

```
输入层 (784) → 隐藏层 (200) → 输出层 (5)
```

- **输入**：28×28 灰度图片，展平为 784 维向量
- **激活函数**：Sigmoid
- **输出**：5 个类别的概率分布，取最大值作为预测结果
- 权重文件位于 `model/` 目录

## 🚀 快速开始

### 环境要求

- Python ≥ 3.7
- pip

### 本地运行

```bash
# 1. 克隆项目
git clone <your-repo-url>
cd mnist-symbols-ai

# 2. 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux / macOS

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动服务
python app.py
```

服务运行在 `http://localhost:8000`。

### Docker 运行

```bash
docker build -t mnist-symbols-ai .
docker run -p 8000:8000 mnist-symbols-ai:latest
```

### Docker Compose

创建 `docker-compose.yml`：

```yaml
services:
  mnist-symbols:
    image: mnist-symbols-ai:latest
    ports:
      - "8000:8000"
    restart: unless-stopped
```

```bash
docker compose up -d
```

## 📡 API 文档

### POST /predict

上传手写符号图片，返回识别结果。

**请求：**

```
POST /predict
Content-Type: multipart/form-data

字段: file (图片文件, 支持 PNG / JPG 等格式)
```

**响应：**

```json
{
  "label": 3,
  "symbol": "≥",
  "name": "大于等于"
}
```

**示例：**

```bash
curl -X POST http://localhost:8000/predict -F "file=@symbol.png"
```

```python
import requests

with open("symbol.png", "rb") as f:
    resp = requests.post("http://localhost:8000/predict", files={"file": f})
    print(resp.json())
```

### Swagger 文档

启动后访问交互式 API 文档：

👉 [http://localhost:8000/docs](http://localhost:8000/docs)

## 📁 项目结构

```
mnist-symbols-ai/
├── app.py              # FastAPI 服务 + 神经网络推理
├── model/
│   ├── wih.npy         # 输入层 → 隐藏层权重 (200×784)
│   └── who.npy         # 隐藏层 → 输出层权重 (5×200)
├── requirements.txt    # Python 依赖
├── Dockerfile          # Docker 构建文件
├── README.md           # 英文文档
└── README_zh-CN.md     # 中文文档
```

## 🛠 技术栈

- **[FastAPI](https://fastapi.tiangolo.com/)** — 高性能 Web 框架
- **[NumPy](https://numpy.org/)** — 数值计算
- **[SciPy](https://scipy.org/)** — Sigmoid 激活函数
- **[Pillow](https://python-pillow.org/)** — 图像处理
- **[Uvicorn](https://www.uvicorn.org/)** — ASGI 服务器
- **[Docker](https://www.docker.com/)** — 容器化部署

## 📄 License

MIT
