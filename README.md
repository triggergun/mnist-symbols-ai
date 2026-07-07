<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.78-009688?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker" alt="Docker">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

<h1 align="center">mnist-symbols-ai</h1>

<p align="center">
  Neural network-based handwritten math symbol recognition service<br>
  Real-time recognition of <b>5</b> common math symbols via RESTful API
</p>

<p align="center">
  <a href="README_zh-CN.md">📖 中文文档</a>
</p>

---

## 📖 Introduction

Recognize handwritten math symbols using a **two-layer fully connected neural network**. Model weights are pre-trained on a MNIST-style symbol dataset. Exposes a lightweight RESTful API for easy integration.

## 🎯 Supported Symbols

| Label | Symbol | Name |
|:---:|:---:|------|
| 0 | `>` | Greater Than |
| 1 | `<` | Less Than |
| 2 | `=` | Equals |
| 3 | `≥` | Greater Than or Equal |
| 4 | `≤` | Less Than or Equal |

## 🧠 Model Architecture

```
Input (784) → Hidden (200) → Output (5)
```

- **Input**: 28×28 grayscale image, flattened to 784-dim vector
- **Activation**: Sigmoid
- **Output**: probability distribution over 5 classes, argmax for prediction
- Weights stored in `model/` directory

## 🚀 Quick Start

### Prerequisites

- Python ≥ 3.7
- pip

### Local

```bash
# 1. Clone
git clone <your-repo-url>
cd mnist-symbols-ai

# 2. Virtual environment (recommended)
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start server
python app.py
```

Server runs at `http://localhost:8000`.

### Docker

```bash
docker build -t mnist-symbols-ai .
docker run -p 8000:8000 mnist-symbols-ai:latest
```

### Docker Compose

`docker-compose.yml`:

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

## 📡 API

### POST /predict

Upload a handwritten symbol image and get the prediction.

**Request:**

```
POST /predict
Content-Type: multipart/form-data

Field: file (image, PNG / JPG supported)
```

**Response:**

```json
{
  "label": 3,
  "symbol": "≥",
  "name": "Greater Than or Equal"
}
```

**Examples:**

```bash
curl -X POST http://localhost:8000/predict -F "file=@symbol.png"
```

```python
import requests

with open("symbol.png", "rb") as f:
    resp = requests.post("http://localhost:8000/predict", files={"file": f})
    print(resp.json())
```

### Swagger Docs

Interactive API documentation available at:

👉 [http://localhost:8000/docs](http://localhost:8000/docs)

## 📁 Project Structure

```
mnist-symbols-ai/
├── app.py              # FastAPI server + neural network inference
├── model/
│   ├── wih.npy         # Input → Hidden weights (200×784)
│   └── who.npy         # Hidden → Output weights (5×200)
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker build file
├── README.md           # English documentation
└── README_zh-CN.md     # Chinese documentation
```

## 🛠 Tech Stack

- **[FastAPI](https://fastapi.tiangolo.com/)** — High-performance web framework
- **[NumPy](https://numpy.org/)** — Numerical computation
- **[SciPy](https://scipy.org/)** — Sigmoid activation
- **[Pillow](https://python-pillow.org/)** — Image processing
- **[Uvicorn](https://www.uvicorn.org/)** — ASGI server
- **[Docker](https://www.docker.com/)** — Containerization

## 📄 License

MIT
