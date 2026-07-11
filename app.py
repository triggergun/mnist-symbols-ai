import io
import logging
import os
import time
from datetime import datetime

from fastapi import FastAPI, Request, UploadFile
from PIL import Image
import numpy

# ---------- 日志配置 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mnist-symbols")

app = FastAPI()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录每个请求的方法、路径、耗时和状态码"""
    start = time.time()
    response = await call_next(request)
    elapsed = (time.time() - start) * 1000
    logger.info(
        "%s %s → %s (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    return response

# 使用绝对路径，避免运行时工作目录不同导致找不到模型文件
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# 标签映射：数字 → 符号和名称
LABEL_MAP = {
    0: {"symbol": ">", "name": "大于号"},
    1: {"symbol": "<", "name": "小于号"},
    2: {"symbol": "=", "name": "等于号"},
    3: {"symbol": "≥", "name": "大于等于"},
    4: {"symbol": "≤", "name": "小于等于"},
}

# 上传图片保存目录，兼容 Windows / Linux / macOS
SAVE_DIR = os.path.join(os.path.expanduser("~"), "mnist-predictions")
os.makedirs(SAVE_DIR, exist_ok=True)


class NeuralNetwork:

    def __init__(self):
        self.wih = numpy.load(
            os.path.join(MODEL_DIR, "wih.npy")
        )

        self.who = numpy.load(
            os.path.join(MODEL_DIR, "who.npy")
        )

        self.activation = lambda x: 1 / (1 + numpy.exp(-x))  # sigmoid 替代 scipy

    def predict(self, img):
        img = img.resize((28, 28))

        data = numpy.asarray(img)

        # 转换成训练数据格式
        data = 255 - data

        data = (data.flatten()/255.0 * 0.99) + 0.01

        hidden = self.activation(
            numpy.dot(
                self.wih,
                data
            )
        )

        output = self.activation(
            numpy.dot(
                self.who,
                hidden
            )
        )

        return int(
            numpy.argmax(output)
        )


model = NeuralNetwork()
logger.info("✅ 模型加载完成，权重文件: model/wih.npy, model/who.npy")


@app.post("/predict")
async def predict(
        file: UploadFile
):
    # 读取上传的图片字节
    contents = await file.read()

    # 保存到用户家目录，文件名: 时间戳_原始文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}_{file.filename}"
    save_path = os.path.join(SAVE_DIR, filename)
    with open(save_path, "wb") as f:
        f.write(contents)
    logger.info("图片已保存: %s", save_path)

    # 用字节流打开图片，避免二次磁盘 IO
    img = Image.open(
        io.BytesIO(contents)
    ).convert(
        "L"
    )

    label = model.predict(img)
    info = LABEL_MAP[label]

    logger.info("识别结果: label=%d, symbol=%s, name=%s", label, info["symbol"], info["name"])

    return {
        "label": label,
        "symbol": info["symbol"],
        "name": info["name"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
