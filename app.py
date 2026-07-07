import os
from fastapi import FastAPI, UploadFile
from PIL import Image
import numpy
import scipy.special

app = FastAPI()

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


class NeuralNetwork:

    def __init__(self):
        self.wih = numpy.load(
            os.path.join(MODEL_DIR, "wih.npy")
        )

        self.who = numpy.load(
            os.path.join(MODEL_DIR, "who.npy")
        )

        self.activation = scipy.special.expit

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


@app.post("/predict")
async def predict(
        file: UploadFile
):
    img = Image.open(
        file.file
    ).convert(
        "L"
    )

    label = model.predict(img)
    info = LABEL_MAP[label]

    return {
        "label": label,
        "symbol": info["symbol"],
        "name": info["name"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
