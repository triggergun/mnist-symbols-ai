FROM python:3.7

# 中文编码支持，Docker 容器内不乱码
ENV LANG=C.UTF-8 \
    PYTHONIOENCODING=utf-8

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
