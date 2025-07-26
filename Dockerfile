FROM python:3.10-bullseye

WORKDIR /app

# 依存ライブラリ
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/

# torch+cpu版をPyTorch公式URLからインストール
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

COPY bot.py /app/

CMD ["python", "bot.py"]