# ========================================
# ベースイメージ
# ========================================
FROM python:3.10-slim

# 環境変数
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DISCORD_TOKEN=""

# 作業ディレクトリ
WORKDIR /app

# ========================================
# 必要パッケージのインストール
# ========================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ========================================
# 先にnumpyを安全なバージョンでインストール
# PaddleOCRのABI不整合を防ぐため
# ========================================
RUN pip install --no-cache-dir --force-reinstall "numpy==1.23.5"

# ========================================
# OpenCVもnumpyに合わせて再インストール
# ========================================
RUN pip install --no-cache-dir --force-reinstall opencv-python==4.7.0.72

# ========================================
# 残りのライブラリ
# ========================================
RUN pip install --no-cache-dir \
    paddleocr==2.7.0.3 \
    discord.py==2.3.2

# ========================================
# BOTコードをコピー
# ========================================
COPY bot.py /app/bot.py

# ========================================
# エントリーポイント
# ========================================
CMD ["python", "bot.py"]