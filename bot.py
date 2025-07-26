import os
import threading
import discord
from flask import Flask
from paddleocr import PaddleOCR
from PIL import Image
import io

# ============================================
# 🔧 必要ライブラリをインストール（libgomp.so.1エラー回避）
# ============================================
print("🔧 Installing libgomp1...")
os.system("apt-get update && apt-get install -y libgomp1")

# ============================================
# Flaskヘルスチェックサーバー
# ============================================
app = Flask(__name__)

@app.route('/')
def health():
    return "OK", 200

def run_health_server():
    print("✅ Flaskヘルスチェックサーバー起動")
    app.run(host="0.0.0.0", port=8080)

threading.Thread(target=run_health_server, daemon=True).start()

# ============================================
# Discord BOT 設定
# ============================================
TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# ============================================
# PaddleOCR 初期化
# ============================================
print("⏳ PaddleOCR 初期化中…")
ocr = PaddleOCR(lang='en', use_textline_orientation=False)  # show_log削除

# ============================================
# OCR 実行関数
# ============================================
def run_paddleocr(image_path):
    result = ocr.ocr(image_path, cls=False)
    texts = []
    for line in result:
        for box, (text, conf) in line:
            texts.append(text)
    return " ".join(texts)

# ============================================
# Discordイベント
# ============================================
@client.event
async def on_ready():
    print(f"✅ Discord BOT 起動完了: {client.user}")

@client.event
async def on_message(message):
    if message.author.bot:
        return

    if message.attachments:
        await message.channel.send("📸 PaddleOCRで画像解析中…")

        for attachment in message.attachments:
            file_path = f"/tmp/{attachment.filename}"
            await attachment.save(file_path)

            # OCR実行
            text_result = run_paddleocr(file_path)
            await message.channel.send(f"✅ OCR結果:\n```\n{text_result}\n```")

            # 画像に簡単な確認メッセージを付けて返す
            img = Image.open(file_path).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)

            await message.channel.send(file=discord.File(buf, filename="result.png"))

# ============================================
# BOT起動
# ============================================
print("🔄 Discord BOT接続開始…")
client.run(TOKEN)