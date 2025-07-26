import os
import threading
from flask import Flask
import discord
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
import io
import re

# === Flask Health Check HTTPサーバー ===
app = Flask(__name__)

@app.route('/')
def health():
    return "OK", 200

def run_health_server():
    app.run(host="0.0.0.0", port=8080)

threading.Thread(target=run_health_server, daemon=True).start()

# === Discord BOT ===
TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# === PaddleOCR 初期化（安定版 2.7対応） ===
print("⏳ PaddleOCR 初期化中…")
ocr = PaddleOCR(use_angle_cls=False, lang='en')  # ✅ show_log削除＆旧バージョン互換OK

# === 時間補正ロジック ===
def correct_time_str(digits: str) -> str:
    """OCR誤認識補正: 6桁以内の数字をhh:mm:ssに近い形に補正"""
    digits = re.sub(r'\D', '', digits)  # 数字以外除去
    if len(digits) <= 4:  # 4桁なら mm:ss
        mm = int(digits[:2])
        ss = int(digits[2:4]) if len(digits) >= 4 else 0
        return f"{mm:02}:{ss:02}"
    elif len(digits) == 5:  # 5桁なら mmm:ss だと仮定
        mm = int(digits[:3]) % 60
        ss = int(digits[3:5])
        return f"{mm:02}:{ss:02}"
    elif len(digits) >= 6:  # 6桁以上なら hh:mm:ss
        hh = int(digits[:2]) % 6  # 6時間超えない補正
        mm = int(digits[2:4]) % 60
        ss = int(digits[4:6]) % 60
        return f"{hh:02}:{mm:02}:{ss:02}"
    return "??:??"

# === PaddleOCRでOCRする関数 ===
def ocr_paddle(image_path):
    result = ocr.ocr(image_path, cls=False)
    texts = []
    if result and isinstance(result[0], list):
        for line in result[0]:
            txt = line[1][0]
            texts.append(txt)
    return " ".join(texts)

# === OCR結果を画像にオーバーレイ表示する ===
def draw_ocr_overlay(image_path, ocr_texts):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    y_offset = 10
    for line in ocr_texts:
        draw.text((10, y_offset), line, fill=(255, 0, 0), font=font)
        y_offset += 20

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

# === Discordメッセージ処理 ===
@client.event
async def on_ready():
    print(f"✅ PaddleOCR Discord BOT起動: {client.user}")

@client.event
async def on_message(message):
    if message.author.bot:
        return
    if message.attachments:
        await message.channel.send("✅ PaddleOCRで解析中…")
        for attachment in message.attachments:
            file_path = f"/tmp/{attachment.filename}"
            await attachment.save(file_path)

            raw_text = ocr_paddle(file_path)
            print("OCR RAW:", raw_text)

            # 数字だけ抽出
            digits_only = re.findall(r'\d+', raw_text)
            times = [correct_time_str(d) for d in digits_only]

            reply = "📖 OCR結果\n"
            for d, t in zip(digits_only, times):
                reply += f"  数字: `{d}` → 時間補正: **{t}**\n"

            # 画像にOCR結果を描画
            overlay_img = draw_ocr_overlay(file_path, digits_only)
            await message.channel.send(reply, file=discord.File(overlay_img, "ocr_result.png"))

client.run(TOKEN)