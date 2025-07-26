import os
import threading
from flask import Flask
import discord
from PIL import Image, ImageOps, ImageFilter
import easyocr
import cv2
import numpy as np
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

reader = None

# クロップ領域
base_y = 1095
row_height = 310
crop_height = 140
num_box_x  = (270, 400)
time_box_x = (400, 630)

# === EasyOCR Reader 初期化 ===
def get_reader():
    global reader
    if reader is None:
        print("⏳ EasyOCR Reader初期化中…")
        reader = easyocr.Reader(['en'], gpu=False)
    return reader

# === 画像前処理（精度UP用） ===
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ガウシアンでノイズ除去
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # 適応的二値化
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 15
    )
    # 反転して白文字黒背景に
    inverted = cv2.bitwise_not(thresh)
    tmp_path = "/tmp/preprocessed.png"
    cv2.imwrite(tmp_path, inverted)
    return tmp_path

# === EasyOCRでテキスト認識 ===
def ocr_easyocr(image_path):
    processed = preprocess_image(image_path)
    r = get_reader()
    result = r.readtext(processed, detail=1)

    # bbox, text, conf の順序で戻るので正しく解釈する
    filtered = [text for (bbox, text, conf) in result if conf >= 0.5]

    # 信頼度0.5未満しか無いならfallbackで全部連結
    if not filtered:
        filtered = [text for (_, text, _) in result]

    return " ".join(filtered)

# === 番号抽出（1〜12の数字） ===
def extract_number(text):
    m = re.search(r"\b([1-9]|1[0-2])\b", text)
    return m.group(1) if m else "?"

# === 時刻抽出（hh:mm:ss or 4桁以上の数字） ===
def extract_time(text):
    # まずhh:mm:ss形式
    m = re.search(r"\d{1,2}[:：]?\d{1,2}[:：]?\d{1,2}", text)
    if m:
        val = m.group(0).replace("：", ":")
        if len(val) == 6 and ":" not in val:
            val = f"{val[0:2]}:{val[2:4]}:{val[4:6]}"
        return val

    # 4桁以上の数字がある場合（例: 5027）
    m2 = re.search(r"\d{4,}", text)
    if m2:
        return m2.group(0)

    return "開戦済"

# === クロップして行ごとOCR ===
def crop_and_ocr_easyocr(img_path):
    img = Image.open(img_path)
    lines = []
    for i in range(3):
        y1 = base_y + i * row_height
        # 行ごとの微調整
        if i == 0: y1 -= 5
        if i == 1: y1 -= 100
        if i == 2: y1 -= 200
        y2 = y1 + crop_height

        # 番号領域クロップ
        num_crop = f"/tmp/num_{i+1}.png"
        img.crop((num_box_x[0], y1, num_box_x[1], y2)).save(num_crop)

        # 時間領域クロップ
        time_crop = f"/tmp/time_{i+1}.png"
        img.crop((time_box_x[0], y1, time_box_x[1], y2)).save(time_crop)

        # OCR実行
        raw_num = ocr_easyocr(num_crop)
        raw_time = ocr_easyocr(time_crop)

        # パターン抽出
        number = extract_number(raw_num)
        time_val = extract_time(raw_time)

        lines.append({
            "raw_num": raw_num,
            "number": number,
            "raw_time": raw_time,
            "time_val": time_val
        })
    return lines

# === Discordイベント ===
@client.event
async def on_ready():
    print(f"✅ EasyOCR Discord BOT起動: {client.user}")

@client.event
async def on_message(message):
    if message.author.bot:
        return

    if message.attachments:
        await message.channel.send("✅ EasyOCR(CPUモード)で番号＆免戦時間を解析中…")
        for attachment in message.attachments:
            file_path = f"/tmp/{attachment.filename}"
            await attachment.save(file_path)
            lines = crop_and_ocr_easyocr(file_path)

            result_msg = ""
            for idx, line in enumerate(lines, start=1):
                result_msg += f"行{idx} → 番号OCR: \"{line['raw_num']}\" → 抽出: {line['number']}\n"
                result_msg += f"　　　 → 時間OCR: \"{line['raw_time']}\" → 抽出: {line['time_val']}\n\n"

            await message.channel.send(result_msg)

client.run(TOKEN)