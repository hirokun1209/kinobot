import os
import threading
from flask import Flask
import discord
from PIL import Image, ImageEnhance, ImageOps
import easyocr
import re
import cv2
import numpy as np

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

base_y = 1095
row_height = 310
crop_height = 140
num_box_x  = (270, 400)
time_box_x = (400, 630)

def get_reader():
    global reader
    if reader is None:
        print("⏳ EasyOCR Reader初期化中…")
        # 英数字専用モデルで精度UP
        reader = easyocr.Reader(['en'], gpu=False, recog_network='english_g2')
    return reader

def preprocess_image(image_path):
    """OCR前に画像を前処理して精度UP"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # コントラスト調整
    img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    # 二値化
    _, img = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)
    processed_path = image_path.replace(".png", "_proc.png")
    cv2.imwrite(processed_path, img)
    return processed_path

def ocr_easyocr(image_path):
    processed = preprocess_image(image_path)
    r = get_reader()
    result = r.readtext(processed, detail=1)
    # 信頼度50%以上の結果だけ採用
    filtered = [text for (text, conf, bbox) in result if conf >= 0.5]
    return " ".join(filtered)

def extract_number(text):
    # 数字だけ抽出
    digits = re.findall(r"\d+", text)
    if not digits:
        return "?"
    # 最後の数字を採用
    number = digits[-1]
    # 長すぎる場合は末尾4桁に制限
    if len(number) > 4:
        number = number[-4:]
    return number

def extract_time(text):
    # 数字だけ連結
    digits = "".join(re.findall(r"\d", text))
    if len(digits) >= 6:
        # hh:mm:ss に変換
        return f"{digits[0:2]}:{digits[2:4]}:{digits[4:6]}"
    elif len(digits) >= 4:
        # hh:mm:00 とみなす
        return f"{digits[0:2]}:{digits[2:4]}:00"
    elif len(digits) >= 2:
        return f"{digits[0:2]}:00:00"
    return "開戦済"

def crop_and_ocr_easyocr(img_path):
    img = Image.open(img_path)
    lines = []
    for i in range(3):
        y1 = base_y + i * row_height
        if i == 0: y1 -= 5
        if i == 1: y1 -= 100
        if i == 2: y1 -= 200
        y2 = y1 + crop_height

        # 番号部分
        num_crop = f"/tmp/num_{i+1}.png"
        img.crop((num_box_x[0], y1, num_box_x[1], y2)).save(num_crop)
        raw_num = ocr_easyocr(num_crop)

        # 時間部分
        time_crop = f"/tmp/time_{i+1}.png"
        img.crop((time_box_x[0], y1, time_box_x[1], y2)).save(time_crop)
        raw_time = ocr_easyocr(time_crop)

        number = extract_number(raw_num)
        time_val = extract_time(raw_time)

        lines.append({
            "raw_num": raw_num,
            "number": number,
            "raw_time": raw_time,
            "time_val": time_val
        })
    return lines

@client.event
async def on_ready():
    print(f"✅ EasyOCR Discord BOT起動: {client.user}")

@client.event
async def on_message(message):
    if message.author.bot:
        return
    if message.attachments:
        await message.channel.send("✅ 精度UP版 EasyOCR(CPUモード)で番号＆免戦時間を解析中…")
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