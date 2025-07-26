import os
import threading
from flask import Flask
import discord
from PIL import Image
import easyocr
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

# 画像の座標設定（必要に応じて調整）
base_y = 1095
row_height = 310
crop_height = 140
num_box_x  = (270, 400)
time_box_x = (400, 630)

# EasyOCR Reader 初期化
def get_reader():
    global reader
    if reader is None:
        print("⏳ EasyOCR Reader初期化中…")
        reader = easyocr.Reader(['en'], gpu=False)
    return reader

# === 番号欄OCR → 数字だけ ===
def ocr_number_only(image_path):
    r = get_reader()
    result = r.readtext(image_path, allowlist="0123456789", detail=0)
    return "".join(result)

# === 時間欄OCR（通常） ===
def ocr_easyocr(image_path):
    r = get_reader()
    result = r.readtext(image_path, detail=0)
    return " ".join(result)

# 数字抽出の後処理
def extract_number(text):
    # 数字がなければ "?" を返す
    return text if text.isdigit() and text != "" else "?"

def extract_time(text):
    # 09:31:07 / 093107 などをキャッチ
    m = re.search(r"\d{1,2}[:：]?\d{2}[:：]?\d{2}", text)
    if m:
        val = m.group(0).replace("：", ":")
        # 6桁の数字のみ → HH:MM:SS に整形
        if len(val) == 6 and ":" not in val:
            val = f"{val[0:2]}:{val[2:4]}:{val[4:6]}"
        return val
    return "開戦済"

# === クロップして番号OCRは数字専用、時間は通常OCR ===
def crop_and_ocr_easyocr(img_path):
    img = Image.open(img_path)
    lines = []
    for i in range(3):
        y1 = base_y + i * row_height
        if i == 0: y1 -= 5
        if i == 1: y1 -= 100
        if i == 2: y1 -= 200
        y2 = y1 + crop_height

        # 番号欄 → 数字専用OCR
        num_crop = f"/tmp/num_{i+1}.png"
        img.crop((num_box_x[0], y1, num_box_x[1], y2)).save(num_crop)
        raw_num = ocr_number_only(num_crop)
        number = extract_number(raw_num)

        # 時間欄 → 通常OCR
        time_crop = f"/tmp/time_{i+1}.png"
        img.crop((time_box_x[0], y1, time_box_x[1], y2)).save(time_crop)
        raw_time = ocr_easyocr(time_crop)
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