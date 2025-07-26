import os
import threading
from flask import Flask
import discord
from PIL import Image
from paddleocr import PaddleOCR
import re

# === Flaskヘルスチェック ===
app = Flask(__name__)

@app.route('/')
def health():
    return "OK", 200

def run_health_server():
    app.run(host="0.0.0.0", port=8080)

threading.Thread(target=run_health_server, daemon=True).start()

# === Discord BOT 設定 ===
TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# === PaddleOCR 初期化（CPU版） ===
print("⏳ PaddleOCR 初期化中…")
ocr = PaddleOCR(use_angle_cls=False, lang='en')

# === OCRの座標設定 ===
base_y = 1095
row_height = 310
crop_height = 140
num_box_x  = (270, 400)
time_box_x = (400, 630)

# === OCR実行 ===
def ocr_paddle(image_path):
    result = ocr.ocr(image_path, cls=False)
    if not result or not result[0]:
        return ""
    # テキストだけ抽出
    return " ".join([line[1][0] for line in result[0]])

# === 数字抽出（番号用） ===
def extract_number(text):
    m = re.search(r"\d+", text)
    return m.group(0) if m else "?"

# === 時刻補正ロジック ===
def correct_time_str(digits):
    # 数字だけ取り出す
    d = re.sub(r"\D", "", digits)
    if len(d) < 4:
        return "開戦済"

    # 2桁ずつ区切る
    parts = [d[i:i+2] for i in range(0, len(d), 2)]
    # 時分秒の候補（3つあれば使う）
    hh = int(parts[0])
    mm = int(parts[1]) if len(parts) > 1 else 0
    ss = int(parts[2]) if len(parts) > 2 else 0

    # 制限：06:00:00以上は存在しないので補正
    if hh >= 6:
        # 4桁なら mm:ss とみなす
        if len(d) == 4:
            mm, ss = int(d[:2]), int(d[2:])
            return f"{mm:02}:{ss:02}"
        # 6桁以上の場合は末尾3つを採用
        d = d[-6:]
        hh = int(d[:2])
        mm = int(d[2:4])
        ss = int(d[4:6])

    # 秒が60超えなら補正
    if ss > 59: ss = 59
    if mm > 59: mm = 59

    return f"{hh:02}:{mm:02}:{ss:02}"

def extract_time(text):
    digits = re.sub(r"\D", "", text)
    return correct_time_str(digits)

# === 画像を3行クロップしてOCR ===
def crop_and_ocr_paddle(img_path):
    img = Image.open(img_path)
    lines = []
    for i in range(3):
        y1 = base_y + i * row_height
        if i == 0: y1 -= 5
        if i == 1: y1 -= 100
        if i == 2: y1 -= 200
        y2 = y1 + crop_height

        # 番号
        num_crop = f"/tmp/num_{i+1}.png"
        img.crop((num_box_x[0], y1, num_box_x[1], y2)).save(num_crop)
        raw_num = ocr_paddle(num_crop)

        # 時刻
        time_crop = f"/tmp/time_{i+1}.png"
        img.crop((time_box_x[0], y1, time_box_x[1], y2)).save(time_crop)
        raw_time = ocr_paddle(time_crop)

        number = extract_number(raw_num)
        time_val = extract_time(raw_time)

        lines.append({
            "raw_num": raw_num,
            "number": number,
            "raw_time": raw_time,
            "time_val": time_val
        })
    return lines

# === Discord イベント ===
@client.event
async def on_ready():
    print(f"✅ PaddleOCR Discord BOT 起動: {client.user}")

@client.event
async def on_message(message):
    if message.author.bot:
        return
    if message.attachments:
        await message.channel.send("✅ PaddleOCR(CPU)で番号＆時間を解析中…")
        for attachment in message.attachments:
            file_path = f"/tmp/{attachment.filename}"
            await attachment.save(file_path)
            lines = crop_and_ocr_paddle(file_path)
            result_msg = ""
            for idx, line in enumerate(lines, start=1):
                result_msg += f"行{idx} → 番号OCR: \"{line['raw_num']}\" → 抽出: {line['number']}\n"
                result_msg += f"　　　 → 時間OCR: \"{line['raw_time']}\" → 抽出: {line['time_val']}\n\n"
            await message.channel.send(result_msg)

client.run(TOKEN)