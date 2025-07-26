import os
import threading
from flask import Flask
import discord
from PIL import Image, ImageDraw, ImageFont
import easyocr
import re
import io

# === Flask Health Check HTTPサーバー ===
app = Flask(__name__)

@app.route('/')
def health():
    return "OK", 200

def run_health_server():
    print("✅ Flaskヘルスチェックサーバー起動")
    app.run(host="0.0.0.0", port=8080)

threading.Thread(target=run_health_server, daemon=True).start()

# === Discord BOT 設定 ===
TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

reader = None

# 画像クロップ用座標
base_y = 1095
row_height = 310
crop_height = 140
num_box_x  = (270, 400)
time_box_x = (400, 630)

def get_reader():
    global reader
    if reader is None:
        print("⏳ EasyOCR Reader初期化中…")
        reader = easyocr.Reader(['en'], gpu=False)
    return reader

def ocr_easyocr(image_path, min_conf=0.3):
    """OCR実行 & 信頼度フィルタ"""
    r = get_reader()
    result = r.readtext(image_path, detail=1)

    texts_only = []
    for res in result:
        # res = (box, text, confidence)
        if isinstance(res, tuple) and len(res) == 3:
            text = res[1]
            conf = res[2]
            if isinstance(conf, (int, float)) and conf >= min_conf:
                texts_only.append(text)
        elif isinstance(res, str):
            texts_only.append(res)

    return " ".join(texts_only), result  # 生テキストと座標付き両方返す

def extract_number(text):
    digits = re.sub(r"[^0-9]", "", text)
    return digits if digits else "?"

def correct_time_str(digits):
    """
    OCRの数字列を時刻に補正
    - 6桁以上なら末尾6桁
    - 秒 > 59なら調整
    - 6時間以上は存在しないので制限
    """
    digits = re.sub(r"[^0-9]", "", digits)
    if len(digits) < 4:
        return "開戦済"

    if len(digits) >= 6:
        digits = digits[-6:]  # 末尾6桁だけ取る

    hh = int(digits[0:2])
    mm = int(digits[2:4])
    ss = int(digits[4:6]) if len(digits) >= 6 else 0

    if ss > 59: ss = ss % 60
    if mm > 59: mm = mm % 60
    if hh > 5: hh = hh % 6  # 6時間以上はないので補正

    return f"{hh:02d}:{mm:02d}:{ss:02d}"

def extract_time(text):
    digits = re.sub(r"[^0-9]", "", text)
    if not digits:
        return "開戦済"
    return correct_time_str(digits)

def crop_and_ocr_easyocr(img_path):
    img = Image.open(img_path)
    lines = []

    for i in range(3):
        y1 = base_y + i * row_height
        if i == 0: y1 -= 5
        if i == 1: y1 -= 100
        if i == 2: y1 -= 200
        y2 = y1 + crop_height

        # 番号領域
        num_crop = f"/tmp/num_{i+1}.png"
        img.crop((num_box_x[0], y1, num_box_x[1], y2)).save(num_crop)
        raw_num, _ = ocr_easyocr(num_crop)
        number = extract_number(raw_num)

        # 時間領域
        time_crop = f"/tmp/time_{i+1}.png"
        img.crop((time_box_x[0], y1, time_box_x[1], y2)).save(time_crop)
        raw_time, _ = ocr_easyocr(time_crop)
        time_val = extract_time(raw_time)

        lines.append({
            "raw_num": raw_num,
            "number": number,
            "raw_time": raw_time,
            "time_val": time_val
        })

    return lines

def draw_ocr_overlay(original_path, ocr_results):
    """OCR結果を画像に書き込んで返す"""
    img = Image.open(original_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    font = ImageFont.load_default()

    y_text = 10
    for idx, line in enumerate(ocr_results, start=1):
        text = f"行{idx}: 番号={line['number']} / 時間={line['time_val']}"
        draw.text((10, y_text), text, fill=(255,0,0), font=font)
        y_text += 20

    # 一時ファイルをメモリに保存
    output_buf = io.BytesIO()
    img.save(output_buf, format="PNG")
    output_buf.seek(0)
    return output_buf

@client.event
async def on_ready():
    print(f"✅ EasyOCR Discord BOT起動: {client.user}")

@client.event
async def on_message(message):
    if message.author.bot:
        return
    if message.attachments:
        await message.channel.send("✅ EasyOCR(CPUモード)で番号＆時間を解析中…")

        for attachment in message.attachments:
            file_path = f"/tmp/{attachment.filename}"
            await attachment.save(file_path)

            # OCR結果
            lines = crop_and_ocr_easyocr(file_path)

            # 結果文字列
            result_msg = ""
            for idx, line in enumerate(lines, start=1):
                result_msg += f"行{idx} → 番号OCR: \"{line['raw_num']}\" → 抽出: {line['number']}\n"
                result_msg += f"　　　 → 時間OCR: \"{line['raw_time']}\" → 抽出: {line['time_val']}\n\n"

            # 画像にもオーバーレイ
            processed_img_buf = draw_ocr_overlay(file_path, lines)
            discord_file = discord.File(processed_img_buf, filename="ocr_result.png")

            await message.channel.send(result_msg, file=discord_file)

# BOT起動
client.run(TOKEN)