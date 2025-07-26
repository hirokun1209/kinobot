import os
import threading
from flask import Flask
import discord
from PIL import Image, ImageEnhance, ImageFilter
import easyocr
import re
import numpy as np
import cv2

# === Flask Health Check HTTPサーバー ===
app = Flask(__name__)

@app.route('/')
def health():
    return "OK", 200

def run_health_server():
    print("✅ Flaskヘルスチェックサーバー起動")
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

# =======================
# OCR Reader初期化
# =======================
def get_reader():
    global reader
    if reader is None:
        print("⏳ EasyOCR Reader初期化中…")
        reader = easyocr.Reader(['en'], gpu=False)
    return reader

# =======================
# OCR前の画像前処理
# =======================
def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")  # グレースケール
    img = img.filter(ImageFilter.SHARPEN)  # シャープ化
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)  # コントラストアップ
    tmp_path = "/tmp/preprocessed.png"
    img.save(tmp_path)
    return tmp_path

# =======================
# OCR本体（フォーマット安全化）
# =======================
def ocr_easyocr(image_path):
    r = get_reader()
    img_path = preprocess_image(image_path)
    result = r.readtext(img_path, detail=1)

    filtered_texts = []

    for item in result:
        # EasyOCRは環境によって戻り値が異なるので安全に判定する
        if isinstance(item, (tuple, list)) and len(item) == 3:
            # パターン1: (bbox, text, conf)
            if isinstance(item[1], str) and isinstance(item[2], (float, int)):
                text, conf = item[1], float(item[2])
            # パターン2: (text, conf, bbox)
            elif isinstance(item[0], str) and isinstance(item[1], (float, int)):
                text, conf = item[0], float(item[1])
            else:
                continue
            if conf >= 0.3:
                filtered_texts.append(text)
        elif isinstance(item, str):
            # detail=0の場合は文字列だけ
            filtered_texts.append(item)

    joined = " ".join(filtered_texts)
    print(f"🔍 OCR結果: {joined}")
    return joined

# =======================
# 数字抽出
# =======================
def extract_number(text):
    m = re.search(r"\d{2,6}", text)
    return m.group(0) if m else "?"

# =======================
# 時間補正ロジック
# =======================
def correct_time_str(raw_digits):
    """
    OCR誤認識の数列を「hh:mm:ss」形式に補正する
    - 6時間以上は存在しないので最大 05:59:59 まで
    """
    # 数字だけ残す
    digits = re.sub(r"\D", "", raw_digits)
    if len(digits) < 4:
        return "開戦済"

    # 6桁に切る
    if len(digits) > 6:
        # 後ろ6桁を優先（誤認識ノイズ前提）
        digits = digits[-6:]

    # 分割
    hh = int(digits[0:2])
    mm = int(digits[2:4])
    ss = int(digits[4:6])

    # 補正（6時間以上は無いので繰り下げ）
    if hh >= 6:
        # 6時間以上なら後ろ4桁を mm:ss とみなして、頭は 00
        hh = 0
        mm = int(digits[0:2])
        ss = int(digits[2:4])

    # 分秒補正
    if mm >= 60:
        mm = mm % 60
    if ss >= 60:
        ss = ss % 60

    return f"{hh:02d}:{mm:02d}:{ss:02d}"

# =======================
# OCR → 時間抽出
# =======================
def extract_time(text):
    digits = re.sub(r"\D", "", text)
    if not digits:
        return "開戦済"
    return correct_time_str(digits)

# =======================
# 画像クロップしてOCR
# =======================
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
        raw_num = ocr_easyocr(num_crop)

        # 時間領域
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

# =======================
# Discord BOTイベント
# =======================
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

# =======================
# 終了しないようループ待機
# =======================
def keep_alive_loop():
    try:
        client.run(TOKEN)
    except Exception as e:
        print(f"❌ BOT実行エラー: {e}")
        # 再起動ループ
        keep_alive_loop()

# BOT開始
keep_alive_loop()