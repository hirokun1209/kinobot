import os
import threading
import time
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

# =====================
# EasyOCR Reader初期化
# =====================
def get_reader():
    global reader
    if reader is None:
        print("⏳ EasyOCR Reader初期化中…")
        reader = easyocr.Reader(['en'], gpu=False)
    return reader

# =====================
# OCR結果を安全に文字列化
# =====================
def ocr_easyocr(image_path):
    """OCR実行 & 信頼度フィルタ（どんな形式でも安全に文字列化）"""
    r = get_reader()
    result = r.readtext(image_path, detail=1)

    filtered_texts = []

    for res in result:
        # --- パターン1: (text, conf, bbox)
        if isinstance(res, tuple) and len(res) >= 2:
            text, conf = res[0], res[1]
            # 信頼度が数値ならフィルタ
            if isinstance(conf, (int, float)):
                if conf >= 0.2:
                    filtered_texts.append(str(text))
            else:
                # 信頼度が文字列なら無条件追加
                filtered_texts.append(str(text))

        # --- パターン2: ネストされたリスト
        elif isinstance(res, list):
            # ネスト内を全部文字列化して連結
            nested_texts = [str(x) for x in res if isinstance(x, (str, int, float))]
            if nested_texts:
                filtered_texts.append("".join(nested_texts))

        # --- パターン3: すでに文字列
        elif isinstance(res, str):
            filtered_texts.append(res)

    # すべて文字列化した上で連結
    return " ".join(filtered_texts)

# =====================
# 数字抽出 (番号)
# =====================
def extract_number(text):
    m = re.search(r"\d{3,6}", text)
    return m.group(0) if m else "?"

# =====================
# 時間補正ロジック
# =====================
def correct_time_str(digits: str):
    """OCRが出した数字列を HH:MM:SS に補正する（06時間以上はない想定）"""
    digits = re.sub(r"\D", "", digits)  # 数字以外除去
    if len(digits) < 4:
        return "開戦済"

    # まず末尾6桁を優先して見る
    if len(digits) >= 6:
        hh = int(digits[-6:-4])
        mm = int(digits[-4:-2])
        ss = int(digits[-2:])
    elif len(digits) == 5:
        hh = int(digits[0])
        mm = int(digits[1:3])
        ss = int(digits[3:])
    else:
        hh = 0
        mm = int(digits[0:2])
        ss = int(digits[2:4])

    # 正規化
    if ss >= 60:
        ss -= 60
        mm += 1
    if mm >= 60:
        mm -= 60
        hh += 1

    # 6時間以上は存在しない → 超えたら引く
    if hh >= 6:
        hh = hh % 6

    return f"{hh:02}:{mm:02}:{ss:02}"

# =====================
# 時間抽出
# =====================
def extract_time(text):
    digits = re.sub(r"\D", "", text)
    if not digits:
        return "開戦済"
    return correct_time_str(digits)

# =====================
# 画像切り出し & OCR
# =====================
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

# =====================
# Discordイベント
# =====================
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

# =====================
# BOT起動（終了しない）
# =====================
def start_bot():
    print("🔄 Discord BOT接続開始…")
    client.run(TOKEN)

while True:
    try:
        start_bot()
    except Exception as e:
        print(f"❌ BOTエラー再起動: {e}")
        time.sleep(5)