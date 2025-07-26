import os
import threading
import time
import re
from flask import Flask
import discord
from PIL import Image, ImageFilter, ImageEnhance
import easyocr

# =========================
# Flask Health Check HTTPサーバー
# =========================
app = Flask(__name__)

@app.route('/')
def health():
    return "OK", 200

def run_health_server():
    print("✅ Flaskヘルスチェックサーバー起動")
    app.run(host="0.0.0.0", port=8080)

# =========================
# Discord BOT設定
# =========================
TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# =========================
# EasyOCR 初期化 (メモリ節約のため1回だけ作る)
# =========================
reader = None
def get_reader():
    global reader
    if reader is None:
        print("⏳ EasyOCR Reader初期化中…")
        reader = easyocr.Reader(['en'], gpu=False)
    return reader

# =========================
# OCR前の画像前処理
# =========================
def preprocess_image(img_path):
    img = Image.open(img_path).convert("L")  # グレースケール
    img = img.filter(ImageFilter.SHARPEN)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)  # コントラスト強調
    tmp_path = "/tmp/preprocessed.png"
    img.save(tmp_path)
    return tmp_path

# =========================
# OCR実行
# =========================
def ocr_easyocr(image_path):
    r = get_reader()
    img_path = preprocess_image(image_path)
    result = r.readtext(img_path, detail=1)  # [(text, confidence, bbox), ...]
    # 信頼度フィルタリング
    filtered = [text for (text, conf, bbox) in result if conf >= 0.3]
    joined = " ".join(filtered)
    print(f"🔍 OCR結果: {joined}")
    return joined

# =========================
# 数字抽出 (番号用)
# =========================
def extract_number(text):
    m = re.search(r"\d{3,6}", text)  # 3～6桁の数字
    return m.group(0) if m else "?"

# =========================
# 時間補正ロジック
# =========================
def extract_time(raw_text):
    # 数字だけにする
    digits = re.sub(r"[^0-9]", "", raw_text)
    if len(digits) < 4:
        return "開戦済"

    # 長すぎる場合は末尾から6桁 or 8桁を取る
    if len(digits) > 8:
        digits = digits[-8:]

    # 4桁なら mm:ss
    if len(digits) == 4:
        mm = digits[:2]
        ss = digits[2:]
        return f"00:{mm}:{ss}"

    # 6桁なら hh:mm:ss
    if len(digits) == 6:
        hh = digits[:2]
        mm = digits[2:4]
        ss = digits[4:]
    else:
        # 8桁なら先頭2桁は無視して後ろ6桁だけ使う
        digits = digits[-6:]
        hh = digits[:2]
        mm = digits[2:4]
        ss = digits[4:]

    # 数値補正ルール
    h, m, s = int(hh), int(mm), int(ss)

    # 06:00:00 以上は存在しないので補正
    if h > 6:
        h = h % 6

    if m > 59:
        m = m % 60
    if s > 59:
        s = s % 60

    return f"{h:02}:{m:02}:{s:02}"

# =========================
# 画像から番号と時間を抽出するメイン処理
# =========================
base_y = 1095
row_height = 310
crop_height = 140
num_box_x  = (270, 400)
time_box_x = (400, 630)

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
        number = extract_number(raw_num)

        # 時間領域
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

# =========================
# Discord BOT イベント
# =========================
@client.event
async def on_ready():
    print(f"✅ EasyOCR Discord BOT起動完了: {client.user}")

@client.event
async def on_message(message):
    if message.author.bot:
        return

    if message.attachments:
        await message.channel.send("✅ 画像解析中… (CPUモード)")

        for attachment in message.attachments:
            file_path = f"/tmp/{attachment.filename}"
            await attachment.save(file_path)

            lines = crop_and_ocr_easyocr(file_path)
            result_msg = ""

            for idx, line in enumerate(lines, start=1):
                result_msg += f"行{idx} → 番号OCR: \"{line['raw_num']}\" → 抽出: {line['number']}\n"
                result_msg += f"　　　 → 時間OCR: \"{line['raw_time']}\" → 抽出: {line['time_val']}\n\n"

            await message.channel.send(result_msg)

# =========================
# 起動処理 (Koyebで落ちないように)
# =========================
def run_discord_bot():
    if not TOKEN:
        print("❌ DISCORD_TOKENが設定されていません！環境変数を確認してください。")
        while True:
            time.sleep(60)  # トークンが無い場合も終了しないように待機
    else:
        print("🔄 Discord BOT接続開始…")
        client.run(TOKEN)

if __name__ == "__main__":
    # Flaskヘルスチェックサーバー起動
    threading.Thread(target=run_health_server, daemon=True).start()

    # Discord BOT起動
    run_discord_bot()