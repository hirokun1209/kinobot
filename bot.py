import os
import threading
import re
from flask import Flask
import discord
from PIL import Image
import easyocr

# === Flask ヘルスチェックサーバー ===
app = Flask(__name__)

@app.route('/')
def health():
    return "OK", 200

def run_health_server():
    print("✅ Flaskヘルスチェックサーバー起動")
    app.run(host="0.0.0.0", port=8080)

# デーモンスレッドでFlask起動
threading.Thread(target=run_health_server, daemon=True).start()

# === Discord BOT ===
TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

reader = None

# 画像切り出し座標
base_y = 1095
row_height = 310
crop_height = 140
num_box_x  = (270, 400)
time_box_x = (400, 630)

def get_reader():
    """EasyOCR初期化"""
    global reader
    if reader is None:
        print("⏳ EasyOCR Reader初期化中…")
        reader = easyocr.Reader(['en'], gpu=False)
    return reader

def ocr_easyocr(image_path):
    """OCR実行 & 信頼度フィルタ"""
    r = get_reader()
    result = r.readtext(image_path, detail=1)
    # detail=1 → [(テキスト, 信頼度, bbox), ...]
    filtered_texts = []
    for res in result:
        if isinstance(res, tuple) and len(res) >= 2:
            text, conf = res[0], res[1]
            # 信頼度が数値ならフィルタ
            if isinstance(conf, (int, float)):
                if conf >= 0.2:
                    filtered_texts.append(text)
            else:
                # 信頼度が文字列だったら無条件で追加
                filtered_texts.append(text)
        elif isinstance(res, str):
            filtered_texts.append(res)

    joined = " ".join(filtered_texts)
    return joined

def extract_number(text):
    """番号OCR結果から数字だけ抽出"""
    m = re.search(r"\d+", text)
    return m.group(0) if m else "?"

def correct_time_str(raw_digits):
    """
    OCR誤認識の数列を「hh:mm:ss」形式に補正する
    - 6時間以上は存在しないので最大 05:59:59 まで
    - 桁不足の場合は安全に補完
    """
    digits = re.sub(r"\D", "", raw_digits)  # 数字だけ残す

    # 桁不足なら開戦済扱い
    if len(digits) < 4:
        return "開戦済"

    # 桁不足なら0埋め
    while len(digits) < 6:
        digits += "0"

    # 桁多すぎなら後ろ6桁だけ使う
    if len(digits) > 6:
        digits = digits[-6:]

    # 安全に数値化
    hh = int(digits[0:2] or 0)
    mm = int(digits[2:4] or 0)
    ss = int(digits[4:6] or 0)

    # 補正（6時間以上はない → 時間と分秒をシフト）
    if hh >= 6:
        # 先頭2桁を分に解釈
        mm = hh
        hh = 0

    # 分秒補正
    if mm >= 60:
        mm = mm % 60
    if ss >= 60:
        ss = ss % 60

    return f"{hh:02d}:{mm:02d}:{ss:02d}"

def extract_time(text):
    """時間OCR結果から補正済みhh:mm:ssを取得"""
    digits = re.sub(r"\D", "", text)
    if not digits:
        return "開戦済"
    return correct_time_str(digits)

def crop_and_ocr_easyocr(img_path):
    """3行分の番号＆時間OCR処理"""
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

# === BOT実行 ===
print("🔄 Discord BOT接続開始…")
client.run(TOKEN)