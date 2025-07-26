import os
import io
import threading
from flask import Flask, send_file
import discord
from PIL import Image, ImageDraw, ImageFont
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

# === Discord BOT 設定 ===
TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

reader = None

# === 画像の切り抜き座標設定 ===
base_y = 1095
row_height = 310
crop_height = 140
num_box_x  = (270, 400)
time_box_x = (400, 630)

# ✅ 日本語フォント対応
def get_jp_font(size=20):
    # Koyeb環境にも標準であるフォントパス
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(font_path, size=size)

def get_reader():
    global reader
    if reader is None:
        print("⏳ EasyOCR Reader初期化中…")
        reader = easyocr.Reader(['en'], gpu=False)
    return reader

# === OCR処理 ===
def ocr_easyocr(image_path):
    r = get_reader()
    # detail=1 にして信頼度も取得
    result = r.readtext(image_path, detail=1)
    # 信頼度フィルタ: 0.3以上だけ残す
    filtered_texts = [text for (bbox, text, conf) in result if conf >= 0.3]
    return " ".join(filtered_texts)

def extract_number(text):
    # 数字4〜6桁くらいを抽出
    m = re.search(r"\d{2,6}", text)
    return m.group(0) if m else "?"

def correct_time_str(digits: str) -> str:
    """OCRからの数字列を補正して hh:mm:ss 形式にする"""
    # まず数字だけに
    digits = re.sub(r"\D", "", digits)

    if len(digits) < 4:
        return "開戦済"

    # 長すぎる場合は末尾6桁を優先
    if len(digits) > 6:
        digits = digits[-6:]

    # hh mm ss に分割
    hh = int(digits[0:2])
    mm = int(digits[2:4])
    ss = int(digits[4:6])

    # 各種補正: mm, ssが60以上は繰り上げ
    if ss >= 60:
        mm += ss // 60
        ss = ss % 60
    if mm >= 60:
        hh += mm // 60
        mm = mm % 60

    # 06:00:00以上なら→上限 05:59:59 に丸める
    if hh >= 6:
        hh = 5
        mm = 59
        ss = 59

    return f"{hh:02d}:{mm:02d}:{ss:02d}"

def extract_time(text):
    digits = re.sub(r"\D", "", text)
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

        # 番号部分の切り抜き
        num_crop = f"/tmp/num_{i+1}.png"
        img.crop((num_box_x[0], y1, num_box_x[1], y2)).save(num_crop)
        raw_num = ocr_easyocr(num_crop)

        # 時間部分の切り抜き
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

# === OCR結果を画像にオーバーレイして返す ===
def draw_ocr_overlay(original_path, ocr_results):
    img = Image.open(original_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = get_jp_font(20)

    y_text = 10
    for idx, line in enumerate(ocr_results, start=1):
        # 日本語含むテキストを描画
        text = f"行{idx}: 番号={line['number']} / 時間={line['time_val']}"
        draw.text((10, y_text), text, fill=(255, 0, 0), font=font)
        y_text += 30

    output_buf = io.BytesIO()
    img.save(output_buf, format="PNG")
    output_buf.seek(0)
    return output_buf

# === Discordイベント ===
@client.event
async def on_ready():
    print(f"✅ EasyOCR Discord BOT起動: {client.user}")

@client.event
async def on_message(message):
    if message.author.bot:
        return

    # 画像添付メッセージのみ処理
    if message.attachments:
        await message.channel.send("✅ EasyOCR(CPUモード)で番号＆免戦時間を解析中…")

        for attachment in message.attachments:
            file_path = f"/tmp/{attachment.filename}"
            await attachment.save(file_path)

            # OCR解析
            lines = crop_and_ocr_easyocr(file_path)

            # テキスト結果をメッセージ送信
            result_msg = ""
            for idx, line in enumerate(lines, start=1):
                result_msg += (
                    f"行{idx} → 番号OCR: \"{line['raw_num']}\" → 抽出: {line['number']}\n"
                    f"　　　 → 時間OCR: \"{line['raw_time']}\" → 抽出: {line['time_val']}\n\n"
                )
            await message.channel.send(result_msg)

            # OCR結果を画像に描画して送る
            processed_img_buf = draw_ocr_overlay(file_path, lines)
            await message.channel.send(file=discord.File(processed_img_buf, "ocr_result.png"))

# === Discord BOT起動 ===
print("🔄 Discord BOT接続開始…")
client.run(TOKEN)