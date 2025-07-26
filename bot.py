import os
import re
import io
import threading
from flask import Flask
import discord
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR

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

# === PaddleOCR 初期化 ===
print("⏳ PaddleOCR 初期化中…")
ocr = PaddleOCR(lang='en', use_angle_cls=False, show_log=False)

# === 画像領域設定 ===
base_y = 1095
row_height = 310
crop_height = 140
num_box_x  = (270, 400)
time_box_x = (400, 630)

# === PaddleOCRで文字認識 ===
def ocr_paddle(image_path):
    result = ocr.ocr(image_path, cls=False)
    texts = []
    for line in result:
        for word in line:
            texts.append(word[1][0])  # 認識文字列だけ抽出
    return " ".join(texts)

# === 数字抽出 ===
def extract_digits(text):
    return "".join(re.findall(r"\d", text))

# === 時間補正ロジック（06:00:00以上は切る） ===
def correct_time_str(digits):
    digits = digits.strip()
    if len(digits) < 4:
        return "開戦済"

    # 桁が足りない場合はゼロ埋め
    if len(digits) == 4:
        digits = "00" + digits
    elif len(digits) == 5:
        digits = "0" + digits

    # HHMMSSとして分割
    hh = int(digits[0:2])
    mm = int(digits[2:4])
    ss = int(digits[4:6]) if len(digits) >= 6 else 0

    # 分・秒が60超えていたら補正
    if mm > 59:
        mm = mm % 60
    if ss > 59:
        ss = ss % 60

    # 6時間超えたらあり得ないので05:59:59に制限
    if hh >= 6:
        return "05:59:59"

    return f"{hh:02d}:{mm:02d}:{ss:02d}"

# === PaddleOCRで番号＆時間OCR ===
def crop_and_ocr_paddle(img_path):
    img = Image.open(img_path)
    lines = []
    for i in range(3):
        y1 = base_y + i * row_height
        y2 = y1 + crop_height

        # 番号部分のOCR
        num_crop = f"/tmp/num_{i+1}.png"
        img.crop((num_box_x[0], y1, num_box_x[1], y2)).save(num_crop)
        raw_num = ocr_paddle(num_crop)
        number_digits = extract_digits(raw_num)

        # 時間部分のOCR
        time_crop = f"/tmp/time_{i+1}.png"
        img.crop((time_box_x[0], y1, time_box_x[1], y2)).save(time_crop)
        raw_time = ocr_paddle(time_crop)
        time_digits = extract_digits(raw_time)
        time_val = correct_time_str(time_digits)

        lines.append({
            "raw_num": raw_num,
            "number": number_digits if number_digits else "?",
            "raw_time": raw_time,
            "time_val": time_val
        })
    return lines

# === 画像にOCR結果を描画（結果画像返却用） ===
def draw_ocr_overlay(original_path, lines):
    img = Image.open(original_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    except:
        font = ImageFont.load_default()

    y_text = 10
    for idx, line in enumerate(lines, start=1):
        txt = f"行{idx}: 番号 {line['number']} / 時間 {line['time_val']}"
        draw.text((10, y_text), txt, fill=(255, 0, 0), font=font)
        y_text += 40

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

# === Discord BOTイベント ===
@client.event
async def on_ready():
    print(f"✅ PaddleOCR Discord BOT起動: {client.user}")

@client.event
async def on_message(message):
    if message.author.bot:
        return

    if message.attachments:
        await message.channel.send("✅ PaddleOCR(CPUモード)で番号＆時間を解析中…")
        for attachment in message.attachments:
            file_path = f"/tmp/{attachment.filename}"
            await attachment.save(file_path)

            # OCR解析
            lines = crop_and_ocr_paddle(file_path)

            # 結果テキスト作成
            result_msg = ""
            for idx, line in enumerate(lines, start=1):
                result_msg += f"行{idx} → 番号OCR: \"{line['raw_num']}\" → 抽出: {line['number']}\n"
                result_msg += f"　　　 → 時間OCR: \"{line['raw_time']}\" → 抽出: {line['time_val']}\n\n"

            # 結果画像を生成
            processed_img = draw_ocr_overlay(file_path, lines)
            await message.channel.send(result_msg, file=discord.File(processed_img, "result.png"))

# === BOT起動 ===
print("🔄 Discord BOT接続開始…")
client.run(TOKEN)