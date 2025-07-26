import os
import threading
from flask import Flask
import discord
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
import io
import re

# === Flask ヘルスチェックサーバー ===
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

# === PaddleOCR 初期化 ===
print("⏳ PaddleOCR 初期化中…")
ocr = PaddleOCR(
    lang='en',
    use_textline_orientation=False  # 最新版ではuse_angle_clsの代わり
)

# === 切り取り領域設定 ===
base_y = 1095
row_height = 310
crop_height = 140
num_box_x  = (270, 400)  # 左の番号領域
time_box_x = (400, 630)  # 右の時間領域

# === 時間補正ロジック ===
def correct_time_str(digits: str) -> str:
    """OCRの数字から時間(HH:MM:SSまたはMM:SS)を補正"""
    digits = re.sub(r"\D", "", digits)  # 数字だけ抽出

    if len(digits) <= 4:
        # 4桁以下 → MM:SS とみなす
        mm = int(digits[:2]) if len(digits) >= 2 else 0
        ss = int(digits[2:4]) if len(digits) >= 4 else 0
        mm = min(mm, 59)
        ss = min(ss, 59)
        return f"{mm:02}:{ss:02}"

    if len(digits) == 5:
        # 5桁 → M:MM:SS
        hh = int(digits[0])
        mm = int(digits[1:3])
        ss = int(digits[3:5])
    else:
        # 6桁以上 → HH:MM:SS
        hh = int(digits[:2])
        mm = int(digits[2:4])
        ss = int(digits[4:6])

    # 補正（範囲外なら繰り下げ）
    if ss >= 60:
        mm += ss // 60
        ss = ss % 60
    if mm >= 60:
        hh += mm // 60
        mm = mm % 60

    # 6時間以上はありえないので補正
    if hh >= 6:
        hh = hh % 6

    return f"{hh:02}:{mm:02}:{ss:02}"

def extract_time(text: str) -> str:
    digits = re.sub(r"\D", "", text)
    if not digits:
        return "開戦済"
    return correct_time_str(digits)

def extract_number(text: str) -> str:
    digits = re.sub(r"\D", "", text)
    return digits if digits else "?"

# === PaddleOCRで画像から数字だけ読む ===
def ocr_paddle_digits(image_path: str) -> str:
    result = ocr.ocr(image_path, cls=False)
    texts = []
    for line in result[0]:
        txt = line[1][0]
        # 数字だけ抽出
        txt_digits = re.sub(r"\D", "", txt)
        if txt_digits:
            texts.append(txt_digits)
    return "".join(texts)

# === 画像を3行分切り出してOCRする ===
def crop_and_ocr_paddle(img_path: str):
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
        raw_num = ocr_paddle_digits(num_crop)

        # 時間領域
        time_crop = f"/tmp/time_{i+1}.png"
        img.crop((time_box_x[0], y1, time_box_x[1], y2)).save(time_crop)
        raw_time = ocr_paddle_digits(time_crop)

        number = extract_number(raw_num)
        time_val = extract_time(raw_time)
        lines.append({
            "raw_num": raw_num,
            "number": number,
            "raw_time": raw_time,
            "time_val": time_val
        })
    return lines

# === OCR結果を画像に書き込む（日本語は使わない） ===
def draw_ocr_overlay(original_img_path: str, lines):
    img = Image.open(original_img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except:
        font = ImageFont.load_default()

    y_text = 20
    for idx, line in enumerate(lines, start=1):
        text = f"Row{idx}: Num={line['number']} Time={line['time_val']}"
        draw.text((10, y_text), text, fill=(255,0,0), font=font)
        y_text += 40

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

# === Discordイベント ===
@client.event
async def on_ready():
    print(f"✅ PaddleOCR Discord BOT起動: {client.user}")

@client.event
async def on_message(message):
    if message.author.bot:
        return

    if message.attachments:
        await message.channel.send("✅ PaddleOCRで番号＆時間を解析中…")

        for attachment in message.attachments:
            file_path = f"/tmp/{attachment.filename}"
            await attachment.save(file_path)

            # OCR処理
            lines = crop_and_ocr_paddle(file_path)

            # テキスト結果まとめ
            result_msg = ""
            for idx, line in enumerate(lines, start=1):
                result_msg += f"Row{idx} → NumOCR: \"{line['raw_num']}\" → 抽出: {line['number']}\n"
                result_msg += f"         → TimeOCR: \"{line['raw_time']}\" → 抽出: {line['time_val']}\n\n"

            # 処理結果画像
            processed_img_buf = draw_ocr_overlay(file_path, lines)

            await message.channel.send(result_msg)
            await message.channel.send(file=discord.File(processed_img_buf, filename="ocr_result.png"))

# BOT起動
client.run(TOKEN)