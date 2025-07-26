import os
import discord
from PIL import Image
import easyocr
import re

TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# EasyOCR 初期化 (GPU無効化)
reader = easyocr.Reader(['en'], gpu=False)

base_y = 1095
row_height = 310
crop_height = 140
num_box_x  = (270, 400)
time_box_x = (400, 630)

def ocr_easyocr(image_path):
    result = reader.readtext(image_path, detail=0)
    return " ".join(result)

def extract_number(text):
    m = re.search(r"\b([1-9]|1[0-2])\b", text)
    return m.group(1) if m else "?"

def extract_time(text):
    m = re.search(r"\d{1,2}[:：]?\d{1,2}[:：]?\d{1,2}", text)
    if m:
        val = m.group(0).replace("：", ":")
        if len(val) == 6 and ":" not in val:
            val = f"{val[0:2]}:{val[2:4]}:{val[4:6]}"
        return val
    return "開戦済"

def crop_and_ocr_easyocr(img_path):
    img = Image.open(img_path)
    lines = []
    for i in range(3):
        y1 = base_y + i * row_height
        if i == 0: y1 -= 5
        if i == 1: y1 -= 100
        if i == 2: y1 -= 200
        y2 = y1 + crop_height
        num_crop = f"/tmp/num_{i+1}.png"
        img.crop((num_box_x[0], y1, num_box_x[1], y2)).save(num_crop)
        raw_num = ocr_easyocr(num_crop)
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
