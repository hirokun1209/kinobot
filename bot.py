import discord
from discord.ext import commands
from paddleocr import PaddleOCR
from datetime import datetime, timedelta
import re
import requests
from io import BytesIO
from PIL import Image

TOKEN = "YOUR_DISCORD_BOT_TOKEN"
intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)

ocr = PaddleOCR(use_angle_cls=True, lang='japan')

def extract_times(ocr_text):
    current_time = None
    shield_times = []

    for text in ocr_text:
        # 現在時刻 (hh:mm:ss)
        if re.match(r"\d{2}:\d{2}:\d{2}", text):
            current_time = text
        
        # 免戦時間 (mm:ss or hh:mm)
        elif re.match(r"\d{1,2}:\d{2}", text):
            shield_times.append(text)

    return current_time, shield_times

def calculate_end_times(current_time_str, shield_times):
    now = datetime.strptime(current_time_str, "%H:%M:%S")
    end_times = []
    for t in shield_times:
        parts = t.split(":")
        if len(parts) == 2:
            minutes, seconds = int(parts[0]), int(parts[1])
            delta = timedelta(minutes=minutes, seconds=seconds)
        elif len(parts) == 3:
            hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
            delta = timedelta(hours=hours, minutes=minutes, seconds=seconds)
        else:
            continue
        end_times.append((t, (now + delta).strftime("%H:%M:%S")))
    return end_times

@bot.event
async def on_message(message):
    if message.attachments:
        for attachment in message.attachments:
            if attachment.filename.lower().endswith(("png", "jpg", "jpeg")):
                # 画像をダウンロード
                img_data = await attachment.read()
                img = Image.open(BytesIO(img_data))

                # OCR実行
                result = ocr.ocr(img, cls=True)
                texts = [line[1][0] for line in result[0]]

                # 時間抽出
                current_time, shield_times = extract_times(texts)

                if current_time:
                    end_times = calculate_end_times(current_time, shield_times)
                    reply = f"現在時刻: {current_time}\n"
                    for st, end in end_times:
                        reply += f"免戦 {st} → 終了 {end}\n"
                else:
                    reply = "現在時刻が見つかりませんでした…"

                await message.channel.send(reply)

bot.run(TOKEN)