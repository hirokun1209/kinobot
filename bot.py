import os
import discord
from discord.ext import commands
from paddleocr import PaddleOCR
from PIL import Image
import io

# Discordトークンを環境変数から読み込み
TOKEN = os.getenv("DISCORD_TOKEN")
PREFIX = "!"

# PaddleOCR初期化（GPU無効化で安定化）
ocr = PaddleOCR(lang='japan', use_angle_cls=False, use_gpu=False)

# Discord Bot 初期化
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=PREFIX, intents=intents)

def crop_and_ocr(image_bytes):
    """画像を3種類のトリミングでOCRして結果を返す"""
    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size
    results = {}

    # 1. 上下35%削り
    top = int(h * 0.35)
    bottom = int(h * 0.65)
    cropped1 = img.crop((0, top, w, bottom))
    res1 = ocr.ocr(cropped1, cls=True)
    text1 = "\n".join([line[1][0] for line in res1[0]]) if res1 else "なし"
    results["上下35%削り"] = text1

    # 2. 右上20%
    crop_w = int(w * 0.2)
    crop_h = int(h * 0.2)
    cropped2 = img.crop((w - crop_w, 0, w, crop_h))
    res2 = ocr.ocr(cropped2, cls=True)
    text2 = "\n".join([line[1][0] for line in res2[0]]) if res2 else "なし"
    results["右上20%"] = text2

    # 3. 右から30%・上から20%
    crop_w2 = int(w * 0.3)
    crop_h2 = int(h * 0.2)
    cropped3 = img.crop((w - crop_w2, 0, w, crop_h2))
    res3 = ocr.ocr(cropped3, cls=True)
    text3 = "\n".join([line[1][0] for line in res3[0]]) if res3 else "なし"
    results["右30%・上20%"] = text3

    return results

@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")

@bot.event
async def on_message(message):
    if message.author.bot:
        return

    # 画像が送られたらOCR
    if message.attachments:
        await message.channel.send("📸 画像を受け取りました！OCR解析中…")

        for attachment in message.attachments:
            img_bytes = await attachment.read()
            ocr_results = crop_and_ocr(img_bytes)

            reply = "🔍 **OCR結果**\n"
            for key, text in ocr_results.items():
                reply += f"\n**{key}**\n{text}\n"

            await message.channel.send(reply)

    await bot.process_commands(message)

# BOT起動
bot.run(TOKEN)