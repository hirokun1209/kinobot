import os
import discord
from discord.ext import commands
from paddleocr import PaddleOCR
from datetime import datetime, timedelta
import re

TOKEN = os.getenv("DISCORD_TOKEN")  # Koyebの環境変数で設定
bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())

# OCR初期化（日本語＋英語対応）
ocr = PaddleOCR(use_angle_cls=True, lang="japan")

# 右上の時間フォーマット例: 12:34 とか 23:59
time_pattern = re.compile(r"(\d{1,2}):(\d{2})")

def calc_future_time(base_time_str, add_minutes):
    """右上時間にOCR抽出時間を加算"""
    base = datetime.strptime(base_time_str, "%H:%M")
    new_time = base + timedelta(minutes=add_minutes)
    return new_time.strftime("%H:%M")

@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")

@bot.command()
async def ocrtime(ctx):
    """画像の文字をOCRして、右上時間＋OCR内の時間を計算"""
    if not ctx.message.attachments:
        return await ctx.send("画像を添付してください！")

    attachment = ctx.message.attachments[0]
    img_path = "/tmp/input.jpg"
    await attachment.save(img_path)

    # OCR実行
    result = ocr.ocr(img_path, cls=True)
    text_blocks = [line[1][0] for line in result[0]]
    all_text = "\n".join(text_blocks)
    await ctx.send(f"📖 OCR結果:\n```\n{all_text}\n```")

    # 右上時間を抽出
    top_time_match = time_pattern.search(all_text)
    if not top_time_match:
        return await ctx.send("右上の時間が見つかりませんでした。")

    base_time = f"{top_time_match.group(1).zfill(2)}:{top_time_match.group(2)}"
    await ctx.send(f"🕒 右上時間: {base_time}")

    # OCR結果から加算すべき時間（分数）を探す例: "免戦時間 30分"
    add_minutes = 0
    for line in text_blocks:
        m = re.search(r"(\d{1,3})分", line)
        if m:
            add_minutes = int(m.group(1))
            break

    if add_minutes > 0:
        new_time = calc_future_time(base_time, add_minutes)
        await ctx.send(f"⏩ {add_minutes}分後の時間は **{new_time}** です！")
    else:
        await ctx.send("加算する時間が見つかりませんでした。")