import discord
import asyncio
import os
from paddleocr import PaddleOCR
from io import BytesIO

TOKEN = os.environ.get("DISCORD_TOKEN")  # 環境変数から取得

if not TOKEN:
    print("❌ ERROR: DISCORD_TOKEN が設定されていません")
    exit(1)

ocr = PaddleOCR(use_angle_cls=True, lang='japan')

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

async def run_ocr(image_bytes: bytes):
    image_stream = BytesIO(image_bytes)
    result = ocr.ocr(image_stream, cls=True)
    texts = []
    for line in result[0]:
        detected_text = line[1][0]
        texts.append(detected_text)
    return texts

@client.event
async def on_ready():
    print(f"✅ ログイン完了: {client.user}")

@client.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    if message.attachments:
        for attachment in message.attachments:
            if attachment.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                await message.channel.send("📸 画像を解析中です…")
                img_bytes = await attachment.read()
                texts = await asyncio.to_thread(run_ocr, img_bytes)
                if texts:
                    reply = "✅ 読み取れた文字:\n```\n" + "\n".join(texts) + "\n```"
                else:
                    reply = "⚠️ 文字が読み取れませんでした"
                await message.channel.send(reply)

if __name__ == "__main__":
    print("🚀 BOTを起動します...")
    client.run(TOKEN)