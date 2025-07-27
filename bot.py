import os
import discord
from discord.ext import commands
from paddleocr import PaddleOCR
from PIL import Image

# === Discord Bot Token ===
TOKEN = os.getenv("DISCORD_TOKEN")  # Koyebでは環境変数に設定

# === PaddleOCRの初期化（日本語対応） ===
ocr = PaddleOCR(use_angle_cls=True, lang='japan')

# === Discord Intents設定 ===
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# === 上下30%をトリミングする関数 ===
def crop_image_center(image_path):
    img = Image.open(image_path)
    w, h = img.size

    # 上下30%カット → 中央40%だけ残す
    top = int(h * 0.3)
    bottom = int(h * 0.7)
    cropped = img.crop((0, top, w, bottom))

    cropped_path = "/tmp/cropped_image.jpg"
    cropped.save(cropped_path)
    return cropped_path

# === OCR結果から必要な情報を抽出 ===
def extract_info(texts):
    server_name = None
    results = []

    for text in texts:
        # サーバー名抽出 [sXXXX]
        if "[s" in text:
            server_name = text.strip()

        # 免戦中 + 時間
        if "免戦中" in text:
            results.append(text.strip())

        # 越域駐騎場 + 番号
        if "越域駐騎場" in text:
            results.append(text.strip())

    return server_name, results

# === 画像メッセージイベント ===
@bot.event
async def on_message(message):
    if message.author.bot:
        return

    # 添付画像があるか確認
    if message.attachments:
        for attachment in message.attachments:
            # 一時ファイルに保存
            img_path = "/tmp/input_image.jpg"
            await attachment.save(img_path)

            # 1️⃣ 上下30%をトリミング
            cropped_path = crop_image_center(img_path)

            # 2️⃣ OCR実行
            ocr_result = ocr.ocr(cropped_path, cls=True)
            texts = [line[1][0] for block in ocr_result for line in block]

            # 3️⃣ 必要情報を抽出
            server_name, extracted = extract_info(texts)

            # 4️⃣ 返信用テキスト作成
            reply_text = "✅ **OCR結果**\n"
            if server_name:
                reply_text += f"📡 サーバー: `{server_name}`\n"
            if extracted:
                reply_text += "\n".join(f"- {t}" for t in extracted)
            else:
                reply_text += "⚠️ 必要な情報が見つかりませんでした。"

            # 5️⃣ Discordへ返信（トリミング画像も送る）
            await message.channel.send(reply_text, file=discord.File(cropped_path))

    # 他のコマンドも処理できるように
    await bot.process_commands(message)

# === Bot起動 ===
bot.run(TOKEN)