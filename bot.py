import os
import re
import io
import discord
from discord.ext import commands
from PIL import Image
from paddleocr import PaddleOCR

# === Discord トークン ===
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# === PaddleOCR 初期化 ===
ocr = PaddleOCR(use_angle_cls=False, lang='en')

# === Discord Bot 設定 ===
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# =========================================
# OCR解析 → サーバー番号/番号/時間を抽出
# =========================================
def parse_ocr_results(ocr_lines):
    parsed_results = []
    current_server = None  # 現在のサーバー番号

    for line in ocr_lines:
        text = line.strip()

        # サーバー番号行 → sXXXX 形式
        server_match = re.match(r's\d+', text)
        if server_match:
            current_server = server_match.group(0)
            continue

        # 番号 + 時間行 → "6 02:38:18" 形式
        match = re.match(r'(\d+)\s+(\d{2}:\d{2}:\d{2})', text)
        if match and current_server:
            number = match.group(1)
            time = match.group(2)

            # s1281 → 防衛, それ以外 → 奪取
            prefix = "防衛" if current_server == "s1281" else "奪取"

            parsed_results.append(f"{prefix} {current_server}-{number}-{time}")

    return parsed_results

# =========================================
# OCR実行関数 (画像→文字列リスト)
# =========================================
def run_ocr_on_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    result = ocr.ocr(image, cls=False)
    extracted_lines = []

    for res in result[0]:
        text = res[1][0]
        extracted_lines.append(text)

    return extracted_lines

# =========================================
# 画像が送られたら自動でOCR & パース
# =========================================
@bot.event
async def on_message(message):
    # Bot自身のメッセージは無視
    if message.author.bot:
        return

    # 添付画像がある場合のみ処理
    if message.attachments:
        for attachment in message.attachments:
            if attachment.filename.lower().endswith((".png", ".jpg", ".jpeg")):
                await message.channel.send("⏳ 画像をOCR解析中…")

                # 画像データを取得
                image_bytes = await attachment.read()

                # OCR実行
                ocr_lines = run_ocr_on_image(image_bytes)
                if not ocr_lines:
                    await message.channel.send("⚠️ OCRで文字が読み取れませんでした。")
                    return

                # OCR結果をパースして 防衛/奪取メッセージに変換
                parsed_results = parse_ocr_results(ocr_lines)

                if parsed_results:
                    final_msg = "\n".join(parsed_results)
                    await message.channel.send(f"✅ 解析結果:\n```\n{final_msg}\n```")
                else:
                    await message.channel.send("⚠️ サーバー番号や時間が抽出できませんでした。")

    # 他のコマンドも処理する
    await bot.process_commands(message)

# =========================================
# 起動時メッセージ
# =========================================
@bot.event
async def on_ready():
    print(f"✅ Botログイン成功: {bot.user}")

# =========================================
# Bot起動
# =========================================
if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)