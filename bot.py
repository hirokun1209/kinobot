import os
import discord
from paddleocr import PaddleOCR

# === Discord トークン（Koyeb の環境変数で設定する） ===
TOKEN = os.getenv("DISCORD_TOKEN")

# === OCR 初期化（日本語対応） ===
ocr = PaddleOCR(use_angle_cls=True, lang='japan')

# === Discord クライアント設定 ===
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True  # メッセージ本文取得
client = discord.Client(intents=intents)

# === 起動時のログ ===
@client.event
async def on_ready():
    print(f"✅ ログイン完了: {client.user}")

# === メッセージ受信時の処理 ===
@client.event
async def on_message(message):
    # BOT自身のメッセージは無視
    if message.author == client.user:
        return

    # 添付ファイルがあるか確認
    if message.attachments:
        for attachment in message.attachments:
            # 対応する画像形式のみ処理
            if any(attachment.filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg"]):
                img_path = f"/tmp/{attachment.filename}"

                # 画像を一時保存
                await attachment.save(img_path)
                print(f"📥 画像保存: {img_path}")

                # OCRで文字認識
                result = ocr.ocr(img_path, cls=True)

                # 文字列だけ抽出
                texts = [word_info[1][0] for line in result for word_info in line]

                # 結果をDiscordに送信
                if texts:
                    reply = "✅ 読み取れた文字:\n```\n" + "\n".join(texts) + "\n```"
                else:
                    reply = "⚠️ 文字が認識できませんでした"

                await message.channel.send(reply)

# === BOT起動 ===
if __name__ == "__main__":
    if TOKEN is None:
        print("❌ DISCORD_TOKEN が設定されていません！")
    else:
        client.run(TOKEN)