import os
import io
import discord
import numpy as np
import cv2
from paddleocr import PaddleOCR

TOKEN = os.getenv("DISCORD_TOKEN")

# ✅ PaddleOCR 初期化 (日本語対応)
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='japan'
)

# ✅ Discord クライアント設定
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True  # Discord API v2 では必須
client = discord.Client(intents=intents)

# ==========================
#  画像 → OCR処理関数
# ==========================
def crop_and_ocr(img_bytes):
    # ✅ バイト列 → numpy画像へ
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("❌ 画像のデコードに失敗しました")

    # ✅ クロップが必要ならここで処理（今はそのまま）
    cropped = img  

    # ✅ OCR実行 (np.ndarrayを渡す)
    result = ocr.ocr(cropped, cls=True)
    return result


# ==========================
#  Discordイベント処理
# ==========================
@client.event
async def on_ready():
    print(f"✅ ログイン成功: {client.user}")


@client.event
async def on_message(message):
    # BOT自身のメッセージは無視
    if message.author.bot:
        return

    # 添付画像があれば処理
    if message.attachments:
        for attachment in message.attachments:
            # 画像ファイルのみ処理
            if any(attachment.filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg"]):
                try:
                    # ✅ Discordから画像バイト取得
                    img_bytes = await attachment.read()

                    # ✅ OCR実行
                    ocr_results = crop_and_ocr(img_bytes)

                    # ✅ OCR結果を文字列化
                    text_lines = []
                    for res in ocr_results:
                        for line in res:
                            text_lines.append(line[1][0])

                    result_text = "\n".join(text_lines) if text_lines else "⚠️ テキストは検出できませんでした。"

                    await message.channel.send(f"📸 OCR結果:\n```\n{result_text}\n```")

                except Exception as e:
                    await message.channel.send(f"❌ OCR処理でエラー発生: {str(e)}")

# ==========================
#  BOT起動
# ==========================
if __name__ == "__main__":
    if not TOKEN:
        print("❌ DISCORD_TOKEN が設定されていません！")
    else:
        client.run(TOKEN)