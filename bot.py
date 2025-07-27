import discord
import os
import re
from io import BytesIO
from PIL import Image
from paddleocr import PaddleOCR
from datetime import timedelta

# 環境変数からトークン取得（Koyebでもそのまま動く）
TOKEN = os.getenv("DISCORD_TOKEN")

# Discordクライアント設定
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
client = discord.Client(intents=intents)

# PaddleOCR 初期化（日本語対応）
ocr = PaddleOCR(use_angle_cls=True, lang='japan')

@client.event
async def on_ready():
    print(f'✅ ログイン成功: {client.user}')

@client.event
async def on_message(message):
    # BOT自身のメッセージは無視
    if message.author.bot:
        return

    # 画像が添付されているメッセージだけ処理
    if message.attachments:
        for attachment in message.attachments:
            # jpg/png/jpegのみ対応
            if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
                # 画像を読み込み
                img_bytes = await attachment.read()
                image = Image.open(BytesIO(img_bytes))

                # ======== 中央30%だけ残すトリミング ========
                w, h = image.size
                top = int(h * 0.35)
                bottom = int(h * 0.65)
                cropped = image.crop((0, top, w, bottom))

                # トリミング後の画像をバッファに保存
                buf = BytesIO()
                cropped.save(buf, format="JPEG")
                buf.seek(0)

                # ======== OCR実行 ========
                result = ocr.ocr(cropped, cls=True)
                extracted_text = [line[1][0] for res in result for line in res]

                # テキストまとめ
                all_text = " ".join(extracted_text)

                # ======== サーバー番号を抽出 (例: [S1245]) ========
                server_match = re.search(r'\[?S\d+\]?', all_text)
                server_num = server_match.group(0).replace("[","").replace("]","") if server_match else "UNKNOWN"

                # ======== 越域駐騎場の番号だけ抽出 ========
                spot_nums = re.findall(r'越域駐騎場(\d+)', all_text)

                # ======== 免戦中の時間 (MM:SS 形式) ========
                times = re.findall(r'免戦中(\d{1,2}:\d{2})', all_text)

                # ======== Discordメッセージの送信時間 (JSTに変換) ========
                base_time = message.created_at + timedelta(hours=9)

                combined = []
                for i in range(min(len(spot_nums), len(times))):
                    raw_time = times[i]

                    # 免戦時間を timedelta に変換
                    parts = raw_time.split(":")
                    if len(parts) == 2:
                        mins = int(parts[0])
                        secs = int(parts[1])
                        delta = timedelta(minutes=mins, seconds=secs)
                    else:
                        delta = timedelta(seconds=0)

                    # 終了時刻を計算 (JST)
                    end_time = (base_time + delta).strftime("%H:%M:%S")

                    # 例: S1245-7-42:20 → 終了 18:12:20
                    combined.append(f"{server_num}-{spot_nums[i]}-{raw_time} → 終了 {end_time}")

                # ======== 結果メッセージ作成 ========
                if combined:
                    reply = f"✅ **OCR結果**\n📡 サーバー: `{server_num}`\n" + "\n".join(f"- {c}" for c in combined)
                else:
                    reply = "❌ 必要な情報が見つかりませんでした…"

                # トリミング画像と一緒に返信
                file = discord.File(buf, filename="cropped.jpg")
                await message.channel.send(reply, file=file)

# ======== BOT起動 ========
client.run(TOKEN)