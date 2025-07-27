import discord
import os
import re
from io import BytesIO
from PIL import Image
from paddleocr import PaddleOCR
from datetime import timedelta
import numpy as np

TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
client = discord.Client(intents=intents)

ocr = PaddleOCR(use_angle_cls=True, lang='japan')

@client.event
async def on_ready():
    print(f'✅ ログイン成功: {client.user}')

@client.event
async def on_message(message):
    if message.author.bot:
        return

    if message.attachments:
        # 解析開始メッセージ
        processing_msg = await message.channel.send("⏳ 解析中です…")

        for attachment in message.attachments:
            if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
                img_bytes = await attachment.read()
                image = Image.open(BytesIO(img_bytes))

                # ======== 中央30%だけ残すトリミング ========
                w, h = image.size
                top = int(h * 0.35)
                bottom = int(h * 0.65)
                cropped = image.crop((0, top, w, bottom))

                # OCR用にNumPy配列へ変換
                cropped_np = np.array(cropped)

                # ======== OCR実行 ========
                result = ocr.ocr(cropped_np, cls=True)
                extracted_text = [line[1][0] for res in result for line in res]
                all_text = " ".join(extracted_text)

                # ======== サーバー番号抽出 ========
                server_match = re.search(r'\[?S\d+\]?', all_text)
                server_num = server_match.group(0).replace("[", "").replace("]", "") if server_match else "UNKNOWN"

                # ======== 越域駐騎場番号 ========
                spot_nums = re.findall(r'越域駐騎場(\d+)', all_text)

                # ======== 免戦中時間（なければ開戦済扱い） ========
                times = re.findall(r'免戦中(\d{1,2}:\d{2})', all_text)

                # メッセージ送信時刻(JST)
                base_time = message.created_at + timedelta(hours=9)

                combined = []
                for i, spot in enumerate(spot_nums):
                    # 免戦時間があれば計算
                    if i < len(times):
                        raw_time = times[i]
                        mins, secs = map(int, raw_time.split(":"))
                        delta = timedelta(minutes=mins, seconds=secs)
                        end_time = (base_time + delta).strftime("%H:%M:%S")
                        combined.append(f"{server_num}-{spot}-{end_time}")
                    else:
                        # 時間がない場合は開戦済
                        combined.append(f"{server_num}-{spot}-開戦済")

                if combined:
                    reply = "🗓 **駐機スケジュール**\n" + "\n".join(f"- {c}" for c in combined)
                else:
                    reply = "❌ 必要な駐機情報が見つかりませんでした…"

                # 「解析中です…」メッセージを編集して結果を表示
                await processing_msg.edit(content=reply)

client.run(TOKEN)