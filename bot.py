import discord
import os
import re
from io import BytesIO
from PIL import Image
from paddleocr import PaddleOCR
from datetime import timedelta, datetime
import numpy as np

TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
client = discord.Client(intents=intents)

ocr = PaddleOCR(use_angle_cls=True, lang='japan')

def crop_center_30(image: Image.Image):
    """中央30%を残すトリミング"""
    w, h = image.size
    top = int(h * 0.35)
    bottom = int(h * 0.65)
    return image.crop((0, top, w, bottom))

def crop_topright_time_area(image: Image.Image):
    """右上(上20% × 右30%)をトリミング"""
    w, h = image.size
    top = 0
    bottom = int(h * 0.2)
    left = int(w * 0.7)
    right = w
    return image.crop((left, top, right, bottom))

def extract_time_from_text(text: str):
    """OCRテキストからHH:MM:SSを抽出"""
    m = re.search(r"(\d{1,2}):(\d{2}):(\d{2})", text)
    if m:
        h, m_, s = map(int, m.groups())
        return datetime.now().replace(hour=h, minute=m_, second=s, microsecond=0)
    return None

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

                # ======== 基準時間取得用の右上20%x30%トリミング ========
                time_area_img = crop_topright_time_area(image)
                time_np = np.array(time_area_img)
                time_result = ocr.ocr(time_np, cls=True)
                time_text = " ".join([line[1][0] for res in time_result for line in res])

                # デバッグ: OCRで読み取った右上の文字
                print("右上OCR結果:", time_text)

                # 時間を抽出（例: 17:31:22）
                base_time = extract_time_from_text(time_text)
                if not base_time:
                    # 取れなかったら投稿時刻
                    base_time = message.created_at + timedelta(hours=9)

                # ======== トリミング画像をDiscordへ送信 ========
                buf = BytesIO()
                time_area_img.save(buf, format="PNG")
                buf.seek(0)
                await message.channel.send("🖼 **基準時間領域**", file=discord.File(buf, "time_area.png"))

                # ======== 中央30%OCRで駐機情報を抽出 ========
                cropped_main = crop_center_30(image)
                cropped_np = np.array(cropped_main)
                result = ocr.ocr(cropped_np, cls=True)
                extracted_text = [line[1][0] for res in result for line in res]

                # デバッグ用：OCRそのままの結果をDiscordに表示
                raw_debug_text = "\n".join(extracted_text)
                await message.channel.send(f"📝 **OCR生データ**\n```\n{raw_debug_text}\n```")

                all_text = " ".join(extracted_text)

                # ======== サーバー番号抽出 ========
                server_match = re.search(r'\[?S\d+\]?', all_text)
                server_num = server_match.group(0).replace("[", "").replace("]", "") if server_match else "UNKNOWN"

                # ======== 越域駐騎場番号 ========
                spot_nums = re.findall(r'越域駐騎場(\d+)', all_text)

                # ======== 免戦中時間（無ければ開戦済） ========
                times = re.findall(r'免戦中(\d{1,2}:\d{2})', all_text)

                combined = []
                for i, spot in enumerate(spot_nums):
                    if i < len(times):
                        raw_time = times[i]
                        mins, secs = map(int, raw_time.split(":"))
                        delta = timedelta(minutes=mins, seconds=secs)
                        end_time = (base_time + delta).strftime("%H:%M:%S")
                        combined.append(f"{server_num}-{spot}-{end_time}")
                    else:
                        combined.append(f"{server_num}-{spot}-開戦済")

                # ======== 最終メッセージ ========
                if combined:
                    reply = "🗓 **駐機スケジュール**\n" + "\n".join(f"- {c}" for c in combined)
                else:
                    reply = "❌ 必要な駐機情報が見つかりませんでした…"

                await processing_msg.edit(content=reply)

client.run(TOKEN)