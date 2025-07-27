import discord
import cv2
import numpy as np
from io import BytesIO
from paddleocr import PaddleOCR
from PIL import Image
from datetime import datetime, timedelta
import re
import os

# ✅ numpy 1.24+ で np.int が削除されたため互換パッチ
if not hasattr(np, "int"):
    np.int = int

# 環境変数からトークン取得
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("環境変数 DISCORD_BOT_TOKEN が設定されていません")

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# ✅ use_angle_cls は非推奨になったので use_textline_orientation に置き換え
ocr = PaddleOCR(lang='japan', use_textline_orientation=True)

def crop_center(image: Image.Image) -> Image.Image:
    """中央部分だけ残す"""
    w, h = image.size
    new_top = int(h * 0.35)
    new_bottom = int(h * 0.65)
    return image.crop((0, new_top, w, new_bottom))

def crop_time_area(image: Image.Image) -> Image.Image:
    """基準時間を読み取るエリア（上20% 右30%）"""
    w, h = image.size
    top = 0
    bottom = int(h * 0.2)
    left = int(w * 0.7)
    right = w
    return image.crop((left, top, right, bottom))

def parse_time_delta(text: str) -> timedelta | None:
    """免戦中テキストから timedelta を作る"""
    text = text.strip()
    # HH:MM:SS パターン
    if re.match(r"^\d{1,2}:\d{2}:\d{2}$", text):
        h, m, s = map(int, text.split(":"))
        return timedelta(hours=h, minutes=m, seconds=s)
    # MM:SS パターン
    elif re.match(r"^\d{1,2}:\d{2}$", text):
        m, s = map(int, text.split(":"))
        return timedelta(minutes=m, seconds=s)
    return None

@client.event
async def on_ready():
    print(f"✅ Botログイン成功: {client.user}")

@client.event
async def on_message(message):
    if message.author.bot:
        return

    if message.attachments:
        await message.channel.send("⏳ 解析中…")

        for attachment in message.attachments:
            img_data = await attachment.read()
            image = Image.open(BytesIO(img_data)).convert("RGB")

            # === 基準時間エリアをトリミングしてOCR ===
            time_crop = crop_time_area(image)
            time_buf = BytesIO()
            time_crop.save(time_buf, format="PNG")
            time_bytes = time_buf.getvalue()

            # トリミング画像を送信（デバッグ用）
            await message.channel.send(file=discord.File(BytesIO(time_bytes), filename="time_area.png"))

            # OCRで基準時間を読み取る
            base_ocr_result = ocr.ocr(np.array(time_crop), cls=True)
            base_time_text = None
            for line in base_ocr_result[0]:
                txt = line[1][0]
                if re.match(r"^\d{2}:\d{2}:\d{2}$", txt):
                    base_time_text = txt
                    break

            if base_time_text is None:
                await message.channel.send("⚠️ 基準時間が読み取れませんでした")
                return

            base_time = datetime.strptime(base_time_text, "%H:%M:%S")
            print(f"[DEBUG] 画像の基準時間: {base_time_text}")

            # === 中央部分をOCR ===
            cropped = crop_center(image)
            result = ocr.ocr(np.array(cropped), cls=True)

            # === デバッグ: OCR結果全部出力 ===
            all_texts = [line[1][0] for line in result[0]]
            print("[DEBUG] OCR抽出テキスト:", all_texts)

            # サーバー番号を取得
            server_match = re.search(r"\[?S(\d+)\]?", " ".join(all_texts))
            server = f"S{server_match.group(1)}" if server_match else "UNKNOWN"

            schedule_lines = []

            # 越域駐騎場と免戦時間の対応を探す
            for i, txt in enumerate(all_texts):
                m = re.search(r"越域駐騎場(\d+)", txt)
                if m:
                    number = m.group(1)
                    # 直後に免戦時間があるか探す
                    end_time = "開戦済"
                    if i + 1 < len(all_texts):
                        next_txt = all_texts[i + 1]
                        delta = parse_time_delta(next_txt)
                        if delta:
                            finish_time = (base_time + delta).strftime("%H:%M:%S")
                            end_time = finish_time

                    schedule_lines.append(f"{server}-{number}-{end_time}")

            if not schedule_lines:
                await message.channel.send("⚠️ 駐騎場情報が見つかりませんでした")
                return

            reply = "🗓 **駐機スケジュール**\n" + "\n".join(schedule_lines)
            await message.channel.send(reply)

client.run(TOKEN)