import discord
import cv2
import numpy as np
from io import BytesIO
from paddleocr import PaddleOCR
from PIL import Image
from datetime import datetime, timedelta
import re
import os
from dotenv import load_dotenv

# === 設定 ===
load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")  # .env に DISCORD_BOT_TOKEN を入れる
DEBUG_MODE = True  # True にするとOCR結果などデバッグ送信

# === Discord初期化 ===
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# === OCR初期化 ===
ocr = PaddleOCR(use_angle_cls=True, lang='japan')

# === トリミング関数 ===
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

# === 免戦時間を timedelta に変換 ===
def parse_time_delta(text: str) -> timedelta | None:
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

# === 基準時間のOCR結果から時間をパース ===
def parse_base_time(all_texts: list[str]) -> datetime | None:
    for txt in all_texts:
        txt_clean = txt.replace(" ", "")
        # H:M:S or HH:MM:SS
        m = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$", txt_clean)
        if m:
            h = int(m.group(1))
            mnt = int(m.group(2))
            sec = int(m.group(3)) if m.group(3) else 0
            return datetime.strptime(f"{h:02d}:{mnt:02d}:{sec:02d}", "%H:%M:%S")
    return None

# === サーバー番号抽出 ===
def extract_server_number(texts: list[str]) -> str:
    joined = " ".join(texts)
    m = re.search(r"[Ss][-]?\s?(\d{3,5})", joined)
    return f"S{m.group(1)}" if m else "UNKNOWN"

# === Discordイベント ===
@client.event
async def on_ready():
    print(f"✅ Logged in as {client.user}")

@client.event
async def on_message(message):
    if message.author.bot:
        return

    if not message.attachments:
        return

    await message.channel.send("⏳ 画像解析中…")

    for attachment in message.attachments:
        img_data = await attachment.read()
        image = Image.open(BytesIO(img_data)).convert("RGB")

        # === 基準時間エリアOCR ===
        time_crop = crop_time_area(image)
        base_ocr_result = ocr.ocr(np.array(time_crop), cls=True)
        time_texts = [line[1][0] for line in base_ocr_result[0]]
        base_time = parse_base_time(time_texts)

        if DEBUG_MODE:
            await message.channel.send(f"📜 **基準時間OCR結果:** {time_texts}")

        if base_time is None:
            await message.channel.send("⚠️ 基準時間が読み取れませんでした（右上が認識できなかった）")
            return

        # === 中央部分OCR ===
        cropped = crop_center(image)
        result = ocr.ocr(np.array(cropped), cls=True)
        all_texts = [line[1][0] for line in result[0]]

        if DEBUG_MODE:
            await message.channel.send(f"📜 **中央OCR結果:** {all_texts}")

        # === サーバー番号取得 ===
        server = extract_server_number(all_texts)

        # === スケジュール作成 ===
        schedule_lines = []
        for i, txt in enumerate(all_texts):
            m = re.search(r"越域駐騎場\s?(\d+)", txt)
            if m:
                number = m.group(1)
                # 直後の免戦時間を探す
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

# === 実行 ===
client.run(TOKEN)