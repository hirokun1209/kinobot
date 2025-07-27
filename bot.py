import discord
import io
import cv2
import numpy as np
from paddleocr import PaddleOCR
from datetime import datetime, timedelta
from PIL import Image

# Discord Botトークン
TOKEN = "YOUR_DISCORD_BOT_TOKEN"

# OCR初期化（日本語）
ocr = PaddleOCR(use_angle_cls=True, lang='japan')

# Discordクライアント
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

def add_time(base_time_str: str, duration_str: str) -> str:
    """右上の時間 + 免戦時間を計算して解除時刻を返す"""
    base_time = datetime.strptime(base_time_str, "%H:%M:%S")

    parts = duration_str.strip().split(":")
    if len(parts) == 3:  # HH:MM:SS
        h, m, s = map(int, parts)
    elif len(parts) == 2:  # MM:SS → 0時間扱い
        h = 0
        m, s = map(int, parts)
    else:
        return base_time_str  # 想定外 → 右上時間そのまま返す

    delta = timedelta(hours=h, minutes=m, seconds=s)
    new_time = (base_time + delta).time()
    return new_time.strftime("%H:%M:%S")

def crop_top_right(img: np.ndarray) -> np.ndarray:
    """右上20%の領域をトリミング"""
    h, w, _ = img.shape
    cropped = img[0:int(h*0.2), int(w*0.8):w]  # 上20% & 右20%
    return cropped

def crop_center_area(img: np.ndarray) -> np.ndarray:
    """上下35%をカットして中央エリアをトリミング"""
    h, w, _ = img.shape
    cropped = img[int(h*0.35):int(h*0.65), 0:w]
    return cropped

def extract_text_from_image(img: np.ndarray):
    """PaddleOCRで文字認識"""
    result = ocr.ocr(img, cls=True)
    text_results = []
    for line in result[0]:
        text_results.append(line[1][0])
    return text_results

def parse_info(center_texts, top_time_texts):
    """OCR結果から必要な情報を抽出 & 整形"""
    # 右上の時間（必ず HH:MM:SS）
    top_time = None
    for t in top_time_texts:
        if ":" in t and len(t.split(":")) == 3:
            top_time = t
            break

    # サーバー番号 / 駐騎場番号 / 免戦時間を抽出
    server = None
    place_num = None
    duration = None

    for t in center_texts:
        if t.startswith("s") and t[1:].isdigit():
            server = t
        elif t.isdigit():
            place_num = t
        elif ":" in t:
            duration = t

    if not (server and place_num and duration and top_time):
        return None  # 必要な情報が揃わない場合

    # モード判定（s1281は警備、それ以外は奪取）
    mode = "警備" if server == "s1281" else "奪取"

    # 解除時刻計算
    new_time = add_time(top_time, duration)

    # フォーマット → ` 警備 s1281-3-20:14:54`
    return f" {mode} {server}-{place_num}-{new_time}"

@client.event
async def on_ready():
    print(f"✅ Logged in as {client.user}")

@client.event
async def on_message(message):
    if message.author.bot:
        return

    if message.attachments:
        results = []
        for attachment in message.attachments:
            img_bytes = await attachment.read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # 右上と中央をトリミング
            top_img = crop_top_right(img_np)
            center_img = crop_center_area(img_np)

            # OCR結果
            top_texts = extract_text_from_image(top_img)
            center_texts = extract_text_from_image(center_img)

            # パース
            info = parse_info(center_texts, top_texts)
            if info:
                results.append(info)
            else:
                results.append("⚠️ 必要な情報が読み取れませんでした")

        await message.channel.send("\n".join(results))

client.run(TOKEN)