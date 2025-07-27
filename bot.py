import os
import discord
import io
import cv2
import re
import numpy as np
from paddleocr import PaddleOCR
from datetime import datetime, timedelta
from PIL import Image

# ✅ Discord Botトークンを環境変数から取得
TOKEN = os.getenv("DISCORD_TOKEN")
if not TOKEN:
    raise ValueError("❌ DISCORD_TOKEN が設定されていません！Koyeb の Environment Variables を確認してください。")

# ✅ OCR初期化（日本語）
ocr = PaddleOCR(use_angle_cls=True, lang='japan')

# ✅ Discordクライアント
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

def add_time(base_time_str: str, duration_str: str):
    """右上の時間 + 免戦時間 → (datetime, 計算後の時刻文字列)"""
    try:
        base_time = datetime.strptime(base_time_str, "%H:%M:%S")
    except ValueError:
        return None, None

    parts = duration_str.strip().split(":")
    if len(parts) == 3:  # HH:MM:SS
        h, m, s = map(int, parts)
    elif len(parts) == 2:  # MM:SS → 0時間扱い
        h = 0
        m, s = map(int, parts)
    else:
        return None, None

    delta = timedelta(hours=h, minutes=m, seconds=s)
    unlock_dt = base_time + delta
    return unlock_dt, unlock_dt.strftime("%H:%M:%S")

def crop_top_right(img: np.ndarray) -> np.ndarray:
    """右上30%の領域をトリミング"""
    h, w, _ = img.shape
    return img[0:int(h * 0.2), int(w * 0.7):w]  # 上20% & 右30%

def crop_center_area(img: np.ndarray) -> np.ndarray:
    """上下35%をカットして中央エリアをトリミング"""
    h, w, _ = img.shape
    return img[int(h * 0.35):int(h * 0.65), 0:w]

def extract_text_from_image(img: np.ndarray):
    """PaddleOCRで文字認識"""
    result = ocr.ocr(img, cls=True)
    return [line[1][0] for line in result[0]] if result and result[0] else []

def extract_server_number(center_texts):
    """OCR結果からサーバー番号(s1234形式)を抽出"""
    for t in center_texts:
        match = re.search(r"[sS]\d{3,4}", t)
        if match:
            return match.group(0).lower().replace("s", "")
    return None

def parse_multiple_places(center_texts, top_time_texts):
    """
    OCR結果から複数駐騎場の免戦時間を解析
    戻り値:
      results: [(datetime, "奪取 1245-7-20:06:18"), ...]
      no_time_places: ["奪取 1245-8-開戦済", ...]
      debug_lines: 計算過程ログ
    """
    results = []
    no_time_places = []
    debug_lines = []

    # ✅ 右上の時間
    top_time = next((t for t in top_time_texts if re.match(r"\d{2}:\d{2}:\d{2}", t)), None)
    if not top_time:
        return [], ["⚠️ 右上の時間が取得できませんでした"], []

    # ✅ サーバー番号
    server_num = extract_server_number(center_texts)
    if not server_num:
        return [], ["⚠️ サーバー番号が取得できませんでした"], []

    # ✅ モード判定
    mode = "警備" if server_num == "1281" else "奪取"

    debug_lines.append(f"📌 サーバー番号: {server_num} ({mode})")
    debug_lines.append(f"📌 右上基準時間: {top_time}\n")

    current_place = None

    for t in center_texts:
        # 駐騎場番号を取得
        place_match = re.search(r"越域駐騎場(\d+)", t)
        if place_match:
            current_place = place_match.group(1)

        # 免戦中の時間
        duration_match = re.search(r"免戦中(\d{1,2}:\d{2}(?::\d{2})?)", t)
        if duration_match and current_place:
            duration = duration_match.group(1)
            debug_lines.append(f"✅ 越域駐騎場{current_place} → 免戦中{duration}")

            unlock_dt, unlock_time = add_time(top_time, duration)
            if unlock_dt:
                debug_lines.append(f"   → {top_time} + {duration} = {unlock_time}\n")
                results.append((unlock_dt, f"{mode} {server_num}-{current_place}-{unlock_time}"))
            else:
                debug_lines.append(f"   → 計算できず → 開戦済\n")
                no_time_places.append(f"{mode} {server_num}-{current_place}-開戦済")

            current_place = None

    return results, no_time_places, debug_lines

@client.event
async def on_ready():
    print(f"✅ ログイン成功！Bot名: {client.user}")

@client.event
async def on_message(message):
    if message.author.bot:
        return

    if message.attachments:
        processing_msg = await message.channel.send("🔄 画像解析中…")

        all_results = []     # 計算済み結果
        all_no_time = []     # 開戦済など
        all_debug_lines = [] # 計算ログ

        for attachment in message.attachments:
            img_bytes = await attachment.read()

            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # トリミング & OCR
            top_img = crop_top_right(img_np)
            center_img = crop_center_area(img_np)
            top_texts = extract_text_from_image(top_img)
            center_texts = extract_text_from_image(center_img)

            # 複数解析
            parsed_results, no_time_places, debug_lines = parse_multiple_places(center_texts, top_texts)
            all_results.extend(parsed_results)
            all_no_time.extend(no_time_places)
            all_debug_lines.extend(debug_lines)

        # ✅ 時間でソート
        all_results.sort(key=lambda x: x[0])
        sorted_texts = [text for _, text in all_results]

        # ✅ 最終結果
        debug_part = "\n".join(all_debug_lines)
        result_part = "\n".join(sorted_texts + all_no_time) if (sorted_texts or all_no_time) else "⚠️ 必要な情報が読み取れませんでした"

        final_msg = f"{debug_part}\n\n📌 時間順:\n{result_part}"

        await processing_msg.edit(content=final_msg)

# ✅ Bot起動
if __name__ == "__main__":
    try:
        client.run(TOKEN)
    except discord.errors.LoginFailure:
        print("❌ Discord トークンが無効です！Koyeb の Environment Variables を確認してください。")
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")