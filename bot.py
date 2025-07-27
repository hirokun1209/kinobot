import os
import discord
import io
import cv2
import re
import asyncio
import numpy as np
from paddleocr import PaddleOCR
from datetime import datetime, timedelta
from PIL import Image

# =======================
#  BOT設定
# =======================
TOKEN = os.getenv("DISCORD_TOKEN")
NOTIFY_CHANNEL_ID = int(os.getenv("NOTIFY_CHANNEL_ID", "0"))  # 通知専用チャンネル

if not TOKEN:
    raise ValueError("❌ DISCORD_TOKEN が設定されていません！")

# =======================
#  Discord Client定義
# =======================
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# =======================
#  OCR初期化
# =======================
ocr = PaddleOCR(use_angle_cls=True, lang='japan')

# =======================
#  通知用管理
# =======================
pending_places = {}  # key=place_id, value=(datetime, "奪取 1245-7-20:06:18", server_num)

SKIP_NOTIFY_START = 2   # 通知無効時間 2時～
SKIP_NOTIFY_END = 14    # ～14時まで

# =======================
#  ユーティリティ関数
# =======================
def add_time(base_time_str: str, duration_str: str):
    """右上の時間 + 免戦時間 → (datetime, 計算後の時刻文字列)"""
    try:
        base_time = datetime.strptime(base_time_str, "%H:%M:%S")
    except ValueError:
        return None, None

    parts = duration_str.strip().split(":")
    if len(parts) == 3:
        h, m, s = map(int, parts)
    elif len(parts) == 2:
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
    return img[0:int(h * 0.2), int(w * 0.7):w]

def crop_center_area(img: np.ndarray) -> np.ndarray:
    """上下35%をカットして中央エリアをトリミング"""
    h, w, _ = img.shape
    return img[int(h * 0.35):int(h * 0.65), 0:w]

def extract_text_from_image(img: np.ndarray):
    """OCRで文字認識"""
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
    """複数駐騎場番号と免戦時間を解析"""
    results = []
    no_time_places = []
    debug_lines = []

    # 右上の時間
    top_time = next((t for t in top_time_texts if re.match(r"\d{2}:\d{2}:\d{2}", t)), None)
    if not top_time:
        return [], ["⚠️ 右上の時間が取得できませんでした"], []

    # サーバー番号
    server_num = extract_server_number(center_texts)
    if not server_num:
        return [], ["⚠️ サーバー番号が取得できませんでした"], []

    # モード判定
    mode = "警備" if server_num == "1281" else "奪取"
    debug_lines.append(f"📌 サーバー番号: {server_num} ({mode})")
    debug_lines.append(f"📌 右上基準時間: {top_time}\n")

    current_place = None
    seen_places = set()

    for t in center_texts:
        # 駐騎場番号
        place_match = re.search(r"越域駐騎場(\d+)", t)
        if place_match:
            current_place = place_match.group(1)
            seen_places.add(current_place)

        # 免戦中の時間
        duration_match = re.search(r"免戦中(\d{1,2}:\d{2}(?::\d{2})?)", t)
        if duration_match and current_place:
            duration = duration_match.group(1)
            debug_lines.append(f"✅ 越域駐騎場{current_place} → 免戦中{duration}")
            unlock_dt, unlock_time = add_time(top_time, duration)
            if unlock_dt:
                debug_lines.append(f"   → {top_time} + {duration} = {unlock_time}\n")
                results.append((unlock_dt, f"{mode} {server_num}-{current_place}-{unlock_time}", server_num))
            else:
                debug_lines.append(f"   → 計算できず → 開戦済\n")
                no_time_places.append(f"{mode} {server_num}-{current_place}-開戦済")
            current_place = None

    # 免戦時間なし → 開戦済扱い
    for p in seen_places:
        if not any(f"-{p}-" in txt for _, txt, _ in results) and not any(f"-{p}-" in txt for txt in no_time_places):
            no_time_places.append(f"{mode} {server_num}-{p}-開戦済")

    return results, no_time_places, debug_lines

def should_skip_notification(dt: datetime):
    """02:00～14:00は通知しない"""
    return SKIP_NOTIFY_START <= dt.hour < SKIP_NOTIFY_END

async def schedule_notification(unlock_dt: datetime, text: str, notify_channel: discord.TextChannel, debug=False):
    """解除予定時刻の2分前・15秒前に通知"""
    now = datetime.now()
    if unlock_dt <= now:
        return

    # デバッグモードの場合は時間帯制限を無視
    if text.startswith("奪取") and (debug or not should_skip_notification(unlock_dt)):
        notify_time_2min = unlock_dt - timedelta(minutes=2)
        notify_time_15sec = unlock_dt - timedelta(seconds=15)

        if notify_time_2min > now:
            await asyncio.sleep((notify_time_2min - now).total_seconds())
            await notify_channel.send(f"⏰ {text} **2分前です！！**")

        if notify_time_15sec > datetime.now():
            await asyncio.sleep((notify_time_15sec - datetime.now()).total_seconds())
            await notify_channel.send(f"⏰ {text} **15秒前です！！**")

async def send_schedule_summary(channel: discord.TextChannel):
    """pending_places から奪取・警備・開戦済を分類して通知"""
    if not pending_places:
        return

    opened, takes, guards = [], [], []

    for dt, txt, server in pending_places.values():
        if dt == datetime.min:
            opened.append(txt)
        elif txt.startswith("奪取"):
            takes.append((dt, txt))
        else:
            guards.append((dt, txt))

    takes.sort(key=lambda x: x[0])
    guards.sort(key=lambda x: x[0])

    lines = []
    if opened:
        lines.append("⚠️ 開戦済")
        lines.extend(opened)
        lines.append("")
    if takes or guards:
        lines.append("⏳ スケジュール")
        if takes:
            lines.append("【奪取】")
            lines.extend(txt for _, txt in takes)
        if guards:
            lines.append("【警備】")
            lines.extend(txt for _, txt in guards)

    final_msg = "📢奪取&警備スケジュールのお知らせ📢\n2分前 & 15秒前に通知します\n\n" + "\n".join(lines)
    await channel.send(final_msg)

# =======================
#  イベント処理
# =======================
@client.event
async def on_ready():
    print(f"✅ ログイン成功！Bot名: {client.user}")

@client.event
async def on_message(message):
    if message.author.bot:
        return

    notify_channel = client.get_channel(NOTIFY_CHANNEL_ID) if NOTIFY_CHANNEL_ID else None

    # ==== デバッグ用 "!1234-7-12:34:56" ====
    if message.content.startswith("!"):
        m = re.match(r"!([0-9]{3,4})-([0-9]+)-([0-9]{2}:[0-9]{2}:[0-9]{2})", message.content)
        if m:
            server_num, place_num, unlock_time = m.groups()
            mode = "警備" if server_num == "1281" else "奪取"
            txt = f"{mode} {server_num}-{place_num}-{unlock_time}"
            dt = datetime.strptime(unlock_time, "%H:%M:%S")

            pending_places[txt] = (dt, txt, server_num)
            await message.channel.send(f"✅ デバッグ登録: {txt}")

            if notify_channel:
                asyncio.create_task(schedule_notification(dt, txt, notify_channel, debug=True))
            return

    # ==== 画像が送られた場合 ====
    if message.attachments:
        processing_msg = await message.channel.send("🔄 画像解析中…")

        for attachment in message.attachments:
            img_bytes = await attachment.read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # トリミング & OCR
            top_img = crop_top_right(img_np)
            center_img = crop_center_area(img_np)
            top_texts = extract_text_from_image(top_img)
            center_texts = extract_text_from_image(center_img)

            parsed_results, no_time_places, debug_lines = parse_multiple_places(center_texts, top_texts)

            # OCR結果の登録
            for dt, txt, server in parsed_results:
                key = txt
                if txt.startswith("奪取"):
                    pending_places[key] = (dt, txt, server)
                    if notify_channel:
                        asyncio.create_task(schedule_notification(dt, txt, notify_channel))
                else:
                    pending_places[key] = (dt, txt, server)

            for txt in no_time_places:
                pending_places[txt] = (datetime.min, txt, "")

        # ==== メッセージ返信 ====
        opened = [txt for dt, txt, _ in pending_places.values() if dt == datetime.min]
        takes = [(dt, txt) for dt, txt, _ in pending_places.values() if dt != datetime.min and txt.startswith("奪取")]
        guards = [(dt, txt) for dt, txt, _ in pending_places.values() if dt != datetime.min and txt.startswith("警備")]

        takes.sort(key=lambda x: x[0])
        guards.sort(key=lambda x: x[0])

        msg_lines = []
        if opened:
            msg_lines.append("⚠️ 開戦済")
            msg_lines.extend(opened)
            msg_lines.append("")
        if takes or guards:
            msg_lines.append("⏳ スケジュール")
            if takes:
                msg_lines.append("【奪取】")
                msg_lines.extend(txt for _, txt in takes)
            if guards:
                msg_lines.append("【警備】")
                msg_lines.extend(txt for _, txt in guards)

        reply_msg = "\n".join(msg_lines) if msg_lines else "⚠️ 情報が見つかりませんでした"
        await processing_msg.edit(content=reply_msg)

        # ==== 1〜12が揃ったら通知チャンネルにまとめ送信 ====
        places_found = {txt.split("-")[1] for _, txt, _ in pending_places.values() if "-" in txt}
        if len(places_found) >= 12 and notify_channel:
            await send_schedule_summary(notify_channel)

# =======================
#  BOT起動
# =======================
if __name__ == "__main__":
    try:
        client.run(TOKEN)
    except discord.errors.LoginFailure:
        print("❌ Discord トークンが無効です！")
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")