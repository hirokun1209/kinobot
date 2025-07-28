import os
import discord
import io
import cv2
import re
import asyncio
import numpy as np
from paddleocr import PaddleOCR
from datetime import datetime, timedelta, timezone
from PIL import Image

# =======================
#  タイムゾーン設定
# =======================
JST = timezone(timedelta(hours=9))  # 日本標準時

# =======================
#  BOT設定
# =======================
TOKEN = os.getenv("DISCORD_TOKEN")
NOTIFY_CHANNEL_ID = int(os.getenv("NOTIFY_CHANNEL_ID", "0"))

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
#  通知管理 (登録時刻も保存)
# =======================
pending_places = {}  # key: txt, value: (解除予定時刻, テキスト, サーバー番号, 登録時刻)

SKIP_NOTIFY_START = 2
SKIP_NOTIFY_END = 14

# =======================
#  ユーティリティ
# =======================
def now_jst():
    """常にJSTの現在時刻を取得"""
    return datetime.now(JST)

def cleanup_old_entries():
    """6時間以上経過した古いデータを削除"""
    now = now_jst()
    expired_keys = [k for k, v in pending_places.items() if (now - v[3]) > timedelta(hours=6)]
    for k in expired_keys:
        del pending_places[k]

def add_time(base_time_str: str, duration_str: str):
    try:
        # 基準時間もJSTとして扱う
        today = now_jst().date()
        base_time_only = datetime.strptime(base_time_str, "%H:%M:%S").time()
        base_time = datetime.combine(today, base_time_only, tzinfo=JST)
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
    h, w, _ = img.shape
    return img[0:int(h * 0.2), int(w * 0.7):w]

def crop_center_area(img: np.ndarray) -> np.ndarray:
    h, w, _ = img.shape
    return img[int(h * 0.35):int(h * 0.65), 0:w]

def extract_text_from_image(img: np.ndarray):
    result = ocr.ocr(img, cls=True)
    return [line[1][0] for line in result[0]] if result and result[0] else []

def extract_server_number(center_texts):
    for t in center_texts:
        match = re.search(r"[sS]\d{3,4}", t)
        if match:
            return match.group(0).lower().replace("s", "")
    return None

def parse_multiple_places(center_texts, top_time_texts):
    results, no_time_places, debug_lines = [], [], []

    top_time = next((t for t in top_time_texts if re.match(r"\d{2}:\d{2}:\d{2}", t)), None)
    if not top_time:
        return [], ["⚠️ 右上の時間が取得できませんでした"], []

    server_num = extract_server_number(center_texts)
    if not server_num:
        return [], ["⚠️ サーバー番号が取得できませんでした"], []

    mode = "警備" if server_num == "1281" else "奪取"
    debug_lines.append(f"📌 サーバー番号: {server_num} ({mode})")
    debug_lines.append(f"📌 右上基準時間: {top_time}\n")

    current_place = None
    seen_places = set()

    for t in center_texts:
        place_match = re.search(r"越域駐騎場(\d+)", t)
        if place_match:
            current_place = place_match.group(1)
            seen_places.add(current_place)

        duration_match = re.search(r"免戦中(\d{1,2}:\d{2}(?::\d{2})?)", t)
        if duration_match and current_place:
            duration = duration_match.group(1)
            unlock_dt, unlock_time = add_time(top_time, duration)
            if unlock_dt:
                results.append((unlock_dt, f"{mode} {server_num}-{current_place}-{unlock_time}", server_num))
            else:
                no_time_places.append(f"{mode} {server_num}-{current_place}-開戦済")
            current_place = None

    for p in seen_places:
        if not any(f"-{p}-" in txt for _, txt, _ in results) and not any(f"-{p}-" in txt for txt in no_time_places):
            no_time_places.append(f"{mode} {server_num}-{p}-開戦済")

    return results, no_time_places, debug_lines

def should_skip_notification(dt: datetime):
    return SKIP_NOTIFY_START <= dt.hour < SKIP_NOTIFY_END

# =======================
#  デバッグ付き通知スケジューラー (JST)
# =======================
async def schedule_notification(unlock_dt: datetime, text: str, notify_channel: discord.TextChannel, debug=False):
    now = now_jst()
    log_msg = f"[DEBUG] schedule_notification開始(JST): {text}, unlock_dt={unlock_dt.strftime('%H:%M:%S')}, now={now.strftime('%H:%M:%S')}"
    print(log_msg)
    if notify_channel:
        await notify_channel.send(log_msg)

    if unlock_dt <= now:
        skip_msg = "[DEBUG] 解除時刻が現在より過去なのでスキップ"
        print(skip_msg)
        if notify_channel:
            await notify_channel.send(skip_msg)
        return

    if text.startswith("奪取") and (debug or not should_skip_notification(unlock_dt)):
        notify_time_2min = unlock_dt - timedelta(minutes=2)
        notify_time_15sec = unlock_dt - timedelta(seconds=15)

        # 2分前通知
        if notify_time_2min > now:
            wait_sec = (notify_time_2min - now).total_seconds()
            dbg = f"[DEBUG] 2分前通知まで {wait_sec:.1f}秒待機予定"
            print(dbg)
            if notify_channel:
                await notify_channel.send(dbg)

            await asyncio.sleep(wait_sec)

            dbg_send = f"[DEBUG] 2分前通知送信: {text}"
            print(dbg_send)
            if notify_channel:
                await notify_channel.send(dbg_send)
            await notify_channel.send(f"⏰ {text} **2分前です！！**")
        else:
            msg = "[DEBUG] 2分前通知はすでに過ぎてる"
            print(msg)
            if notify_channel:
                await notify_channel.send(msg)

        # 15秒前通知
        now2 = now_jst()
        if notify_time_15sec > now2:
            wait_sec = (notify_time_15sec - now2).total_seconds()
            dbg2 = f"[DEBUG] 15秒前通知まで {wait_sec:.1f}秒待機予定"
            print(dbg2)
            if notify_channel:
                await notify_channel.send(dbg2)

            await asyncio.sleep(wait_sec)

            dbg_send2 = f"[DEBUG] 15秒前通知送信: {text}"
            print(dbg_send2)
            if notify_channel:
                await notify_channel.send(dbg_send2)
            await notify_channel.send(f"⏰ {text} **15秒前です！！**")
        else:
            msg2 = "[DEBUG] 15秒前通知はすでに過ぎてる"
            print(msg2)
            if notify_channel:
                await notify_channel.send(msg2)
    else:
        skip_cond = "[DEBUG] 通知スキップ条件に該当 (奪取以外 or 時間帯スキップ)"
        print(skip_cond)
        if notify_channel:
            await notify_channel.send(skip_cond)

# =======================
#  イベント
# =======================
@client.event
async def on_ready():
    print(f"✅ ログイン成功！Bot名: {client.user}")

@client.event
async def on_message(message):
    if message.author.bot:
        return

    cleanup_old_entries()
    notify_channel = client.get_channel(NOTIFY_CHANNEL_ID) if NOTIFY_CHANNEL_ID else None

    # ==== デバッグ用 "!1234-7-12:34:56" ====
    if message.content.startswith("!"):
        m = re.match(r"!([0-9]{3,4})-([0-9]+)-([0-9]{2}:[0-9]{2}:[0-9]{2})", message.content)
        if m:
            server_num, place_num, unlock_time = m.groups()
            mode = "警備" if server_num == "1281" else "奪取"
            txt = f"{mode} {server_num}-{place_num}-{unlock_time}"

            # 今日の日付のJSTに変換
            today = now_jst().date()
            unlock_dt = datetime.combine(today, datetime.strptime(unlock_time, "%H:%M:%S").time(), tzinfo=JST)

            # ✅ pending_places に登録
            pending_places[txt] = (unlock_dt, txt, server_num, now_jst())

            dbg = f"[DEBUG] デバッグ登録(JST): {txt} (now={now_jst().strftime('%H:%M:%S')})"
            print(dbg)
            await message.channel.send(f"✅ {dbg}")

            if notify_channel:
                await notify_channel.send(f"✅ {dbg}")
                asyncio.create_task(schedule_notification(unlock_dt, txt, notify_channel, debug=True))
            return

    # ==== 画像が送られた場合 ====
    if message.attachments:
        processing_msg = await message.channel.send("🔄 画像解析中…")

        for attachment in message.attachments:
            img_bytes = await attachment.read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            top_img = crop_top_right(img_np)
            center_img = crop_center_area(img_np)
            top_texts = extract_text_from_image(top_img)
            center_texts = extract_text_from_image(center_img)

            parsed_results, no_time_places, debug_lines = parse_multiple_places(center_texts, top_texts)

            for dt, txt, server in parsed_results:
                if txt not in pending_places:
                    pending_places[txt] = (dt, txt, server, now_jst())
                    if txt.startswith("奪取") and notify_channel:
                        asyncio.create_task(schedule_notification(dt, txt, notify_channel))

            for txt in no_time_places:
                if txt not in pending_places:
                    pending_places[txt] = (datetime.min.replace(tzinfo=JST), txt, "", now_jst())

        cleanup_old_entries()

        opened = [txt for dt, txt, _, _ in pending_places.values() if dt == datetime.min.replace(tzinfo=JST)]
        takes = [(dt, txt) for dt, txt, _, _ in pending_places.values() if dt != datetime.min.replace(tzinfo=JST) and txt.startswith("奪取")]
        guards = [(dt, txt) for dt, txt, _, _ in pending_places.values() if dt != datetime.min.replace(tzinfo=JST) and txt.startswith("警備")]

        takes.sort(key=lambda x: x[0])
        guards.sort(key=lambda x: x[0])

        msg_lines = []
        if opened:
            msg_lines.extend(opened)
        msg_lines.extend(txt for _, txt in takes)
        msg_lines.extend(txt for _, txt in guards)

        reply_msg = "\n".join(msg_lines) if msg_lines else "⚠️ 情報が見つかりませんでした"
        await processing_msg.edit(content=reply_msg)

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