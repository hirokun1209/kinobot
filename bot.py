# OCR BOT（スケジュール通知付き + HTTPサーバーでUptimeRobot対応）
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
from fastapi import FastAPI
import uvicorn
from threading import Thread

# =======================
# タイムゾーン設定
# =======================
JST = timezone(timedelta(hours=9))

# =======================
# BOT設定
# =======================
TOKEN = os.getenv("DISCORD_TOKEN")
NOTIFY_CHANNEL_ID = int(os.getenv("NOTIFY_CHANNEL_ID", "0"))
READABLE_CHANNEL_IDS = [int(x) for x in os.getenv("ALLOWED_CHANNEL_IDS", "").split(",") if x.strip().isdigit()]
if not TOKEN:
    raise ValueError("❌ DISCORD_TOKEN が設定されていません！")

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# =======================
# FastAPI HTTP サーバー（スリープ防止）
# =======================
app = FastAPI()

from fastapi.responses import JSONResponse

@app.get("/")
@app.get("/ping")
@app.get("/ping/")
def root():
    return JSONResponse(content={"status": "ok"})

def run_server():
    import time
    time.sleep(3)  # サービス安定のために3秒遅延
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

# =======================
# OCR初期化
# =======================
ocr = PaddleOCR(use_angle_cls=True, lang='japan')

# =======================
# 管理構造
# =======================
pending_places = {}
summary_blocks = []
active_tasks = set()
sent_notifications = set()
SKIP_NOTIFY_START = 2
SKIP_NOTIFY_END = 14

# =======================
# 過去予定の自動削除
# =======================
EXPIRE_GRACE = timedelta(minutes=2)

async def remove_expired_entries():
    now = now_jst()
    # pending_placesの削除
    for k, (dt, *_rest) in list(pending_places.items()):
        if dt + EXPIRE_GRACE < now:
            del pending_places[k]
    # summary_blocksの削除と通知メッセージ削除
    for block in list(summary_blocks):
        block["events"] = [ev for ev in block["events"] if ev[0] + EXPIRE_GRACE >= now]
        if block["msg"] and block["max"] + EXPIRE_GRACE < now:
            try:
                await block["msg"].delete()
            except:
                pass
            block["msg"] = None
        if not block["events"]:
            summary_blocks.remove(block)
    # タスクの削除
    for task in list(active_tasks):
        if task.done(): continue
        try:
            unlock_dt = task.get_coro().cr_frame.f_locals.get("unlock_dt")
            if isinstance(unlock_dt, datetime) and unlock_dt + EXPIRE_GRACE < now:
                task.cancel()
        except:
            pass

# =======================
# ユーティリティ
# =======================
# =======================
# ユーティリティ関数群
# =======================

# 現在時刻（JST）
def now_jst():
    return datetime.now(JST)

# トップ右の時間部分の切り取り
def crop_top_right(img):
    h, w = img.shape[:2]
    return img[0:int(h*0.2), int(w*0.7):]

# 中央エリアの切り取り（越域駐騎場など）
def crop_center_area(img):
    h, w = img.shape[:2]
    return img[int(h*0.35):int(h*0.65), :]

# OCRでテキスト抽出
def extract_text_from_image(img):
    result = ocr.ocr(img, cls=True)
    return [line[1][0] for line in result[0]] if result and result[0] else []

# s1234 を抽出（サーバー番号）
def extract_server_number(center_texts):
    for t in center_texts:
        m = re.search(r"[sS](\d{3,4})", t)
        if m:
            return m.group(1)
    return None

# 時間の加算（免戦時間に加算）
def add_time(base_time_str, duration_str):
    today = now_jst().date()
    try:
        base_time = datetime.strptime(base_time_str, "%H:%M:%S").time()
    except:
        return None, None
    base_dt = datetime.combine(today, base_time, tzinfo=JST)

    # 00:00:00〜02:00:01 は翌日として扱う
    if base_time < datetime.strptime("02:00:01", "%H:%M:%S").time():
        base_dt += timedelta(days=1)

    parts = duration_str.split(":")
    if len(parts) == 3:
        h, m, s = map(int, parts)
    elif len(parts) == 2:
        h, m, s = 0, *map(int, parts)
    else:
        return None, None
    dt = base_dt + timedelta(hours=h, minutes=m, seconds=s)
    return dt, dt.strftime("%H:%M:%S")
# =======================
# 通知処理（抜粋）
# =======================
async def schedule_notification(unlock_dt, text, channel):
    if unlock_dt <= now_jst():
        return
    if not (8 <= unlock_dt.hour or unlock_dt.hour < 2):
        return
    if text.startswith("奪取"):
        if not is_within_5_minutes_of_another(unlock_dt):
            t = unlock_dt - timedelta(minutes=2)
            if t > now_jst() and (text, "2min") not in sent_notifications:
                sent_notifications.add((text, "2min"))
                await asyncio.sleep((t - now_jst()).total_seconds())
                msg = await channel.send(f"⏰ {text} **2分前です！！**")
                await asyncio.sleep(120)
                await msg.delete()
        t15 = unlock_dt - timedelta(seconds=15)
        if t15 > now_jst() and (text, "15s") not in sent_notifications:
            sent_notifications.add((text, "15s"))
            await asyncio.sleep((t15 - now_jst()).total_seconds())
            msg = await channel.send(f"⏰ {text} **15秒前です！！**")
            await asyncio.sleep(120)
            await msg.delete()

# =======================
# 自動リセット（通知なし）
# =======================
async def daily_reset_task():
    await client.wait_until_ready()
    while not client.is_closed():
        now = now_jst()
        next_reset = datetime.combine(now.date(), datetime.strptime("02:00:00", "%H:%M:%S").time(), tzinfo=JST)
        if now >= next_reset:
            next_reset += timedelta(days=1)
        await asyncio.sleep((next_reset - now).total_seconds())
        pending_places.clear()
        summary_blocks.clear()
        sent_notifications.clear()
        for task in list(active_tasks):
            task.cancel()
        active_tasks.clear()

# =======================
# 定期クリーンアップ
# =======================
async def periodic_cleanup_task():
    await client.wait_until_ready()
    while not client.is_closed():
        await remove_expired_entries()
        await asyncio.sleep(60)

        
# =======================
# コマンドベースのリセット
# =======================
async def reset_all(message):
    pending_places.clear()
    summary_blocks.clear()
    sent_notifications.clear()
    for task in list(active_tasks):
        task.cancel()
    active_tasks.clear()
    await message.channel.send("✅ 全ての予定と通知をリセットしました")

# =======================
# Discordイベント
# =======================
@client.event
async def on_ready():
    print("✅ ログイン成功！")
    print(f"📌 通知チャンネル: {NOTIFY_CHANNEL_ID}")
    print(f"📌 読み取り許可チャンネル: {READABLE_CHANNEL_IDS}")
    asyncio.create_task(daily_reset_task())  # ✅ 自動リセットスケジューラー起動
    asyncio.create_task(periodic_cleanup_task())  # ✅ 過去予定の削除スケジューラー起動
        
@client.event
async def on_message(message):
    if message.author.bot or message.channel.id not in READABLE_CHANNEL_IDS:
        return

    channel = client.get_channel(NOTIFY_CHANNEL_ID)

    if message.content.strip() == "!reset":
        await reset_all(message)
        return

    if message.content.strip() == "!debug":
        if pending_places:
            lines = ["✅ 現在の登録された予定:"]
            lines += [f"・{v[1]}" for v in sorted(pending_places.values(), key=lambda x: x[0])]
            await message.channel.send("\n".join(lines))
        else:
            await message.channel.send("⚠️ 登録された予定はありません")
        return

    manual = re.findall(r"\b(\d{3,4})-(\d+)-(\d{2}:\d{2}:\d{2})\b", message.content)
    if manual:
        for server, place, t in manual:
            if len(server) == 3:
                server = "1" + server
            mode = "警備" if server == "1268" else "奪取"
            txt = f"{mode} {server}-{place}-{t}"
            dt = datetime.combine(now_jst().date(), datetime.strptime(t, "%H:%M:%S").time(), tzinfo=JST)
            if txt not in pending_places:
                pending_places[txt] = (dt, txt, server, now_jst())
                await message.channel.send(f"✅手動登録:{txt}")
                task = asyncio.create_task(handle_new_event(dt, txt, channel))
                active_tasks.add(task)
                task.add_done_callback(lambda t: active_tasks.discard(t))
                if txt.startswith("奪取"):
                    task2 = asyncio.create_task(schedule_notification(dt, txt, channel))
                    active_tasks.add(task2)
                    task2.add_done_callback(lambda t: active_tasks.discard(t))
        return

    if message.attachments:
        status = await message.channel.send("🔄解析中…")
        new_results = []
        for a in message.attachments:
            b = await a.read()
            img = Image.open(io.BytesIO(b)).convert("RGB")
            np_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            top = crop_top_right(np_img)
            center = crop_center_area(np_img)
            top_txts = extract_text_from_image(top)
            center_txts = extract_text_from_image(center)
            parsed = parse_multiple_places(center_txts, top_txts)
            for dt, txt in parsed:
                if txt not in pending_places:
                    pending_places[txt] = (dt, txt, "", now_jst())
                    new_results.append(txt)
                    task = asyncio.create_task(handle_new_event(dt, txt, channel))
                    active_tasks.add(task)
                    task.add_done_callback(lambda t: active_tasks.discard(t))
                    if txt.startswith("奪取"):
                        task2 = asyncio.create_task(schedule_notification(dt, txt, channel))
                        active_tasks.add(task2)
                        task2.add_done_callback(lambda t: active_tasks.discard(t))
        await status.edit(content=(
            "✅ OCR読み取り完了！登録された予定:\n" + "\n".join([f"・{txt}" for txt in new_results])
            if new_results else "⚠️ OCR処理完了しましたが、新しい予定は見つかりませんでした。"
        ))

# =======================
# 起動
# =======================
import asyncio

async def start_discord_bot():
    await client.start(TOKEN)

async def main():
    # FastAPIをバックグラウンドで起動
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, run_server)
    
    # Discord bot を開始（client.run ではなく start）
    await start_discord_bot()

if __name__ == "__main__":
    asyncio.run(main())