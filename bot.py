# OCR BOT（スケジュール通知付き）
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
# ユーティリティ
# =======================
def now_jst():
    return datetime.now(JST)

def cleanup_old_entries():
    now = now_jst()
    for k in list(pending_places):
        if (now - pending_places[k][3]) > timedelta(hours=6):
            del pending_places[k]

def crop_top_right(img):
    h, w = img.shape[:2]
    return img[0:int(h*0.2), int(w*0.7):]

def crop_center_area(img):
    h, w = img.shape[:2]
    return img[int(h*0.35):int(h*0.65), :]

def extract_text_from_image(img):
    result = ocr.ocr(img, cls=True)
    return [line[1][0] for line in result[0]] if result and result[0] else []

def extract_server_number(center_texts):
    for t in center_texts:
        m = re.search(r"[sS](\d{3,4})", t)
        if m:
            return m.group(1)
    return None

def add_time(base_time_str, duration_str):
    today = now_jst().date()
    try:
        base_time = datetime.strptime(base_time_str, "%H:%M:%S").time()
    except:
        return None, None
    base_dt = datetime.combine(today, base_time, tzinfo=JST)
    parts = duration_str.split(":")
    if len(parts) == 3:
        h, m, s = map(int, parts)
    elif len(parts) == 2:
        h, m, s = 0, *map(int, parts)
    else:
        return None, None
    dt = base_dt + timedelta(hours=h, minutes=m, seconds=s)
    return dt, dt.strftime("%H:%M:%S")

def parse_multiple_places(center_texts, top_time_texts):
    res = []
    top_time = next((t for t in top_time_texts if re.match(r"\d{2}:\d{2}:\d{2}", t)), None)
    server = extract_server_number(center_texts)
    if not top_time or not server:
        return []
    mode = "警備" if server == "1268" else "奪取"
    current = None
    for t in center_texts:
        p = re.search(r"越域駐騎場(\d+)", t)
        if p:
            current = p.group(1)
        d = re.search(r"免戦中(\d{1,2}:\d{2}(?::\d{2})?)", t)
        if d and current:
            dt, unlock = add_time(top_time, d.group(1))
            if dt:
                res.append((dt, f"{mode} {server}-{current}-{unlock}"))
            current = None
    return res

# =======================
# ブロック・通知処理
# =======================
def find_or_create_block(new_dt):
    for block in summary_blocks:
        if new_dt <= block["max"] + timedelta(minutes=45):
            return block
    new_block = {"events": [], "min": new_dt, "max": new_dt, "msg": None}
    summary_blocks.append(new_block)
    return new_block

def format_block_msg(block, with_footer=True):
    lines = ["⏰ スケジュールのお知らせ📢", ""]
    unique_events = sorted(set(block["events"]), key=lambda x: x[0])
    lines += [f"{txt}  " for _, txt in unique_events]
    if with_footer:
        diff = int((block["min"] - now_jst()).total_seconds() // 60)
        lines += ["", f"⚠️ {diff}分後に始まるよ⚠️" if diff < 30 else "⚠️ 30分後に始まるよ⚠️"]
    return "\n".join(lines)

async def schedule_block_summary(block, channel):
    try:
        await asyncio.sleep(max(0, (block["min"] - timedelta(minutes=30) - now_jst()).total_seconds()))
        if not block["msg"]:
            block["msg"] = await channel.send(format_block_msg(block, True))
        else:
            try:
                await block["msg"].edit(content=format_block_msg(block, True))
            except discord.NotFound:
                block["msg"] = await channel.send(format_block_msg(block, True))

        await asyncio.sleep(max(0, (block["min"] - now_jst()).total_seconds()))
        if block["msg"]:
            try:
                await block["msg"].edit(content=format_block_msg(block, False))
            except discord.NotFound:
                pass  # メッセージが削除されていた場合は無視
    except Exception as e:
        print(f"[ERROR] schedule_block_summary failed: {e}")
async def handle_new_event(dt, txt, channel):
    block = find_or_create_block(dt)
    if (dt, txt) not in block["events"]:
        block["events"].append((dt, txt))
    block["min"] = min(block["min"], dt)
    block["max"] = max(block["max"], dt)
    if block["msg"]:
        await block["msg"].edit(content=format_block_msg(block, True))
    else:
        task = asyncio.create_task(schedule_block_summary(block, channel))
        active_tasks.add(task)
        task.add_done_callback(lambda t: active_tasks.discard(t))

def is_within_5_minutes_of_another(target_dt):
    times = sorted([v[0] for v in pending_places.values()])
    for dt in times:
        if dt != target_dt and abs((dt - target_dt).total_seconds()) <= 300:
            return True
    return False

async def schedule_notification(unlock_dt, text, channel):
    if unlock_dt <= now_jst(): return
    if text.startswith("奪取") and not (SKIP_NOTIFY_START <= unlock_dt.hour < SKIP_NOTIFY_END):
        # 2分前通知（5分以内に別の予定がなければ）
        if not is_within_5_minutes_of_another(unlock_dt):
            t = unlock_dt - timedelta(minutes=2)
            if t > now_jst() and (text, "2min") not in sent_notifications:
                sent_notifications.add((text, "2min"))
                await asyncio.sleep((t - now_jst()).total_seconds())
                await channel.send(f"⏰ {text} **2分前です！！**")
        # 15秒前通知（常に送信）
        t15 = unlock_dt - timedelta(seconds=15)
        if t15 > now_jst() and (text, "15s") not in sent_notifications:
            sent_notifications.add((text, "15s"))
            await asyncio.sleep((t15 - now_jst()).total_seconds())
            await channel.send(f"⏰ {text} **15秒前です！！**")

# =======================
# リセット処理
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

@client.event
async def on_message(message):
    if message.author.bot or message.channel.id not in READABLE_CHANNEL_IDS:
        return

    cleanup_old_entries()
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
if __name__ == "__main__":
    client.run(TOKEN)
