# OCR BOT（スケジュール通知付き + HTTPサーバーでUptimeRobot対応）
import os
import discord
import io
import cv2
import re
import asyncio
import numpy as np
from paddleocr import PaddleOCR
from datetime import datetime, timedelta, timezone, time
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
COPY_CHANNEL_ID = int(os.getenv("COPY_CHANNEL_ID", "0"))
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
    import time as _time
    _time.sleep(3)  # サービス安定のために3秒遅延
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

# =======================
# OCR初期化
# =======================
ocr = PaddleOCR(use_angle_cls=True, lang='japan')

# =======================
# 管理構造
# =======================
# txt: str -> {
#     "dt": datetime,
#     "txt": str,
#     "server": str,
#     "created_at": datetime,
#     "main_msg_id": Optional[int],
#     "copy_msg_id": Optional[int]
# }
pending_places = {}
copy_queue = []
summary_blocks = []
pending_copy_queue = []
manual_summary_msg_ids = []
active_tasks = set()
sent_notifications = set()
sent_notifications_tasks = {}
SKIP_NOTIFY_START = 2
SKIP_NOTIFY_END = 14

def store_copy_msg_id(txt, msg_id):
    if txt in pending_places:
        pending_places[txt]["copy_msg_id"] = msg_id

# =======================
# 過去予定の自動削除
# =======================
EXPIRE_GRACE = timedelta(minutes=2)  # 終了から2分猶予してから削除

async def remove_expired_entries():
    now = now_jst()

    # いま存在するブロックまとめのメッセージID一覧を先に集める（まとめ誤消し防止）
    block_msg_ids = {b["msg"].id for b in summary_blocks if b.get("msg")}

    # pending_placesの削除 + メッセージも削除
    for k, v in list(pending_places.items()):
        dt = v["dt"]
        if dt + EXPIRE_GRACE < now:
            # 通知チャンネルの削除（ブロックまとめのIDは除外）
            if v.get("main_msg_id") and v["main_msg_id"] not in block_msg_ids:
                ch = client.get_channel(NOTIFY_CHANNEL_ID)
                try:
                    msg = await ch.fetch_message(v["main_msg_id"])
                    await msg.delete()
                except:
                    pass

            # コピー用チャンネルの削除
            if v.get("copy_msg_id"):
                ch = client.get_channel(COPY_CHANNEL_ID)
                try:
                    msg = await ch.fetch_message(v["copy_msg_id"])
                    await msg.delete()
                except:
                    pass

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
        if task.done():
            continue
        try:
            unlock_dt = task.get_coro().cr_frame.f_locals.get("unlock_dt")
            if isinstance(unlock_dt, datetime) and unlock_dt + EXPIRE_GRACE < now:
                task.cancel()
        except:
            pass

# =======================
# ユーティリティ
# =======================
def now_jst():
    return datetime.now(JST)

def cleanup_old_entries():
    now = now_jst()
    for k in list(pending_places):
        if (now - pending_places[k]["created_at"]) > timedelta(hours=6):
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
    if base_time < datetime.strptime("06:00:00", "%H:%M:%S").time():
        base_dt += timedelta(days=1)  # 翌日扱い
    parts = duration_str.split(":")
    if len(parts) == 3:
        h, m, s = map(int, parts)
    elif len(parts) == 2:
        h, m, s = 0, *map(int, parts)
    else:
        return None, None
    dt = base_dt + timedelta(hours=h, minutes=m, seconds=s)
    return dt, dt.strftime("%H:%M:%S")

def extract_imsen_durations(texts: list[str]) -> list[str]:
    durations = []
    for text in texts:
        matches = re.findall(r"免戦中([0-9:\-日分秒hmsHMShms％%日]+)", text)
        for raw in matches:
            corrected = correct_imsen_text(raw)
            durations.append(corrected)
    return durations

def parse_multiple_places(center_texts, top_time_texts):
    res = []

    # 上部時間の抽出（補正後）
    def extract_top_time(txts):
        for t in txts:
            if re.fullmatch(r"\d{2}:\d{2}:\d{2}", t):
                return t
        for t in txts:
            digits = re.sub(r"[^\d]", "", t)
            if len(digits) >= 6:
                h, m, s = digits[:2], digits[2:4], digits[4:6]
                return f"{int(h):02}:{int(m):02}:{int(s):02}"
        return None

    top_time = extract_top_time(top_time_texts)
    server = extract_server_number(center_texts)
    if not top_time or not server:
        return []

    mode = "警備" if server == "1268" else "奪取"

    # ✅ グループ構築
    groups = []
    current_group = {"place": None, "lines": []}

    for line in center_texts:
        match = re.search(r"越域駐騎場(\d+)", line)
        if match:
            if current_group["place"] and current_group["lines"]:
                groups.append(current_group)
            current_group = {"place": match.group(1), "lines": []}
        else:
            current_group["lines"].append(line)

    if current_group["place"] and current_group["lines"]:
        groups.append(current_group)

    # ✅ 各グループの免戦時間抽出
    for g in groups:
        durations = extract_imsen_durations(g["lines"])
        if not durations:
            continue
        raw_d = durations[0]
        d = correct_imsen_text(raw_d)
        dt, unlock = add_time(top_time, d)
        if dt:
            res.append((dt, f"{mode} {server}-{g['place']}-{unlock}", d))

    return res

def correct_imsen_text(text: str) -> str:
    digits = re.sub(r"\D", "", text)

    if ":" in text:
        parts = re.findall(r"\d+", text)
        digits = "".join(parts)

    if len(digits) >= 7:
        try:
            h = int(digits[0:2]); m = int(digits[2:4]); s = int(digits[4:6])
            if 0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60:
                return f"{h:02}:{m:02}:{s:02}"
        except:
            pass

    if len(digits) == 6:
        try:
            h, m, s = int(digits[:2]), int(digits[2:4]), int(digits[4:])
            if 0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60:
                return f"{h:02}:{m:02}:{s:02}"
        except:
            pass

    if len(digits) == 5:
        try:
            h, m, s = int(digits[0]), int(digits[1:3]), int(digits[3:])
            if 0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60:
                return f"{h:02}:{m:02}:{s:02}"
        except:
            pass

    if len(digits) == 4:
        try:
            m, s = int(digits[:2]), int(digits[2:])
            if 0 <= m < 60 and 0 <= s < 60:
                return f"00:{m:02}:{s:02}"
        except:
            pass

    return text

# =======================
# ブロック・通知処理
# =======================
async def send_to_copy_channel(dt, txt):
    if COPY_CHANNEL_ID == 0:
        return None
    channel = client.get_channel(COPY_CHANNEL_ID)
    if not channel:
        return None

    msg = await channel.send(content=txt.replace("🕒 ", ""))

    # 🔸 削除処理だけ別タスクで起動（非同期）
    async def auto_delete():
        await asyncio.sleep(max(0, (dt - now_jst()).total_seconds() + 120))
        try:
            await msg.delete()
        except:
            pass

    asyncio.create_task(auto_delete())
    return msg.id

def find_or_create_block(new_dt):
    for block in summary_blocks:
        if new_dt <= block["max"] + timedelta(minutes=45):
            return block
    # task と lock を追加
    new_block = {
        "events": [],
        "min": new_dt,
        "max": new_dt,
        "msg": None,
        "task": None,
        "lock": asyncio.Lock(),
    }
    summary_blocks.append(new_block)
    return new_block
import math

def format_block_msg(block, with_footer=True):
    lines = ["⏰ スケジュールのお知らせ📢", ""]
    unique_events = sorted(set(block["events"]), key=lambda x: x[0])
    lines += [f"{txt}  " for _, txt in unique_events]
    if with_footer:
        diff = math.ceil((block["min"] - now_jst()).total_seconds() / 60)
        lines += ["", f"⚠️ {diff}分後に始まるよ⚠️" if diff < 30 else "⚠️ 30分後に始まるよ⚠️"]
    return "\n".join(lines)

async def schedule_block_summary(block, channel):
    try:
        # 開始30分前の案内
        await asyncio.sleep(max(0, (block["min"] - timedelta(minutes=30) - now_jst()).total_seconds()))

        if not block["msg"]:
            block["msg"] = await channel.send(format_block_msg(block, True))
        else:
            try:
                await block["msg"].edit(content=format_block_msg(block, True))
            except discord.NotFound:
                block["msg"] = await channel.send(format_block_msg(block, True))

        # 開始時刻になったらフッター差し替え
        await asyncio.sleep(max(0, (block["min"] - now_jst()).total_seconds()))
        if block["msg"]:
            try:
                await block["msg"].edit(content=format_block_msg(block, False))
            except discord.NotFound:
                pass
    except Exception as e:
        print(f"[ERROR] schedule_block_summary failed: {e}")
    finally:
        # タスク参照を必ずクリア（多重起動防止のため）
        block["task"] = None

async def handle_new_event(dt, txt, channel):
    block = find_or_create_block(dt)

    # 予定を追加
    if (dt, txt) not in block["events"]:
        block["events"].append((dt, txt))

    # ブロックの範囲更新
    block["min"] = min(block["min"], dt)
    block["max"] = max(block["max"], dt)

    # 古いイベントを整理（今回追加分は必ず残す）
    now = now_jst()
    block["events"] = [(d, t) for (d, t) in block["events"] if (t in pending_places or t == txt) and d > now]

    # すでにまとめメッセージがあるなら編集
    if block["msg"]:
        try:
            await block["msg"].edit(content=format_block_msg(block, True))
            if txt in pending_places:
                pending_places[txt]["main_msg_id"] = block["msg"].id
        except discord.NotFound:
            block["msg"] = await channel.send(format_block_msg(block, True))
            if txt in pending_places:
                pending_places[txt]["main_msg_id"] = block["msg"].id
        return

    # まとめメッセージがまだ無い場合：タスクを1本だけ起動
    async with block["lock"]:
        if block["task"] is None or block["task"].done():
            task = asyncio.create_task(schedule_block_summary(block, channel))
            block["task"] = task
            active_tasks.add(task)
            task.add_done_callback(lambda t: active_tasks.discard(t))

def is_within_5_minutes_of_another(target_dt):
    times = sorted([v["dt"] for v in pending_places.values()])
    for dt in times:
        if dt != target_dt and abs((dt - target_dt).total_seconds()) <= 300:
            return True
    return False

async def schedule_notification(unlock_dt, text, channel):
    if unlock_dt <= now_jst():
        return

    # 通知時間制限: 00:00〜06:00はスキップ
    if 0 <= unlock_dt.hour < 6:
        return

    if text.startswith("奪取"):
        now = now_jst()
        t_2min = unlock_dt - timedelta(minutes=2)
        t_15s = unlock_dt - timedelta(seconds=15)

        async def notify_2min():
            if t_2min > now and (text, "2min") not in sent_notifications and not is_within_5_minutes_of_another(unlock_dt):
                sent_notifications.add((text, "2min"))
                await asyncio.sleep((t_2min - now_jst()).total_seconds())
                msg = await channel.send(f"⏰ {text} **2分前です！！**")
                await asyncio.sleep(120)
                await msg.delete()

        async def notify_15s():
            if t_15s > now and (text, "15s") not in sent_notifications:
                sent_notifications.add((text, "15s"))
                await asyncio.sleep((t_15s - now_jst()).total_seconds())
                msg = await channel.send(f"⏰ {text} **15秒前です！！**")
                await asyncio.sleep(120)
                await msg.delete()

        sent_notifications_tasks[(text, "2min")] = asyncio.create_task(notify_2min())
        sent_notifications_tasks[(text, "15s")] = asyncio.create_task(notify_15s())

async def process_copy_queue():
    while True:
        await asyncio.sleep(30)
        if pending_copy_queue:
            queue_copy = sorted(pending_copy_queue, key=lambda x: x[0])
            pending_copy_queue.clear()
            for dt, txt in queue_copy:
                msg = await send_to_copy_channel(dt, txt)
                store_copy_msg_id(txt, msg)

# =======================
# 自動リセット処理（毎日02:00）
# =======================
async def daily_reset_task():
    await client.wait_until_ready()
    while not client.is_closed():
        now = now_jst()
        next_reset = datetime.combine(now.date(), datetime.strptime("02:00:00", "%H:%M:%S").time(), tzinfo=JST)
        if now >= next_reset:
            next_reset += timedelta(days=1)
        await asyncio.sleep((next_reset - now).total_seconds())

        # ✅ チャンネル上のメッセージ削除処理追加
        for entry in list(pending_places.values()):
            if entry.get("main_msg_id"):
                ch = client.get_channel(NOTIFY_CHANNEL_ID)
                try:
                    msg = await ch.fetch_message(entry["main_msg_id"])
                    await msg.delete()
                except:
                    pass

            if entry.get("copy_msg_id"):
                ch = client.get_channel(COPY_CHANNEL_ID)
                try:
                    msg = await ch.fetch_message(entry["copy_msg_id"])
                    await msg.delete()
                except:
                    pass

        # summary_blocks まとめメッセージ削除
        for block in summary_blocks:
            if block.get("msg"):
                try:
                    await block["msg"].delete()
                except:
                    pass

        # 通知予約(2分前/15秒前)タスクのキャンセル
        for key, task in list(sent_notifications_tasks.items()):
            task.cancel()
        sent_notifications_tasks.clear()

        # 手動通知(!s)のまとめメッセージ削除
        if manual_summary_msg_ids:
            ch2 = client.get_channel(NOTIFY_CHANNEL_ID)
            if ch2:
                for mid in list(manual_summary_msg_ids):
                    try:
                        msg = await ch2.fetch_message(mid)
                        await msg.delete()
                    except:
                        pass
            manual_summary_msg_ids.clear()

        # 内部状態の初期化
        pending_places.clear()
        summary_blocks.clear()
        sent_notifications.clear()
        for task in list(active_tasks):
            task.cancel()
        active_tasks.clear()
        # ✅ 通知は送らない（silent reset）

# =======================
# 過去予定の定期削除（1分ごと）
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
    # 通知/コピーの個別メッセージ削除（登録分）
    for entry in list(pending_places.values()):
        if entry.get("main_msg_id"):
            ch = client.get_channel(NOTIFY_CHANNEL_ID)
            try:
                msg = await ch.fetch_message(entry["main_msg_id"])
                await msg.delete()
            except:
                pass
        if entry.get("copy_msg_id"):
            ch = client.get_channel(COPY_CHANNEL_ID)
            try:
                msg = await ch.fetch_message(entry["copy_msg_id"])
                await msg.delete()
            except:
                pass

    # まとめメッセージ削除
    for block in list(summary_blocks):
        if block.get("msg"):
            try:
                await block["msg"].delete()
            except:
                pass
    summary_blocks.clear()

    # 手動通知(!s)まとめ削除
    if manual_summary_msg_ids:
        ch2 = client.get_channel(NOTIFY_CHANNEL_ID)
        if ch2:
            for mid in list(manual_summary_msg_ids):
                try:
                    msg = await ch2.fetch_message(mid)
                    await msg.delete()
                except:
                    pass
        manual_summary_msg_ids.clear()

    # 予約タスクのキャンセル＆一覧クリア
    for key, task in list(sent_notifications_tasks.items()):
        task.cancel()
    sent_notifications_tasks.clear()
    sent_notifications.clear()

    # 保険：コピー用チャンネルの直近bot投稿を軽くパージ（取りこぼし対策）
    try:
        ch_copy = client.get_channel(COPY_CHANNEL_ID)
        if ch_copy:
            async for m in ch_copy.history(limit=100):
                if m.author == client.user:
                    try:
                        await m.delete()
                    except:
                        pass
    except:
        pass

    # 状態クリア
    pending_places.clear()

    # 他タスクキャンセル
    for t in list(active_tasks):
        t.cancel()
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
    asyncio.create_task(daily_reset_task())      # ✅ 自動リセット
    asyncio.create_task(periodic_cleanup_task()) # ✅ 過去予定の削除
    asyncio.create_task(process_copy_queue())    # ✅ コピーキュー処理

async def auto_dedup():
    seen = {}
    to_remove = []

    for v in pending_places.values():
        match = re.fullmatch(r"(奪取|警備)\s+(\d{4})-(\d+)-(\d{2}:\d{2}:\d{2})", v["txt"])
        if not match:
            continue
        mode, server, place, timestr = match.groups()
        key = (server, place)
        current_dt = v["dt"]

        if key not in seen:
            seen[key] = (current_dt, v["txt"])
        else:
            prev_dt, prev_txt = seen[key]
            if current_dt < prev_dt:
                to_remove.append(prev_txt)
                seen[key] = (current_dt, v["txt"])
            else:
                to_remove.append(v["txt"])

    for txt in to_remove:
        if txt in pending_places:
            entry = pending_places.pop(txt)

            if entry.get("main_msg_id"):
                ch = client.get_channel(NOTIFY_CHANNEL_ID)
                try:
                    msg = await ch.fetch_message(entry["main_msg_id"])
                    await msg.delete()
                except:
                    pass

            if entry.get("copy_msg_id"):
                ch = client.get_channel(COPY_CHANNEL_ID)
                try:
                    msg = await ch.fetch_message(entry["copy_msg_id"])
                    await msg.delete()
                except:
                    pass

@client.event
async def on_message(message):
    if message.author.bot or message.channel.id not in READABLE_CHANNEL_IDS:
        return

    cleanup_old_entries()
    channel = client.get_channel(NOTIFY_CHANNEL_ID)

    # ==== !reset ====
    if message.content.strip() == "!reset":
        await reset_all(message)
        return

    # ==== !del 奪取 1272-4-06:24:35 ====
    match = re.fullmatch(r"!del\s+(奪取|警備)\s+(\d{4})-(\d+)-(\d{2}:\d{2}:\d{2})", message.content.strip())
    if match:
        mode, server, place, t = match.groups()
        txt = f"{mode} {server}-{place}-{t}"
        removed = False

        # pending から除去＆コピー用メッセージ削除
        copy_msg_id = None
        if txt in pending_places:
            entry = pending_places.pop(txt)
            removed = True
            copy_msg_id = entry.get("copy_msg_id")

        # 予約タスクのキャンセル（!n からも消す）
        for key in [(txt, "2min"), (txt, "15s")]:
            task = sent_notifications_tasks.pop(key, None)
            if task:
                task.cancel()

        # コピー用チャンネルの当該メッセージ削除
        if copy_msg_id:
            ch_copy = client.get_channel(COPY_CHANNEL_ID)
            if ch_copy:
                try:
                    msg = await ch_copy.fetch_message(copy_msg_id)
                    await msg.delete()
                except:
                    pass

        # summary_blocks からも該当行を消す（空になったらまとめを削除）
        for block in list(summary_blocks):
            before = len(block["events"])
            block["events"] = [ev for ev in block["events"] if ev[1] != txt]
            after = len(block["events"])
            if before != after:
                removed = True
                if block["events"]:
                    block["min"] = min(ev[0] for ev in block["events"])
                    block["max"] = max(ev[0] for ev in block["events"])
                    if block.get("msg"):
                        try:
                            await block["msg"].edit(content=format_block_msg(block, True))
                        except:
                            pass
                else:
                    # もうこのまとめは不要なのでメッセージを消す
                    if block.get("msg"):
                        try:
                            await block["msg"].delete()
                        except:
                            pass
                    summary_blocks.remove(block)

        if removed:
            await message.channel.send(f"🗑️ 予定を削除しました: {txt}")
        else:
            await message.channel.send(f"⚠️ 該当する予定が見つかりません: {txt}")
        return
        
    # ==== !debug ====
    if message.content.strip() == "!debug":
        if pending_places:
            lines = ["✅ 現在の登録された予定:"]
            lines += [f"・{v['txt']}" for v in sorted(pending_places.values(), key=lambda x: x["dt"])]
            await message.channel.send("\n".join(lines))
        else:
            await message.channel.send("⚠️ 登録された予定はありません")
        return

    # ==== !s ====
    if message.content.strip() == "!s":
        if not pending_places:
            await message.channel.send("⚠️ 登録された予定はありません")
            return

        ch = client.get_channel(NOTIFY_CHANNEL_ID)
        if not ch:
            await message.channel.send("⚠️ 通知チャンネルが見つかりません")
            return

        sorted_places = sorted(pending_places.values(), key=lambda x: x["dt"])
        lines = ["📢 手動通知: 現在登録されているスケジュール一覧", ""]
        for v in sorted_places:
            lines.append(f"{v['txt']}")

        try:
            msg = await ch.send("\n".join(lines))
            # まとめメッセージは予定ごとの main_msg_id に紐付けない
            manual_summary_msg_ids.append(msg.id)
        except:
            await message.channel.send("⚠️ 通知の送信に失敗しました")
            return

        await message.channel.send("📤 通知チャンネルへ送信しました")
        return

    # ==== !c ====
    if message.content.strip() == "!c":
        if not pending_places:
            await message.channel.send("⚠️ 登録された予定はありません")
            return

        ch = client.get_channel(COPY_CHANNEL_ID)
        if not ch:
            await message.channel.send("⚠️ コピー用チャンネルが見つかりません")
            return

        sorted_places = sorted(pending_places.values(), key=lambda x: x["dt"])
        for v in sorted_places:
            txt = v["txt"]
            try:
                msg = await ch.send(content=txt.replace("🕒 ", ""))
                v["copy_msg_id"] = msg.id
            except:
                pass

        await message.channel.send("📤 コピー用チャンネルへ送信しました")
        return

    # ==== !n ====
    if message.content.strip() == "!n":
        if not sent_notifications_tasks:
            await message.channel.send("⚠️ 通知予約はありません")
            return

        two_min_lines = ["🕑 **2分前通知予約**:"]
        fifteen_sec_lines = ["⏱ **15秒前通知予約**:"]
        for (txt, kind), task in sorted(sent_notifications_tasks.items(), key=lambda x: (x[0][1], x[0][0])):
            status = " (キャンセル済)" if task.cancelled() else ""
            if kind == "2min":
                two_min_lines.append(f"・{txt}{status}")
            elif kind == "15s":
                fifteen_sec_lines.append(f"・{txt}{status}")

        msg = "\n".join(two_min_lines + [""] + fifteen_sec_lines)
        await message.channel.send(msg)
        return

    # ==== !ocrdebug ====
    if message.content.strip() == "!ocrdebug":
        if not message.attachments:
            await message.channel.send("⚠️ 画像を添付してください（OCR結果とトリミング画像を確認します）")
            return

        a = message.attachments[0]
        b = await a.read()
        img = Image.open(io.BytesIO(b)).convert("RGB")
        np_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # トリミング
        top = crop_top_right(np_img)
        center = crop_center_area(np_img)

        # OCRテキスト抽出
        top_txts = extract_text_from_image(top)
        center_txts = extract_text_from_image(center)

        # 補正関数
        def extract_and_correct_base_time(txts):
            if not txts:
                return "??:??:??"
            raw = txts[0].strip()
            digits = re.sub(r"\D", "", raw)
            if len(digits) >= 8:
                try:
                    h = int(digits[0:2]); m = int(digits[2:4]); s = int(digits[6:8])
                    if 0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60:
                        return f"{h:02}:{m:02}:{s:02}"
                except:
                    pass
            if len(digits) >= 6:
                try:
                    h, m, s = int(digits[:2]), int(digits[2:4]), int(digits[4:6])
                    if 0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60:
                        return f"{h:02}:{m:02}:{s:02}"
                except:
                    pass
            if len(digits) == 5:
                try:
                    h, m, s = int(digits[0]), int(digits[1:3]), int(digits[3:])
                    if 0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60:
                        return f"{h:02}:{m:02}:{s:02}"
                except:
                    pass
            if len(digits) == 4:
                try:
                    m, s = int(digits[:2]), int(digits[2:])
                    if 0 <= m < 60 and 0 <= s < 60:
                        return f"00:{m:02}:{s:02}"
                except:
                    pass
            return "??:??:??"

        # 予定抽出
        parsed_preview = parse_multiple_places(center_txts, top_txts)
        preview_lines = [f"・{txt}" for _, txt, _ in parsed_preview] if parsed_preview else ["(なし)"]
        preview_text = "\n".join(preview_lines)

        # 免戦時間抽出 ＋ 補正
        durations = extract_imsen_durations(center_txts)
        duration_text = "\n".join(durations) if durations else "(抽出なし)"

        # 上部OCR結果を安全に整形
        top_txts_str = "\n".join(top_txts) if top_txts else "(検出なし)"

        # 送信
        await message.channel.send(
            f"📸 **上部OCR結果（基準時刻）**:\n```\n{top_txts_str}\n```\n"
            f"📋 **補正後の予定一覧（奪取 or 警備）**:\n```\n{preview_text}\n```\n"
            f"⏳ **補正後の免戦時間一覧**:\n```\n{duration_text}\n```"
        )
        return

    # ==== !a 奪取 1234-1-12:00:00 130000 or 13:00:00 ====
    match = re.fullmatch(r"!a\s+(奪取|警備)\s+(\d{4})-(\d+)-(\d{2}:\d{2}:\d{2})\s+(\d{6}|\d{1,2}:\d{2}:\d{2})", message.content.strip())
    if match:
        mode, server, place, timestr, raw = match.groups()
        old_txt = f"{mode} {server}-{place}-{timestr}"

        try:
            if ":" in raw:
                h, m, s = map(int, raw.split(":"))
            else:
                h, m, s = int(raw[:2]), int(raw[2:4]), int(raw[4:])
        except:
            await message.channel.send("⚠️ 時間の指定が不正です")
            return

        base = datetime.strptime(timestr, "%H:%M:%S").replace(tzinfo=JST)
        new_dt = base.replace(hour=h, minute=m, second=s)

        # ⏰ 00:00:00〜05:59:59 の場合は日付を翌日に補正
        if timedelta(hours=h, minutes=m, seconds=s) < timedelta(hours=6):
            new_dt += timedelta(days=1)
        new_txt = f"{mode} {server}-{place}-{new_dt.strftime('%H:%M:%S')}"

        # 既存の予約とメッセージの扱い（コピーは「削除せず編集」に変更）
        old_copy_msg_id = None
        if old_txt in pending_places:
            old_entry = pending_places.pop(old_txt)
            # 通知予約タスクのキャンセル（一覧 !n からも確実に消す）
            for key in [(old_txt, "2min"), (old_txt, "15s")]:
                task = sent_notifications_tasks.pop(key, None)
                if task:
                    task.cancel()

            # 通知まとめメッセージは消さず（この後編集で反映）
            # コピー用は「送信済みメッセージを編集」するためIDだけ保持
            old_copy_msg_id = old_entry.get("copy_msg_id")

        # 新規（更新後）を登録
        pending_places[new_txt] = {
            "dt": new_dt,
            "txt": new_txt,
            "server": server,
            "created_at": now_jst(),
            "main_msg_id": None,
            "copy_msg_id": None,
        }

        # ブロック反映（まとめメッセージ側の行を差し替える）
        block = find_or_create_block(new_dt)
        replaced = False
        for i, (ev_dt, ev_txt) in enumerate(block["events"]):
            if ev_txt == old_txt:
                block["events"][i] = (new_dt, new_txt)
                replaced = True
                break
        if not replaced:
            block["events"].append((new_dt, new_txt))

        if block["msg"]:
            try:
                await block["msg"].edit(content=format_block_msg(block, True))
                pending_places[new_txt]["main_msg_id"] = block["msg"].id
            except:
                pass
        else:
            # まとめメッセージが未作成ならスケジューラに任せる
            pass

        # コピー用チャンネル：旧メッセージがあれば上書き編集、無ければ新規送信
        copy_ch = client.get_channel(COPY_CHANNEL_ID)
        if copy_ch:
            if old_copy_msg_id:
                try:
                    msg = await copy_ch.fetch_message(old_copy_msg_id)
                    await msg.edit(content=new_txt.replace("🕒 ", ""))
                    pending_places[new_txt]["copy_msg_id"] = msg.id
                except:
                    # 取得できなければ新規送信
                    msg = await copy_ch.send(content=new_txt.replace("🕒 ", ""))
                    pending_places[new_txt]["copy_msg_id"] = msg.id
            else:
                msg = await copy_ch.send(content=new_txt.replace("🕒 ", ""))
                pending_places[new_txt]["copy_msg_id"] = msg.id

        # 通知スケジュールへ再登録（!n にも即時反映される）
        notify_ch = client.get_channel(NOTIFY_CHANNEL_ID)
        if notify_ch:
            await schedule_notification(new_dt, new_txt, notify_ch)

        await message.channel.send(f"✅ 更新しました → `{new_txt}`")
        return

    # ==== 手動追加（例: 1234-1-12:34:56）====
    manual = re.findall(r"\b(\d{3,4})-(\d+)-(\d{2}:\d{2}:\d{2})\b", message.content)
    if manual:
        for server, place, t in manual:
            if len(server) == 3:
                server = "1" + server
            mode = "警備" if server == "1268" else "奪取"
            txt = f"{mode} {server}-{place}-{t}"
            t_obj = datetime.strptime(t, "%H:%M:%S").time()
            dt = datetime.combine(now_jst().date(), t_obj, tzinfo=JST)
            if t_obj < time(6, 0, 0):
                dt += timedelta(days=1)
            if txt not in pending_places:
                pending_places[txt] = {
                    "dt": dt,
                    "txt": txt,
                    "server": server,
                    "created_at": now_jst(),
                    "main_msg_id": None,
                    "copy_msg_id": None,
                }
                await message.channel.send(f"✅手動登録:{txt}")
                task = asyncio.create_task(handle_new_event(dt, txt, channel))
                active_tasks.add(task)
                task.add_done_callback(lambda t: active_tasks.discard(t))
                if txt.startswith("奪取"):
                    task2 = asyncio.create_task(schedule_notification(dt, txt, channel))
                    active_tasks.add(task2)
                    task2.add_done_callback(lambda t: active_tasks.discard(t))
        return

    # ==== 通常画像送信 ====
    if message.attachments:
        status = await message.channel.send("🔄解析中…")
        grouped_results = []

        def extract_and_correct_base_time(txts):
            if not txts:
                return "??:??:??"
            raw = txts[0].strip()
            digits = re.sub(r"\D", "", raw)
            if len(digits) >= 8:
                try:
                    h = int(digits[0:2]); m = int(digits[2:4]); s = int(digits[6:8])
                    if 0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60:
                        return f"{h:02}:{m:02}:{s:02}"
                except:
                    pass
            if len(digits) >= 6:
                try:
                    h, m, s = int(digits[:2]), int(digits[2:4]), int(digits[4:6])
                    if 0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60:
                        return f"{h:02}:{m:02}:{s:02}"
                except:
                    pass
            if len(digits) == 5:
                try:
                    h, m, s = int(digits[0]), int(digits[1:3]), int(digits[3:])
                    if 0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60:
                        return f"{h:02}:{m:02}:{s:02}"
                except:
                    pass
            if len(digits) == 4:
                try:
                    m, s = int(digits[:2]), int(digits[2:])
                    if 0 <= m < 60 and 0 <= s < 60:
                        return f"00:{m:02}:{s:02}"
                except:
                    pass
            return "??:??:??"

        for a in message.attachments:
            b = await a.read()
            img = Image.open(io.BytesIO(b)).convert("RGB")
            np_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            top = crop_top_right(np_img)
            center = crop_center_area(np_img)
            top_txts = extract_text_from_image(top)
            center_txts = extract_text_from_image(center)

            base_time = extract_and_correct_base_time(top_txts)
            parsed = parse_multiple_places(center_txts, top_txts)

            image_results = []
            for dt, txt, raw in parsed:
                if txt not in pending_places:
                    pending_places[txt] = {
                        "dt": dt,
                        "txt": txt,
                        "server": "",
                        "created_at": now_jst(),
                        "main_msg_id": None,
                        "copy_msg_id": None,
                    }
                    # ✅ 自動重複除去（同じサーバー・駐機場で後の時刻を削除）
                    await auto_dedup()
                    display_txt = f"{txt} ({raw})"
                    image_results.append(display_txt)
                    task = asyncio.create_task(handle_new_event(dt, txt, channel))
                    active_tasks.add(task)
                    task.add_done_callback(lambda t: active_tasks.discard(t))
                    if txt.startswith("奪取"):
                        task2 = asyncio.create_task(schedule_notification(dt, txt, channel))
                        active_tasks.add(task2)
                        task2.add_done_callback(lambda t: active_tasks.discard(t))

            if image_results:
                grouped_results.append((base_time, image_results))

        if grouped_results:
            lines = [
                "✅ 解析完了！登録されました",
                "",
                "🧭 **次の操作:**",
                "　📤 `!c` → ⏰ 時間コピー用チャンネルにスケジュールを送信",
                "　📢 `!s` → 📝 通知チャンネルに手動でスケジュールを通知",
                "",
            ]
            for base_time, txts in grouped_results:
                lines.append(f"📸 [基準時間: {base_time}]")
                lines += [f"・{txt}" for txt in txts]
                lines.append("")
            await status.edit(content="\n".join(lines))
        else:
            await status.edit(content="⚠️ 解析完了しましたが、新しい予定は見つかりませんでした。")
        return

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