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
# txt: str -> {
#     "dt": datetime,
#     "txt": str,
#     "server": str,
#     "created_at": datetime,
#     "main_msg_id": Optional[int],
#     "copy_msg_id": Optional[int]
# }
pending_places = {}
summary_blocks = []
active_tasks = set()
sent_notifications = set()
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

    # pending_placesの削除 + メッセージも削除
    for k, v in list(pending_places.items()):
        dt = v["dt"]
        if dt + EXPIRE_GRACE < now:
            # 通知チャンネルの削除
            if "main_msg_id" in v and v["main_msg_id"]:
                ch = client.get_channel(NOTIFY_CHANNEL_ID)
                try:
                    msg = await ch.fetch_message(v["main_msg_id"])
                    await msg.delete()
                except:
                    pass

            # コピー用チャンネルの削除
            if "copy_msg_id" in v and v["copy_msg_id"]:
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
    if base_time < datetime.strptime("02:00:01", "%H:%M:%S").time():
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
            cleaned = re.sub(r"[^\d:]", "", raw)  # 数字と : のみ残す

            # コロンが2つ → 正常形式
            if cleaned.count(":") == 2:
                durations.append(cleaned)
                continue

            # コロン1つ → M:Sとみなす
            if cleaned.count(":") == 1:
                parts = cleaned.split(":")
                if len(parts) == 2:
                    m, s = parts
                    durations.append(f"00:{int(m):02}:{int(s):02}")
                    continue

            # コロンがない → 文字数によって補正
            numbers_only = re.sub(r"[^\d]", "", raw)
            if len(numbers_only) == 6:
                # HHMMSS
                h, m, s = numbers_only[:2], numbers_only[2:4], numbers_only[4:6]
                durations.append(f"{int(h):02}:{int(m):02}:{int(s):02}")
            elif len(numbers_only) == 5:
                # HMMSS
                h, m, s = numbers_only[:1], numbers_only[1:3], numbers_only[3:5]
                durations.append(f"{int(h):02}:{int(m):02}:{int(s):02}")
            elif len(numbers_only) == 4:
                # MMSS
                m, s = numbers_only[:2], numbers_only[2:4]
                durations.append(f"00:{int(m):02}:{int(s):02}")
            elif len(numbers_only) == 3:
                # MSS
                m, s = numbers_only[:1], numbers_only[1:3]
                durations.append(f"00:{int(m):02}:{int(s):02}")
            else:
                # フォールバック
                durations.append("00:00:00")

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
            # 新しい駐騎場番号を見つけたら前のグループ保存
            if current_group["place"] and current_group["lines"]:
                groups.append(current_group)
            # 新しいグループ開始
            current_group = {"place": match.group(1), "lines": []}
        else:
            current_group["lines"].append(line)

    # 最後のグループも追加
    if current_group["place"] and current_group["lines"]:
        groups.append(current_group)

    # ✅ 各グループの免戦時間抽出
    for g in groups:
        durations = extract_imsen_durations(g["lines"])
        if not durations:
            continue
        d = durations[0]  # 最初の免戦時間を使用
        dt, unlock = add_time(top_time, d)
        if dt:
            res.append((dt, f"{mode} {server}-{g['place']}-{unlock}", d))

    return res
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
    return msg.id  # ← この行が重要！
    await asyncio.sleep(max(0, (dt - now_jst()).total_seconds() + 120))  # 2分猶予で削除
    try:
        await msg.delete()
    except:
        pass
    return msg.id
def store_copy_msg_id(txt, msg_id):
    if txt in pending_places:
        pending_places[txt]["copy_msg_id"] = msg_id
def find_or_create_block(new_dt):
    for block in summary_blocks:
        if new_dt <= block["max"] + timedelta(minutes=45):
            return block
    new_block = {"events": [], "min": new_dt, "max": new_dt, "msg": None}
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
                pass
    except Exception as e:
        print(f"[ERROR] schedule_block_summary failed: {e}")

async def handle_new_event(dt, txt, channel):
    block = find_or_create_block(dt)
    if (dt, txt) not in block["events"]:
        block["events"].append((dt, txt))
        # 通常通知チャンネルに加え、コピー専用にも送信
        copy_task = asyncio.create_task(send_to_copy_channel(dt, txt))
        copy_task.add_done_callback(lambda t: store_copy_msg_id(txt, t.result()))

    block["min"] = min(block["min"], dt)
    block["max"] = max(block["max"], dt)

    # 🧹 古いイベントを削除（過去時刻 or 削除済みのもの）
    now = now_jst()
    block["events"] = [
        (d, t) for (d, t) in block["events"]
        if t in pending_places and d > now
    ]

    if block["msg"]:
        try:
            await block["msg"].edit(content=format_block_msg(block, True))
        except discord.NotFound:
            block["msg"] = await channel.send(format_block_msg(block, True))
            # 🆕 ここで main_msg_id を保存
            if txt in pending_places:
                pending_places[txt]["main_msg_id"] = block["msg"].id
    else:
        task = asyncio.create_task(schedule_block_summary(block, channel))
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

    # 通知時間制限: 02:00〜08:00はスキップ
    if not (8 <= unlock_dt.hour or unlock_dt.hour < 2):
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

        # 並列で通知をスケジュール
        asyncio.create_task(notify_2min())
        asyncio.create_task(notify_15s())
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
            if "main_msg_id" in entry and entry["main_msg_id"]:
                ch = client.get_channel(NOTIFY_CHANNEL_ID)
                try:
                    msg = await ch.fetch_message(entry["main_msg_id"])
                    await msg.delete()
                except:
                    pass

            if "copy_msg_id" in entry and entry["copy_msg_id"]:
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
    # 個別メッセージ削除（通知・コピー用）
    for entry in list(pending_places.values()):
        # 通知チャンネルの削除
        if "main_msg_id" in entry and entry["main_msg_id"]:
            ch = client.get_channel(NOTIFY_CHANNEL_ID)
            try:
                msg = await ch.fetch_message(entry["main_msg_id"])
                await msg.delete()
            except:
                pass

        # コピー用チャンネルの削除
        if "copy_msg_id" in entry and entry["copy_msg_id"]:
            ch = client.get_channel(COPY_CHANNEL_ID)
            try:
                msg = await ch.fetch_message(entry["copy_msg_id"])
                await msg.delete()
            except:
                pass

    # ⬇⬇⬇ まとめメッセージも削除 ⬇⬇⬇
    for block in summary_blocks:
        if block.get("msg"):
            try:
                await block["msg"].delete()
            except:
                pass

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

    cleanup_old_entries()
    channel = client.get_channel(NOTIFY_CHANNEL_ID)

    # ==== !reset ====
    if message.content.strip() == "!reset":
        # 通知チャンネルとコピー用チャンネルのメッセージ削除
        for entry in pending_places.values():
            # 通知チャンネルのメッセージ削除
            if "main_msg_id" in entry and entry["main_msg_id"]:
                ch = client.get_channel(NOTIFY_CHANNEL_ID)
                try:
                    msg = await ch.fetch_message(entry["main_msg_id"])
                    await msg.delete()
                except:
                    pass

            # コピー用チャンネルのメッセージ削除
            if "copy_msg_id" in entry and entry["copy_msg_id"]:
                ch = client.get_channel(COPY_CHANNEL_ID)
                try:
                    msg = await ch.fetch_message(entry["copy_msg_id"])
                    await msg.delete()
                except:
                    pass

        # 内部状態の初期化
        pending_places.clear()
        for task in list(active_tasks):
            task.cancel()
        summary_blocks.clear()

        await message.channel.send("♻️ 全予定をリセットしました")
        return

    # ==== !del 奪取 1272-4-06:24:35 ====
    match = re.fullmatch(r"!del\s+(奪取|警備)\s+(\d{4})-(\d+)-(\d{2}:\d{2}:\d{2})", message.content.strip())
    if match:
        mode, server, place, t = match.groups()
        txt = f"{mode} {server}-{place}-{t}"

        removed = False

        # pending_places から削除＋通知削除
        if txt in pending_places:
            entry = pending_places.pop(txt)
            removed = True

            # 通知チャンネルの削除
            if "main_msg_id" in entry and entry["main_msg_id"]:
                ch = client.get_channel(NOTIFY_CHANNEL_ID)
                try:
                    msg = await ch.fetch_message(entry["main_msg_id"])
                    await msg.delete()
                except:
                    pass

            # コピー用チャンネルの削除
            if "copy_msg_id" in entry and entry["copy_msg_id"]:
                ch = client.get_channel(COPY_CHANNEL_ID)
                try:
                    msg = await ch.fetch_message(entry["copy_msg_id"])
                    await msg.delete()
                except:
                    pass

        # summary_blocks から削除
        for block in summary_blocks:
            before = len(block["events"])
            block["events"] = [ev for ev in block["events"] if ev[1] != txt]
            after = len(block["events"])
            if before != after:
                removed = True
                if block["msg"]:
                    try:
                        await block["msg"].edit(content=format_block_msg(block, True))
                    except:
                        pass

        if removed:
            await message.channel.send(f"🗑️ 削除しました → `{txt}`")
        else:
            await message.channel.send(f"⚠️ 該当の予定が見つかりませんでした → `{txt}`")
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

            # 🧪 1つ飛ばし補正（例: "11814822" → "11:14:22"）
            if len(digits) >= 8:
                try:
                    h = int(digits[0] + digits[1])
                    m = int(digits[3] + digits[4])
                    s = int(digits[6] + digits[7])
                    if 0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60:
                        return f"{h:02}:{m:02}:{s:02}"
                except:
                    pass

            # 🧪 通常の6桁（HHMMSS）補正
            if len(digits) >= 6:
                try:
                    h, m, s = int(digits[:2]), int(digits[2:4]), int(digits[4:6])
                    if 0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60:
                        return f"{h:02}:{m:02}:{s:02}"
                except:
                    pass

            # 🧪 5桁（HMMSS）→ H:MM:SS
            if len(digits) == 5:
                try:
                    h, m, s = int(digits[0]), int(digits[1:3]), int(digits[3:])
                    if 0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60:
                        return f"{h:02}:{m:02}:{s:02}"
                except:
                    pass

            # 🧪 4桁（MMSS）→ 00:MM:SS
            if len(digits) == 4:
                try:
                    m, s = int(digits[:2]), int(digits[2:])
                    if 0 <= m < 60 and 0 <= s < 60:
                        return f"00:{m:02}:{s:02}"
                except:
                    pass

            return "??:??:??"

        # 補正実行
        top_time_corrected = extract_and_correct_base_time(top_txts)
        top_raw_text = "\n".join(top_txts) if top_txts else "(検出なし)"
        center_text = "\n".join(center_txts) if center_txts else "(検出なし)"

        # 予定抽出
        parsed_preview = parse_multiple_places(center_txts, top_txts)
        preview_lines = [f"・{txt}" for _, txt, _ in parsed_preview] if parsed_preview else ["(なし)"]
        preview_text = "\n".join(preview_lines)

        # 免戦時間抽出
        durations = extract_imsen_durations(center_txts)
        duration_text = "\n".join(durations) if durations else "(抽出なし)"

        # 送信
        await message.channel.send(
            f"📸 **上部OCR結果（基準時刻）**:\n```\n{top_raw_text}\n```\n"
            f"🛠️ **補正後の基準時間** → `{top_time_corrected}`\n\n"
            f"📸 **中央OCR結果（サーバー・免戦）**:\n```\n{center_text}\n```\n"
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
        new_txt = f"{mode} {server}-{place}-{new_dt.strftime('%H:%M:%S')}"

        if old_txt in pending_places:
            old_entry = pending_places.pop(old_txt)

            # 通知チャンネル削除
            if "main_msg_id" in old_entry and old_entry["main_msg_id"]:
                ch = client.get_channel(NOTIFY_CHANNEL_ID)
                try:
                    msg = await ch.fetch_message(old_entry["main_msg_id"])
                    await msg.delete()
                except:
                    pass

            # コピー用チャンネル削除
            if "copy_msg_id" in old_entry and old_entry["copy_msg_id"]:
                ch = client.get_channel(COPY_CHANNEL_ID)
                try:
                    msg = await ch.fetch_message(old_entry["copy_msg_id"])
                    await msg.delete()
                except:
                    pass

        # 再登録
        pending_places[new_txt] = {
            "dt": new_dt,
            "txt": new_txt,
            "server": server,
            "created_at": now_jst(),
            "main_msg_id": None,
            "copy_msg_id": None,
        }

        # 通常通知チャンネルに反映（block更新）
        block = find_or_create_block(new_dt)
        block["events"].append((new_dt, new_txt))
        if block["msg"]:
            try:
                await block["msg"].edit(content=format_block_msg(block, True))
                pending_places[new_txt]["main_msg_id"] = block["msg"].id
            except:
                pass
        else:
            task = asyncio.create_task(schedule_block_summary(block, client.get_channel(NOTIFY_CHANNEL_ID)))
            active_tasks.add(task)
            task.add_done_callback(lambda t: active_tasks.discard(t))

        # 奪取なら個別通知（2分/15秒前）、過去ならスキップ
        if new_txt.startswith("奪取") and new_dt > now_jst():
            task2 = asyncio.create_task(schedule_notification(new_dt, new_txt, client.get_channel(NOTIFY_CHANNEL_ID)))
            active_tasks.add(task2)
            task2.add_done_callback(lambda t: active_tasks.discard(t))

        # コピー用チャンネルに再送
        copy_task = asyncio.create_task(send_to_copy_channel(new_dt, new_txt))
        copy_task.add_done_callback(lambda t: store_copy_msg_id(new_txt, t.result()))
        
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
            dt = datetime.combine(now_jst().date(), datetime.strptime(t, "%H:%M:%S").time(), tzinfo=JST)
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

            # 🧪 1つ飛ばし補正（例: "11814822" → "11:14:22"）
            if len(digits) >= 8:
                try:
                    h = int(digits[0] + digits[1])
                    m = int(digits[3] + digits[4])
                    s = int(digits[6] + digits[7])
                    if 0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60:
                        return f"{h:02}:{m:02}:{s:02}"
                except:
                    pass

            # 🧪 通常の6桁（HHMMSS）補正
            if len(digits) >= 6:
                try:
                    h, m, s = int(digits[:2]), int(digits[2:4]), int(digits[4:6])
                    if 0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60:
                        return f"{h:02}:{m:02}:{s:02}"
                except:
                    pass

            # 🧪 5桁（HMMSS）→ H:MM:SS
            if len(digits) == 5:
                try:
                    h, m, s = int(digits[0]), int(digits[1:3]), int(digits[3:])
                    if 0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60:
                        return f"{h:02}:{m:02}:{s:02}"
                except:
                    pass

            # 🧪 4桁（MMSS）→ 00:MM:SS
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

        # ✅ 共通の補正ロジックに置き換え
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
                    "copy_msg_id": None
                }
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
        lines = ["✅ 解析完了！登録されました"]
        for base_time, txts in grouped_results:
            lines.append(f"\n📸 [基準時間: {base_time}]")
            lines += [f"・{txt}" for txt in txts]
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