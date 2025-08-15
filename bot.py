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
PRE_NOTIFY_CHANNEL_ID = int(os.getenv("PRE_NOTIFY_CHANNEL_ID", "0"))
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
# 直近の画像グループ（!gで使う）
last_groups = {}     # {grp_id: [ {mode,server,place,dt,txt,main_msg_id,copy_msg_id}, ... ]}
last_groups_seq = 0  # 採番
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
        # ↓↓↓ 追加：5分前メッセの掃除
        if block.get("msg_5min"):
            try:
                await block["msg_5min"].delete()
            except:
                pass
            block["msg_5min"] = None
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

def parse_txt_fields(txt: str):
    m = re.fullmatch(r"(奪取|警備)\s+(\d{4})-(\d+)-(\d{2}:\d{2}:\d{2})", txt)
    return m.groups() if m else None

async def upsert_copy_channel_sorted(new_entries: list[tuple[datetime, str]]):
    """
    コピー用CHを「編集で」並べ替え＋挿入。
    - 既存のbot投稿をチャンネルの現在順（古い→新しい）で取得
    - desired(理想の順)を dt でソートして作る
    - 既存[i] を desired[i] に edit で置き換え
    - 足りない分は末尾に send
    - pending_places の copy_msg_id を再ひも付け
    """
    ch = client.get_channel(COPY_CHANNEL_ID)
    if not ch:
        return

    # 1) いまチャンネルに出ている「自分の投稿」を古い順に取得
    existing_msgs = []
async def upsert_copy_channel_sorted(new_entries: list[tuple[datetime, str]]):
    """
    コピー用チャンネルを pending_places の内容と完全一致させる。
    - dt昇順で再配置
    - 余分なメッセージは削除
    - 足りない分は新規送信
    - copy_msg_id を再ひも付け
    ※ new_entries は互換のため受け取るが、同期は pending_places 全体を基準にする
    """
    ch = client.get_channel(COPY_CHANNEL_ID)
    if not ch:
        return

    # 1) 既存（自分の投稿のみ）を古い順で回収
    existing_msgs = []
    async for m in ch.history(limit=200, oldest_first=True):
        if m.author == client.user:
            existing_msgs.append(m)

    # 2) 望ましい一覧（pending_places 全体）を dt 昇順で作る
    desired_pairs = sorted(
        [(v["dt"], v["txt"]) for v in pending_places.values()],
        key=lambda x: x[0]
    )
    desired_texts = [txt for _, txt in desired_pairs]

    # 3) 既存と desired を同じ長さに揃える（編集/追加/削除）
    text_to_msgid = {}

    # 3-1) 編集で合わせる
    for i in range(min(len(existing_msgs), len(desired_texts))):
        cur_msg = existing_msgs[i]
        target  = desired_texts[i].replace("🕒 ", "")
        if cur_msg.content != target:
            try:
                await cur_msg.edit(content=target)
            except:
                pass
        text_to_msgid[desired_texts[i]] = cur_msg.id

    # 3-2) 足りないぶんを追加
    if len(desired_texts) > len(existing_msgs):
        for txt in desired_texts[len(existing_msgs):]:
            try:
                m = await ch.send(content=txt.replace("🕒 ", ""))
                text_to_msgid[txt] = m.id
            except:
                pass

    # 3-3) 余っているぶんを削除
    if len(existing_msgs) > len(desired_texts):
        for m in existing_msgs[len(desired_texts):]:
            try:
                await m.delete()
            except:
                pass

    # 4) copy_msg_id を再ひも付け
    for txt, ent in list(pending_places.items()):
        ent["copy_msg_id"] = text_to_msgid.get(txt, None)

async def apply_adjust_for_server_place(server: str, place: str, sec_adj: int):
    # server/place に一致する予定を sec_adj 秒ずらす（早い時間だけ残す・同時刻は統合）
    candidates = []
    for txt, ent in list(pending_places.items()):
        g = parse_txt_fields(txt)
        if g and g[1] == server and g[2] == place:
            candidates.append((txt, ent))
    if not candidates:
        return None

    candidates.sort(key=lambda x: x[1]["dt"])
    old_txt, entry = candidates[0]
    old_dt = entry["dt"]
    mode, _, _, _ = parse_txt_fields(old_txt)
    new_dt = old_dt + timedelta(seconds=sec_adj)
    new_txt = f"{mode} {server}-{place}-{new_dt.strftime('%H:%M:%S')}"

    # 旧通知予約キャンセル
    for key in [(old_txt, "2min"), (old_txt, "15s")]:
        task = sent_notifications_tasks.pop(key, None)
        if task:
            task.cancel()

    # 同時刻が既にある → 統合（新規作成せず old を消す）
    if new_txt in pending_places and new_txt != old_txt:
        old_entry = pending_places.pop(old_txt, None)
        await retime_event_in_summary(old_txt, pending_places[new_txt]["dt"], new_txt, client.get_channel(NOTIFY_CHANNEL_ID))
        try:
            if old_entry and old_entry.get("copy_msg_id"):
                ch_copy = client.get_channel(COPY_CHANNEL_ID)
                if ch_copy:
                    msg = await ch_copy.fetch_message(old_entry["copy_msg_id"])
                    await msg.delete()
        except:
            pass
        notify_ch = client.get_channel(NOTIFY_CHANNEL_ID)
        if notify_ch and mode == "奪取":
            await schedule_notification(pending_places[new_txt]["dt"], new_txt, notify_ch)
        return (old_txt, new_txt)

    # 差し替え
    old_main_id = entry.get("main_msg_id")
    old_copy_id = entry.get("copy_msg_id")
    pending_places.pop(old_txt, None)
    pending_places[new_txt] = {
        "dt": new_dt, "txt": new_txt, "server": server,
        "created_at": entry.get("created_at", now_jst()),
        "main_msg_id": old_main_id, "copy_msg_id": old_copy_id,
    }

    # まとめ編集
    await retime_event_in_summary(old_txt, new_dt, new_txt, client.get_channel(NOTIFY_CHANNEL_ID))

    # コピー編集（自動新規はしない）
    if old_copy_id:
        ch_copy = client.get_channel(COPY_CHANNEL_ID)
        if ch_copy:
            try:
                msg = await ch_copy.fetch_message(old_copy_id)
                await msg.edit(content=new_txt.replace("🕒 ", ""))
                pending_places[new_txt]["copy_msg_id"] = msg.id
            except discord.NotFound:
                pending_places[new_txt]["copy_msg_id"] = None
            except:
                pass

    # 通知再登録（奪取のみ）
    notify_ch = client.get_channel(NOTIFY_CHANNEL_ID)
    if notify_ch and new_txt.startswith("奪取"):
        await schedule_notification(new_dt, new_txt, notify_ch)

    # --- 修正後: 同(server,place)は new_txt だけ残し、他は全削除 ---
    for txt, ent in list(pending_places.items()):
        g2 = parse_txt_fields(txt)
        if g2 and g2[1] == server and g2[2] == place and txt != new_txt:
            # まとめから外す（新行は既に入っている）
            await retime_event_in_summary(txt, new_dt, new_txt, client.get_channel(NOTIFY_CHANNEL_ID))

            # コピー用メッセージがあれば削除
            try:
                if ent.get("copy_msg_id"):
                    ch_copy = client.get_channel(COPY_CHANNEL_ID)
                    if ch_copy:
                        msg = await ch_copy.fetch_message(ent["copy_msg_id"])
                        await msg.delete()
            except:
                pass

            # 旧通知予約キャンセル
            for key in [(txt, "2min"), (txt, "15s")]:
                task = sent_notifications_tasks.pop(key, None)
                if task:
                    task.cancel()

            # pending から除去
            pending_places.pop(txt, None)

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
        "msg_5min": None,
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
        # ① 開始30分前まで待つ
        await asyncio.sleep(max(0, (block["min"] - timedelta(minutes=30) - now_jst()).total_seconds()))

        # 30分前：まとめメッセ（フッター付き）を送信/更新
        content_with_footer = format_block_msg(block, with_footer=True)
        if not block["msg"]:
            block["msg"] = await channel.send(content_with_footer)
        else:
            try:
                await block["msg"].edit(content=content_with_footer)
            except discord.NotFound:
                block["msg"] = await channel.send(content_with_footer)

        # ② 開始5分前まで待つ
        await asyncio.sleep(max(0, (block["min"] - timedelta(minutes=5) - now_jst()).total_seconds()))

        # 5分前：短い別メッセージを送る（単独）
        try:
            block["msg_5min"] = await channel.send("⚠️ 5分後に始まるよ⚠️")
        except Exception:
            block["msg_5min"] = None

        # ③ 開始時刻まで待つ
        await asyncio.sleep(max(0, (block["min"] - now_jst()).total_seconds()))

        # 開始時刻：まとめメッセのフッターだけ消す（本文は残す）
        if block["msg"]:
            try:
                await block["msg"].edit(content=format_block_msg(block, with_footer=False))
            except discord.NotFound:
                pass

        # 開始時刻：5分前メッセージは削除
        if block.get("msg_5min"):
            try:
                await block["msg_5min"].delete()
            except Exception:
                pass
            finally:
                block["msg_5min"] = None

    except Exception as e:
        print(f"[ERROR] schedule_block_summary failed: {e}")
    finally:
        # タスク参照をクリア（多重起動防止）
        block["task"] = None

async def minus_one_for_places(place_ids: list[str]):
    """
    指定した『駐機場番号』の予定すべてを -1秒。
    反映:
      - pending_places の dt/txt を更新（キー差し替え）
      - summary_blocks の events 更新 & まとめメッセージ .edit()
      - 既存のコピー用メッセージがあれば .edit()
      - 通知予約（2分前/15秒前）→ 一旦キャンセルして再登録（奪取のみ）
    戻り値: 更新後テキストのリスト
    """
    targets = set(str(p) for p in place_ids)
    updated = []

    notify_ch = client.get_channel(NOTIFY_CHANNEL_ID)
    copy_ch   = client.get_channel(COPY_CHANNEL_ID)

    for old_key, ent in list(pending_places.items()):
        m = re.fullmatch(r"(奪取|警備)\s+(\d{4})-(\d+)-(\d{2}:\d{2}:\d{2})", ent["txt"])
        if not m:
            continue
        mode, server, place, timestr = m.groups()
        if place not in targets:
            continue

        old_txt = ent["txt"]
        old_dt  = ent["dt"]
        new_dt  = old_dt - timedelta(seconds=1)
        new_txt = f"{mode} {server}-{place}-{new_dt.strftime('%H:%M:%S')}"

        # 旧通知予約キャンセル
        for key in [(old_txt, "2min"), (old_txt, "15s")]:
            task = sent_notifications_tasks.pop(key, None)
            if task:
                task.cancel()

        # pending_places 差し替え（キー変更）
        entry = pending_places.pop(old_txt)
        entry["dt"]  = new_dt
        entry["txt"] = new_txt
        pending_places[new_txt] = entry

        # summary_blocks 更新
        for block in summary_blocks:
            changed = False
            for i, (d, t) in enumerate(list(block["events"])):
                if t == old_txt:
                    block["events"][i] = (new_dt, new_txt)
                    changed = True
            if changed:
                if block["events"]:
                    block["min"] = min(ev[0] for ev in block["events"])
                    block["max"] = max(ev[0] for ev in block["events"])
                else:
                    block["min"] = block["max"] = new_dt
                if block.get("msg"):
                    try:
                        await block["msg"].edit(content=format_block_msg(block, True))
                        pending_places[new_txt]["main_msg_id"] = block["msg"].id
                    except:
                        pass

        # コピー用メッセージがあれば .edit()
        copy_id = entry.get("copy_msg_id")
        if copy_id and copy_ch:
            try:
                msg = await copy_ch.fetch_message(copy_id)
                await msg.edit(content=new_txt.replace("🕒 ", ""))
            except:
                pass

        # 通知予約を再登録（奪取のみ）
        if notify_ch and mode == "奪取":
            await schedule_notification(new_dt, new_txt, notify_ch)

        updated.append(new_txt)

    return updated

async def retime_event_in_summary(old_txt: str, new_dt: datetime, new_txt: str, channel):
    """
    通知チャンネルのまとめメッセージを編集で更新する:
      - 古い行(old_txt)は全ブロックから削除
      - 新しい行(new_txt)を該当ブロックへ追加
      - すべて時間順に整列
    """
    # 1) 古い行を全ブロックから除去
    for block in list(summary_blocks):
        if not block.get("events"):
            continue
        block["events"] = [(d, t) for (d, t) in block["events"] if t != old_txt]
        if block["events"]:
            block["events"].sort(key=lambda x: x[0])
            block["min"] = min(d for d, _ in block["events"])
            block["max"] = max(d for d, _ in block["events"])
        else:
            # 空になったブロックは min/max を新DTに仮置き
            block["min"] = block["max"] = new_dt

    # 2) 新しい行を入れる（なければブロック作成）
    target_block = find_or_create_block(new_dt)
    if (new_dt, new_txt) not in target_block["events"]:
        target_block["events"].append((new_dt, new_txt))
    target_block["events"].sort(key=lambda x: x[0])
    target_block["min"] = min(target_block["min"], new_dt)
    target_block["max"] = max(target_block["max"], new_dt)

    # 3) まとめメッセージを一括編集
    for block in summary_blocks:
        if block.get("events"):
            block["events"].sort(key=lambda x: x[0])
        if block.get("msg"):
            try:
                await block["msg"].edit(content=format_block_msg(block, True))
            except:
                pass

    # 4) pending_places の main_msg_id を新しいテキストにひも付け（ブロックにメッセージがある場合）
    if target_block.get("msg") and new_txt in pending_places:
        pending_places[new_txt]["main_msg_id"] = target_block["msg"].id

# === 手動まとめ(!s)メッセージを最新状態で上書き ===
async def refresh_manual_summaries():
    if not manual_summary_msg_ids:
        return
    ch = client.get_channel(NOTIFY_CHANNEL_ID)
    if not ch:
        return

    # pending_places を時刻順に並べて、!s と同じレイアウトで作り直す
    sorted_places = sorted(pending_places.values(), key=lambda x: x["dt"])
    lines = ["📢 手動通知: 現在登録されているスケジュール一覧", ""]
    for v in sorted_places:
        lines.append(v["txt"])
    new_content = "\n".join(lines)

    # 既に送ってある手動まとめメッセージを全部「編集」で上書き
    for mid in list(manual_summary_msg_ids):
        try:
            msg = await ch.fetch_message(mid)
            await msg.edit(content=new_content)
        except:
            # （消されてる等の例外は無視）
            pass

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
        t_15s  = unlock_dt - timedelta(seconds=15)

        # 送信先：事前通知専用があればそちら、無ければ従来の通知チャンネル
        pre_ch = client.get_channel(PRE_NOTIFY_CHANNEL_ID) or channel

        async def notify_2min():
            if t_2min > now and (text, "2min") not in sent_notifications and not is_within_5_minutes_of_another(unlock_dt):
                sent_notifications.add((text, "2min"))
                await asyncio.sleep((t_2min - now_jst()).total_seconds())
                try:
                    msg = await pre_ch.send(f"⏰ {text} **2分前です！！**")
                    await asyncio.sleep(120)
                    await msg.delete()
                except Exception:
                    pass

        async def notify_15s():
            if t_15s > now and (text, "15s") not in sent_notifications:
                sent_notifications.add((text, "15s"))
                await asyncio.sleep((t_15s - now_jst()).total_seconds())
                try:
                    msg = await pre_ch.send(f"⏰ {text} **15秒前です！！**")
                    await asyncio.sleep(120)
                    await msg.delete()
                except Exception:
                    pass

        sent_notifications_tasks[(text, "2min")] = asyncio.create_task(notify_2min())
        sent_notifications_tasks[(text, "15s")]  = asyncio.create_task(notify_15s())

async def process_copy_queue():
    while True:
        await asyncio.sleep(30)
        if pending_copy_queue:
            # 溜まった分を時刻順にして一括反映
            batch = sorted(pending_copy_queue, key=lambda x: x[0])  # [(dt, txt), ...]
            pending_copy_queue.clear()
            await upsert_copy_channel_sorted(batch)
        await asyncio.sleep(2)   # ポーリング間隔を短く
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
            if block.get("msg_5min"):        # ← 追加
                try:
                    await block["msg_5min"].delete()
                except:
                    pass
                block["msg_5min"] = None
        await purge_my_messages(PRE_NOTIFY_CHANNEL_ID, limit=200)

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

# --- 追加: 自分のメッセージだけを一括削除するヘルパー ---
async def purge_my_messages(channel_id: int, limit: int = 200):
    if not channel_id:
        return
    ch = client.get_channel(channel_id)
    if not ch:
        return
    try:
        async for m in ch.history(limit=limit):
            if m.author == client.user:
                try:
                    await m.delete()
                except:
                    pass
    except:
        pass

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
    # 予定ごとの個別メッセージ削除（通知/コピー）
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

    # まとめメッセージと「5分前メッセージ」を確実に削除
    for block in list(summary_blocks):
        # ← これが重要（5分前通知の消し忘れ対策）
        if block.get("msg_5min"):
            try:
                await block["msg_5min"].delete()
            except:
                pass
            block["msg_5min"] = None

        if block.get("msg"):
            try:
                await block["msg"].delete()
            except:
                pass
    summary_blocks.clear()

    # 手動まとめ(!s)の削除
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

    # 通知予約タスク（2分前/15秒前）のキャンセル＆一覧クリア
    for key, task in list(sent_notifications_tasks.items()):
        task.cancel()
    sent_notifications_tasks.clear()
    sent_notifications.clear()

    # コピー用チャンネルを軽く掃除（保険）
    await purge_my_messages(COPY_CHANNEL_ID, limit=100)

    # ✅ 事前通知チャンネル（2分前/15秒前など）を掃除 ← これが重要
    await purge_my_messages(PRE_NOTIFY_CHANNEL_ID, limit=200)

    # 状態クリア
    pending_places.clear()

    # 他タスクも停止
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
    global last_groups_seq, last_groups
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

        # pending_places から削除（通知チャンネルのまとめメッセは消さない）
        if txt in pending_places:
            entry = pending_places.pop(txt)
            removed = True

            # コピー用チャンネルの該当メッセージだけ削除
            if entry.get("copy_msg_id"):
                ch = client.get_channel(COPY_CHANNEL_ID)
                try:
                    msg = await ch.fetch_message(entry["copy_msg_id"])
                    await msg.delete()
                except:
                    pass

        # summary_blocks から該当行だけ削除し、メッセージは編集で更新
        for block in summary_blocks:
            before = len(block["events"])
            block["events"] = [ev for ev in block["events"] if ev[1] != txt]
            after = len(block["events"])

            if before != after:
                removed = True
                if block["events"]:
                    block["min"] = min(ev[0] for ev in block["events"])
                    block["max"] = max(ev[0] for ev in block["events"])
                else:
                    block["min"] = block["max"] = datetime.max.replace(tzinfo=JST)

                if block.get("msg"):
                    try:
                        await block["msg"].edit(content=format_block_msg(block, True))
                    except:
                        pass

        # 通知予約も確実にキャンセル（!n からも消える）
        for key in [(txt, "2min"), (txt, "15s")]:
            task = sent_notifications_tasks.pop(key, None)
            if task and not task.cancelled():
                task.cancel()

        if removed:
            # 手動まとめ(!s)が既に送られている場合は、編集で最新化
            await refresh_manual_summaries()
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

    # ==== !1 駐騎場ナンバーで一括 -1 秒 ====
    # 例) "!1 1 12 11" → place が 1,12,11 の予定をそれぞれ -1 秒
    if message.content.strip().startswith("!1"):
        parts = message.content.strip().split()
        if len(parts) < 2:
            await message.channel.send("⚠️ 使い方: `!1 <駐騎場> <駐騎場> ...` 例: `!1 1 12 11`")
            return

        target_places = set(parts[1:])  # 文字列のまま比較（txt内の place は数字文字列）
        if not pending_places:
            await message.channel.send("⚠️ 登録された予定がありません")
            return

        updated = []  # (old_txt, new_txt) for レポート

        # txt のキーが変わるので、走査用に元のキー一覧を固定
        original_items = list(pending_places.items())

        for old_txt, entry in original_items:
            # 形式: "<モード> <server>-<place>-<HH:MM:SS>"
            m = re.fullmatch(r"(奪取|警備)\s+(\d{4})-(\d+)-(\d{2}:\d{2}:\d{2})", old_txt)
            if not m:
                continue
            mode, server, place, hhmmss = m.groups()
            if place not in target_places:
                continue  # 対象外の駐騎場

            old_dt = entry["dt"]
            new_dt = old_dt - timedelta(seconds=1)
            # 深夜帯基準の補正は不要（相対 -1 秒のみ）
            new_txt = f"{mode} {server}-{place}-{new_dt.strftime('%H:%M:%S')}"

            # 通知予約のキャンセル（旧txt名）
            for key in [(old_txt, "2min"), (old_txt, "15s")]:
                task = sent_notifications_tasks.pop(key, None)
                if task:
                    task.cancel()

            # pending_places のキー更新
            old_main_id = entry.get("main_msg_id")
            old_copy_id = entry.get("copy_msg_id")
            pending_places.pop(old_txt, None)
            pending_places[new_txt] = {
                "dt": new_dt,
                "txt": new_txt,
                "server": server,
                "created_at": entry.get("created_at", now_jst()),
                "main_msg_id": old_main_id,
                "copy_msg_id": old_copy_id,
            }

            # まとめメッセージ（ブロック）側：古い行を外して新行を追加して整形
            for block in summary_blocks:
                # 古い行を除去
                before = len(block["events"])
                block["events"] = [(d, t) for (d, t) in block["events"] if t != old_txt]
                # 新行を追加（同一ブロックかどうかは時刻レンジで許容）
                if new_dt <= block["max"] + timedelta(minutes=45):
                    block["events"].append((new_dt, new_txt))
                    block["min"] = min(block["min"], new_dt) if before else new_dt
                    block["max"] = max(block["max"], new_dt) if before else new_dt
                    # 時刻順ソート
                    block["events"].sort(key=lambda x: x[0])
                    # メッセージ編集
                    if block.get("msg"):
                        try:
                            await block["msg"].edit(content=format_block_msg(block, True))
                            pending_places[new_txt]["main_msg_id"] = block["msg"].id
                        except:
                            pass
                    break
            else:
                # 同一ブロックが無かった場合は、新規ブロックに追加（メッセ送信は自動で行わない）
                nb = find_or_create_block(new_dt)
                nb["events"].append((new_dt, new_txt))
                nb["min"] = min(nb["min"], new_dt)
                nb["max"] = max(nb["max"], new_dt)

            # コピーチャンネル：既存メッセがあれば編集で更新（新規送信はしない）
            if old_copy_id:
                ch_copy = client.get_channel(COPY_CHANNEL_ID)
                if ch_copy:
                    try:
                        msg = await ch_copy.fetch_message(old_copy_id)
                        await msg.edit(content=new_txt.replace("🕒 ", ""))
                        pending_places[new_txt]["copy_msg_id"] = msg.id
                    except discord.NotFound:
                        pending_places[new_txt]["copy_msg_id"] = None
                    except:
                        pass

            # 通知予約を新時刻で再登録
            notify_ch = client.get_channel(NOTIFY_CHANNEL_ID)
            if notify_ch and new_txt.startswith("奪取"):
                await schedule_notification(new_dt, new_txt, notify_ch)

            updated.append((old_txt, new_txt))

        # 手動まとめ(!s)が既に送られていれば、**ここで編集で最新化**
        await refresh_manual_summaries()
        batch = [(pending_places[n]["dt"], n) for _, n in updated if n in pending_places]
        if batch:
            await upsert_copy_channel_sorted(batch)
        if not updated:
            await message.channel.send("⚠️ 対象の駐騎場の予定が見つかりませんでした")
        else:
            lines = ["✅ -1秒の適用が完了しました", ""]
            for o, n in updated:
                lines.append(f"・{o} → {n}")
            await message.channel.send("\n".join(lines))
        return

    # ==== !g 画像グループ単位で±秒オフセット ====
    m_g = re.fullmatch(r"!g\s+(.+)", message.content.strip())
    if m_g:
        arg_str = m_g.group(1).strip()
        tokens = arg_str.split()

        if not last_groups:
            await message.channel.send("⚠️ 対象グループがありません。まず画像を送って解析してください。")
            return

        group_adjust_map = {}

        # パターンC: "!g 1" → デフォルト -1秒
        if len(tokens) == 1 and re.fullmatch(r"\d+", tokens[0]):
            gid = int(tokens[0])
            if gid in last_groups:
                group_adjust_map[gid] = -1

        # パターンA: "<grp> <grp> ... <±sec>"
        elif len(tokens) >= 2 and all(re.fullmatch(r"\d+", t) for t in tokens[:-1]) and re.fullmatch(r"[-+]?\d+", tokens[-1]):
            common_adj = int(tokens[-1])
            for gid_str in tokens[:-1]:
                gid = int(gid_str)
                if gid in last_groups:
                    group_adjust_map[gid] = common_adj

        # パターンB: "<grp>:<±sec> ..."
        else:
            ok = True
            for t in tokens:
                m2 = re.fullmatch(r"(\d+):([-+]?\d+)", t)
                if not m2:
                    ok = False
                    break
                gid = int(m2.group(1)); sec = int(m2.group(2))
                if gid in last_groups:
                    group_adjust_map[gid] = sec
            if not ok and not group_adjust_map:
                await message.channel.send("⚠️ 使い方: `!g <grp> <grp> ... <±sec>` または `!g <grp>:<±sec>` または `!g <grp>`")
                return

        if not group_adjust_map:
            await message.channel.send("⚠️ 指定グループが見つかりません")
            return

        updated_pairs = []
        skipped = 0
        for gid, sec_adj in group_adjust_map.items():
            for e in last_groups.get(gid, []):
                res = await apply_adjust_for_server_place(e["server"], e["place"], sec_adj)
                if res: updated_pairs.append(res)
                else: skipped += 1

        await refresh_manual_summaries()
        batch = [(pending_places[n]["dt"], n) for _, n in updated_pairs if n in pending_places]
        if batch:
            await upsert_copy_channel_sorted(batch)
        if updated_pairs:
            msg = ["✅ 反映しました:"]
            msg += [f"　{o} → {n}" for o, n in updated_pairs]
            if skipped: msg.append(f"ℹ️ 一部スキップ: {skipped}件")
            await message.channel.send("\n".join(msg))
        else:
            await message.channel.send("（変更なし）")
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
    # ==== !glist 現在のグループ一覧表示 ====
    if message.content.strip() == "!glist":
        if not last_groups:
            await message.channel.send("⚠️ 現在グループはありません。まず画像を送って解析してください。")
            return
        lines = ["📸 現在の画像グループ:"]
        for gid, events in last_groups.items():
            lines.append(f"　G{gid}:")
            for e in events:
                lines.append(f"　　・{e['server']}-{e['place']}-{e['dt'].strftime('%H:%M:%S')}")
        await message.channel.send("\n".join(lines))
        return
        
    # ==== !a 奪取 1234-1-12:00:00 130000 or 13:00:00 ====
    match = re.fullmatch(
        r"!a\s+(奪取|警備)\s+(\d{4})-(\d+)-(\d{2}:\d{2}:\d{2})\s+(\d{6}|\d{1,2}:\d{2}:\d{2})",
        message.content.strip()
    )
    if match:
        mode, server, place, timestr, raw = match.groups()
        old_txt = f"{mode} {server}-{place}-{timestr}"

        # ---- 入力時刻のパース ----
        try:
            if ":" in raw:
                h, m, s = map(int, raw.split(":"))
            else:
                h, m, s = int(raw[:2]), int(raw[2:4]), int(raw[4:])
        except:
            await message.channel.send("⚠️ 時間の指定が不正です")
            return

        # ---- 新日時の決定（過去扱い防止）----
        if old_txt in pending_places:
            base_date = pending_places[old_txt]["dt"].date()
        else:
            base_date = now_jst().date()

        new_time = time(h, m, s)
        new_dt = datetime.combine(base_date, new_time, tzinfo=JST)

        # 00:00〜05:59 は翌日扱い
        if new_time < time(6, 0, 0):
            new_dt += timedelta(days=1)
        # それでも現在以下なら翌日に繰上げ
        if new_dt <= now_jst():
            new_dt += timedelta(days=1)

        new_txt = f"{mode} {server}-{place}-{new_dt.strftime('%H:%M:%S')}"

        # ---- 旧エントリ情報の回収 & 通知予約キャンセル ----
        old_main_msg_id = None
        old_copy_msg_id = None
        if old_txt in pending_places:
            old_entry = pending_places.pop(old_txt)
            old_main_msg_id = old_entry.get("main_msg_id")
            old_copy_msg_id = old_entry.get("copy_msg_id")

            # 旧通知予約キャンセル（!n からも消える）
            for key in [(old_txt, "2min"), (old_txt, "15s")]:
                task = sent_notifications_tasks.pop(key, None)
                if task:
                    task.cancel()

        # ---- 新エントリを登録（ID は引き継ぐ、無ければ None のまま）----
        pending_places[new_txt] = {
            "dt": new_dt,
            "txt": new_txt,
            "server": server,
            "created_at": now_jst(),
            "main_msg_id": old_main_msg_id,  # まとめメッセのID（retime側で再設定される）
            "copy_msg_id": old_copy_msg_id,  # コピー用メッセのID（あれば編集する）
        }

        # ---- 通知チャンネルのまとめメッセ：古い行を削除→新行を時刻順に追加（削除はせず編集で更新） ----
        await retime_event_in_summary(old_txt, new_dt, new_txt, channel)

        # ---- コピーチャンネル：存在する場合のみ「編集」。無ければ何もしない（自動新規送信しない）----
        if old_copy_msg_id:
            copy_ch = client.get_channel(COPY_CHANNEL_ID)
            if copy_ch:
                try:
                    msg = await copy_ch.fetch_message(old_copy_msg_id)
                    await msg.edit(content=new_txt.replace("🕒 ", ""))
                except discord.NotFound:
                    pending_places[new_txt]["copy_msg_id"] = None
                except Exception:
                    pass

        # ---- 通知予約を新しい時間で再登録（!n に反映）----
        notify_ch = client.get_channel(NOTIFY_CHANNEL_ID)
        if notify_ch:
            await schedule_notification(new_dt, new_txt, notify_ch)

            # 手動まとめ(!s)が既に送られている場合は、編集で最新化
            await refresh_manual_summaries()
        await upsert_copy_channel_sorted([(new_dt, new_txt)])
        await message.channel.send(f"✅ 更新しました → `{new_txt}`")
        return

    # ==== 手動追加（例: 1234-1-12:34:56）====
    manual = re.findall(r"\b(\d{3,4})-(\d+)-(\d{2}:\d{2}:\d{2})\b", message.content)
    if manual:
        entries_to_copy = []  # ← コピー用チャンネルへ流す分を一括で持つ
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
                # ← コピー用チャンネルへも反映（後で一括で時刻順差し込み）
                entries_to_copy.append((dt, txt))
        # ← 複数件をまとめてコピー用チャンネルに時刻順で差し込み
        if entries_to_copy:
            await upsert_copy_channel_sorted(entries_to_copy)
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
            structured_entries_for_this_image = []  # ← !g用
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
                g = parse_txt_fields(txt)
                if g:
                    _mode, _server, _place, _ = g
                    structured_entries_for_this_image.append({
                        "mode": _mode, "server": _server, "place": _place,
                        "dt": dt, "txt": txt,
                        "main_msg_id": pending_places.get(txt, {}).get("main_msg_id"),
                        "copy_msg_id": pending_places.get(txt, {}).get("copy_msg_id"),
                    })
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
                    pending_copy_queue.append((dt, txt))
                    display_txt = f"{txt} ({raw})"
                    image_results.append(display_txt)
                    task = asyncio.create_task(handle_new_event(dt, txt, channel))
                    active_tasks.add(task)
                    task.add_done_callback(lambda t: active_tasks.discard(t))
                    if txt.startswith("奪取"):
                        task2 = asyncio.create_task(schedule_notification(dt, txt, channel))
                        active_tasks.add(task2)
                        task2.add_done_callback(lambda t: active_tasks.discard(t))

            if structured_entries_for_this_image:
                last_groups_seq += 1
                gid = last_groups_seq
                last_groups[gid] = structured_entries_for_this_image
                if image_results:
                    grouped_results.append((gid, base_time, image_results))
        if grouped_results:
            lines = [
                "✅ 解析完了！登録されました",
                "",
                "　📤 !c → ⏰ 時間コピー用チャンネルにスケジュールを送信",
                "　📢 !s → 📝 通知チャンネルに手動でスケジュールを通知",
                "　⏪ !1 → 📝 駐騎場ナンバーで-1秒可 ※実際と異なっている場合",
                "　🛠 !g → 画像グループをまとめて±秒 ",
                "",
            ]
            for gid, base_time_str, txts in grouped_results:
                lines.append(f"📸 [G{gid} | 基準時間: {base_time_str}]")
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