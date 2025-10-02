import os
import io
import re
import base64
import asyncio
import unicodedata
from typing import List, Tuple, Dict, Optional, Set

import discord
from discord.ext import commands, tasks
from PIL import Image, ImageDraw, ImageOps

# Google Vision
from google.cloud import vision

# OpenAI (official SDK v1)
from openai import OpenAI

# 時刻スケジュール用
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# ---------------------------
# ENV/bootstrap
# ---------------------------
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GOOGLE_CLOUD_VISION_JSON = os.environ.get("GOOGLE_CLOUD_VISION_JSON", "")

# 自動処理する送信専用チャンネル（カンマ区切りで複数OK）
INPUT_CHANNEL_IDS = {
    int(x) for x in os.environ.get("INPUT_CHANNEL_IDS", "").split(",") if x.strip().isdigit()
}
# 通知（一覧＋開始時刻ボード）チャンネル
NOTIFY_CHANNEL_ID = int(os.environ.get("NOTIFY_CHANNEL_ID", "0") or 0)
# コピー専用チャンネル（即時通知／時間が過ぎたら削除）
COPY_CHANNEL_ID = int(os.environ.get("COPY_CHANNEL_ID", "0") or 0)
# アラート専用チャンネル（2分前/15秒前、5秒後に削除）
ALERT_CHANNEL_ID = int(os.environ.get("ALERT_CHANNEL_ID", "0") or 0)
# ⏰ アラートでメンションするロールID（カンマ区切り）
ALERT_ROLE_IDS = {
    int(x) for x in os.environ.get("ALERT_ROLE_IDS", "").split(",") if x.strip().isdigit()
}
# タイムゾーン（例: Asia/Tokyo）
TIMEZONE = ZoneInfo(os.environ.get("TIMEZONE", "Asia/Tokyo"))

if not DISCORD_TOKEN or not OPENAI_API_KEY:
    raise RuntimeError("DISCORD_TOKEN / OPENAI_API_KEY が未設定です。")

# Google Vision 認証（JSON文字列→一時ファイル）
if GOOGLE_CLOUD_VISION_JSON:
    cred_path = "/tmp/gcv_credentials.json"
    with open(cred_path, "w", encoding="utf-8") as f:
        f.write(GOOGLE_CLOUD_VISION_JSON)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path

gcv_client = vision.ImageAnnotatorClient()
oai_client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------
# Discord Bot
# ---------------------------
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)
ALLOWED_ALERT_MENTIONS = discord.AllowedMentions(everyone=False, users=False, roles=True)

# ---------------------------
# Parameters / Config
# ---------------------------

# 入力画像は最初に横幅708へ等比リサイズ
TARGET_WIDTH = 708

# スライス境界（%）
CUTS = [7.51, 11.63, 25.21, 29.85, 33.62, 38.41, 61.90]  # 上からの境界％
# 残すブロック番号（1始まり）
KEEP = [2, 4, 6, 7]

# 横方向トリミング（左/右を％でカット）
TRIM_RULES = {
    7: (20.0, 50.0),   # 駐騎ナンバー＋免戦時間
    6: (32.48, 51.50), # サーバー番号
    4: (44.0, 20.19),  # 停戦終了
    2: (75.98, 10.73), # 時計
}

# ---------------------------
# 正規表現 / ルール
# ---------------------------

# 「免戦中」厳密
RE_IMMUNE = re.compile(r"免\s*戦\s*中")
# 「免 / 戦 / 戰 / 中」いずれか1文字でも含めば候補（誤検知抑制のため時刻併記も必須にする）
RE_IMMUNE_LOOSE = re.compile(r"[免戦戰中]")

# タイトル系（圧縮用=形）
RE_TITLE_COMPACT = re.compile(r"(?:越\s*域|戦\s*闘)\s*駐[\u4E00-\u9FFF]{0,3}\s*場")

# 時刻検出
RE_TIME_STRICT = re.compile(r"\d{1,2}[：:]\d{2}(?:[：:]\d{2})?")
# 緩め（区切りに . ・ / などや空白を許容。分/秒が1桁でもOK）
RE_TIME_LOOSE  = re.compile(
    r"\d{1,2}\s*[：:\.\-・／/]\s*\d{1,2}(?:\s*[：:\.\-・／/]\s*\d{1,2})?"
)

# OCRでサーバー拾う用（[S1234], S1234, s1234 など）
RE_SERVER = re.compile(r"\[?\s*[sS]\s*([0-9]{2,5})\]?")

# 免戦行の先頭に「数字 免...」が来るパターン用（例: "1 免戦中 05:27:35"）
RE_BARE_PLACE_BEFORE_IMMUNE = re.compile(r"^\s*(\d{1,3})\D*免")

# コピーCHの自動行パターン（例: "1234-5-17:00:00"）※孤児クリーンアップで使用
RE_COPY_LINE = re.compile(r"^\s*\d{2,5}-\d{1,3}-\d{2}:\d{2}:\d{2}\s*$")

# フォールバック：ブロック高さに対する“必ず残す”上部割合
FALLBACK_KEEP_TOP_RATIO = 0.35

def _has_time_like(s: str) -> bool:
    """行に“時刻っぽい”表記があるか（緩め判定）"""
    s = unicodedata.normalize("NFKC", s)
    return bool(RE_TIME_STRICT.search(s) or RE_TIME_LOOSE.search(s))

def _extract_time_like(s: str) -> Optional[str]:
    """
    行から時刻らしきものを1つ抽出（厳密→緩めの順）。
    抽出したら区切りを : に統一し、MM/SS が1桁なら0埋めして返す。
    例）"7:5" -> "7:05", "7・5・3" -> "7:05:03"
    """
    if not s:
        return None
    s = unicodedata.normalize("NFKC", s)
    # 区切りを : に寄せる
    s = re.sub(r"[．。·•･・／/]", ":", s)
    s = re.sub(r"\s+", "", s)

    m = RE_TIME_STRICT.search(s)
    if not m:
        m = RE_TIME_LOOSE.search(s)
    if not m:
        return None

    raw = m.group(0)
    raw = re.sub(r"[．。·•･・／/]", ":", raw)
    raw = re.sub(r"\s+", "", raw)

    parts = re.split(r"[：:]", raw)
    if len(parts) < 2 or len(parts) > 3:
        return None

    a = parts[0]
    b = parts[1].zfill(2)
    if len(parts) == 3:
        c = parts[2].zfill(2)
        return f"{a}:{b}:{c}"
    return f"{a}:{b}"

def _is_immune_line(s: str) -> bool:
    """
    その行が「免戦中系」を指しているかの緩め判定：
    - 厳密マッチ（免\s*戦\s*中）or
    - 1文字でも含む & 行内に時刻っぽい表記がある
    """
    n = unicodedata.normalize("NFKC", s)
    return bool(RE_IMMUNE.search(n) or (RE_IMMUNE_LOOSE.search(n) and _has_time_like(n)))

def _extract_place(line: str) -> Optional[int]:
    """
    タイトル行から駐騎場ナンバーだけを抽出。
    条件:
      - 行に「場/场」と、かつ「越/域/駐/驻/戦/戰/闘」のいずれかを含む
      - 「免戦行」でも許容（同一行に '...場 4 免戦中 ...' などがあるケース対応）
      - 「場」の直後にある 1-3 桁の数字のみ採用（行末数字のフォールバックはしない）
    """
    s = unicodedata.normalize("NFKC", line)
    if not re.search(r"[场場]", s):
        return None
    if not re.search(r"[越域駐驻戦戰闘]", s):
        return None
    m = re.search(r"[场場]\s*([0-9]{1,3})\b", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None

# --------------- チャンネル取得（デバッグ出力込み） ---------------
async def _get_text_channel(cid: int) -> Optional[discord.TextChannel]:
    if not cid:
        return None
    ch = bot.get_channel(cid)
    if ch is None:
        try:
            ch = await bot.fetch_channel(cid)  # type: ignore
        except Exception:
            return None
    if not isinstance(ch, discord.TextChannel):
        return None
    # 送信権限チェック
    me = ch.guild.me
    if me is None:
        return None
    perms = ch.permissions_for(me)
    if not perms.send_messages:
        return None
    return ch

# ---------------------------
# スケジューラ（一覧ボード＋アラート＋コピーメッセ削除）
# ---------------------------

SCHEDULE_LOCK = asyncio.Lock()
COPY_LOCK = asyncio.Lock()  # コピーCHの並べ替え挿入を直列化

# item: {
#   "when": datetime, "server": str, "place": int, "timestr": "HH:MM:SS",
#   "key": (server, place, timestr),
#   "skip2m": bool, "sent_2m": bool, "sent_15s": bool,
#   "copy_msg_id": Optional[int]
# }
SCHEDULE: List[Dict] = []

SCHEDULE_MSG_ID: Optional[int] = None  # 通知チャンネルの一覧メッセージ

def _next_occurrence_today_or_tomorrow(hms: str) -> datetime:
    """今日のその時刻、過ぎていれば翌日の同時刻（TZ考慮）"""
    now = datetime.now(TIMEZONE)
    hh, mm, ss = map(int, hms.split(":"))
    candidate = now.replace(hour=hh, minute=mm, second=ss, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate

def _render_schedule_board() -> str:
    """
    通知チャンネルのスケジュール表示（SCHEDULEは常にwhen昇順）:
    """
    header = "🗓️ 今後のスケジュール🗓️"
    if not SCHEDULE:
        return f"{header}\n🈳 登録された予定はありません"
    lines = []
    for item in SCHEDULE:
        t = item["when"].astimezone(TIMEZONE).strftime("%H:%M:%S")
        lines.append(f"・{item['server']}-{item['place']}-{t}")
    return f"{header}\n" + "\n".join(lines)

async def _ensure_schedule_message(channel: discord.TextChannel) -> None:
    """一覧の固定メッセージを作成/更新"""
    global SCHEDULE_MSG_ID
    content = _render_schedule_board()
    if SCHEDULE_MSG_ID is None:
        msg = await channel.send(content)
        SCHEDULE_MSG_ID = msg.id
    else:
        try:
            msg = await channel.fetch_message(SCHEDULE_MSG_ID)
            await msg.edit(content=content)
        except discord.NotFound:
            msg = await channel.send(content)
            SCHEDULE_MSG_ID = msg.id

def _recompute_skip2m_flags() -> None:
    """次の予定が5分以内ならこの予定の2分前通知を抑制する"""
    for i, it in enumerate(SCHEDULE):
        it["skip2m"] = False
        if i + 1 < len(SCHEDULE):
            nxt = SCHEDULE[i + 1]
            if (nxt["when"] - it["when"]) <= timedelta(minutes=5):
                it["skip2m"] = True

async def _refresh_board():
    """通知用のボードを最新化"""
    if NOTIFY_CHANNEL_ID:
        ch = await _get_text_channel(NOTIFY_CHANNEL_ID)
        if ch:
            await _ensure_schedule_message(ch)

def _alert_prefix() -> str:
    if not ALERT_ROLE_IDS:
        return ""
    return " ".join(f"<@&{rid}>" for rid in ALERT_ROLE_IDS)

async def _send_temp_alert(channel: discord.TextChannel, text: str):
    """アラート送信→5秒後に削除（指定ロールのみメンション）"""
    try:
        prefix = _alert_prefix()
        txt = f"{prefix} {text}".strip()
        msg = await channel.send(txt, allowed_mentions=ALLOWED_ALERT_MENTIONS)
        async def _deleter():
            try:
                await asyncio.sleep(5)
                await msg.delete()
            except Exception:
                pass
        asyncio.create_task(_deleter())
    except Exception as e:
        print(f"[alert] send/delete failed: {e}")

async def _delete_copy_message_if_exists(it: Dict):
    """コピー専用チャンネルの個別メッセージを削除"""
    if not COPY_CHANNEL_ID:
        return
    mid = it.get("copy_msg_id")
    if not mid:
        return
    try:
        ch = await _get_text_channel(COPY_CHANNEL_ID)
        if ch:
            msg = await ch.fetch_message(mid)
            await msg.delete()
    except Exception as e:
        print(f"[copy] delete failed: {e}")
    finally:
        it["copy_msg_id"] = None

# ---- ここから：コピーCHの安全な“割り込み挿入”ロジック ----

def _fmt_copy_line(it: Dict) -> str:
    return f"{it['server']}-{it['place']}-{it['timestr']}"

async def _insert_copy_sorted(new_items: List[Dict]):
    """
    コピーCHメッセージ列に new_items を**時間順**で割り込み挿入。
    """
    if not COPY_CHANNEL_ID or not new_items:
        return
    ch = await _get_text_channel(COPY_CHANNEL_ID)
    if not ch:
        return

    # new_items 自体を時間順に
    new_items = sorted(new_items, key=lambda x: x["when"])

    async with COPY_LOCK:
        for it in new_items:
            line_new = _fmt_copy_line(it)

            # SCHEDULE のスナップショット & 自分の位置と target を特定
            async with SCHEDULE_LOCK:
                sched = list(SCHEDULE)
                try:
                    idx = next(i for i, x in enumerate(sched) if tuple(x["key"]) == tuple(it["key"]))
                except StopIteration:
                    idx = len(sched)

            # idx 以降で最初に copy_msg_id を持つ item を探す（=時間順列の割込み先）
            target = None
            for k in range(idx, len(sched)):
                if sched[k].get("copy_msg_id"):
                    target = (k, sched[k])
                    break

            if not target:
                try:
                    msg = await ch.send(line_new)
                    async with SCHEDULE_LOCK:
                        it["copy_msg_id"] = msg.id
                except Exception as e:
                    print(f"[copy] send failed (tail): {e}")
                continue

            k, tgt_item = target
            target_msg_id = tgt_item["copy_msg_id"]

            try:
                msg_obj = await ch.fetch_message(target_msg_id)
            except Exception as e:
                print(f"[copy] fetch target failed: {e}")
                try:
                    msg = await ch.send(line_new)
                    async with SCHEDULE_LOCK:
                        it["copy_msg_id"] = msg.id
                except Exception as e2:
                    print(f"[copy] fallback send failed: {e2}")
                continue

            carry_text = msg_obj.content
            carry_item = tgt_item

            try:
                await msg_obj.edit(content=line_new)
                async with SCHEDULE_LOCK:
                    it["copy_msg_id"] = msg_obj.id
                    carry_item["copy_msg_id"] = None
            except Exception as e:
                print(f"[copy] edit target failed: {e}")
                try:
                    msg = await ch.send(line_new)
                    async with SCHEDULE_LOCK:
                        it["copy_msg_id"] = msg.id
                except Exception as e2:
                    print(f"[copy] fallback send failed2: {e2}")
                continue

            # 連鎖で後続へ押し出し
            pos = k + 1
            while True:
                async with SCHEDULE_LOCK:
                    next_item = SCHEDULE[pos] if pos < len(SCHEDULE) else None
                    next_msg_id = next_item.get("copy_msg_id") if next_item else None

                if not next_item:
                    try:
                        msg2 = await ch.send(carry_text)
                        async with SCHEDULE_LOCK:
                            carry_item["copy_msg_id"] = msg2.id
                    except Exception as e:
                        print(f"[copy] chain tail send failed: {e}")
                    break

                if not next_msg_id:
                    try:
                        msg2 = await ch.send(carry_text)
                        async with SCHEDULE_LOCK:
                            carry_item["copy_msg_id"] = msg2.id
                    except Exception as e:
                        print(f"[copy] chain send(no next id) failed: {e}")
                    break

                try:
                    next_msg = await ch.fetch_message(next_msg_id)
                except Exception as e:
                    print(f"[copy] fetch next failed: {e}")
                    try:
                        msg2 = await ch.send(carry_text)
                        async with SCHEDULE_LOCK:
                            carry_item["copy_msg_id"] = msg2.id
                    except Exception as e2:
                        print(f"[copy] chain tail send2 failed: {e2}")
                    break

                prev_text = next_msg.content
                try:
                    await next_msg.edit(content=carry_text)
                    async with SCHEDULE_LOCK:
                        carry_item["copy_msg_id"] = next_msg.id
                except Exception as e:
                    print(f"[copy] edit next failed: {e}")
                    try:
                        msg2 = await ch.send(carry_text)
                        async with SCHEDULE_LOCK:
                            carry_item["copy_msg_id"] = msg2.id
                    except Exception as e2:
                        print(f"[copy] chain fallback send failed: {e2}")
                    break

                carry_text = prev_text
                carry_item = next_item
                pos += 1

# ---- ここまで：コピーCHの安全な“割り込み挿入”ロジック ----

# ---- ここから：1分おきの監視・並び替えガード ----

async def _reorder_copy_channel_to_match_schedule():
    """
    コピー用チャンネルのメッセージ内容を、SCHEDULEの**時間順**に整える。
    - 欠落している copy_msg_id は補完（割り込み挿入）
    - 既存メッセージは内容を入れ替えて順序を“見かけ上”揃える（送信順は変えられないため内容で整列）
    """
    if not COPY_CHANNEL_ID:
        return
    ch = await _get_text_channel(COPY_CHANNEL_ID)
    if not ch:
        return

    # 1) まず欠落しているメッセージを補完（ロック無しで安全APIを使用）
    async with SCHEDULE_LOCK:
        missing_items = [it for it in SCHEDULE if not it.get("copy_msg_id")]
    if missing_items:
        await _insert_copy_sorted(missing_items)

    # 2) 再スナップショット（copy_msg_id が生えているもの）
    async with SCHEDULE_LOCK:
        sched_items_with_msg = [it for it in SCHEDULE if it.get("copy_msg_id")]
    if not sched_items_with_msg:
        return

    # 3) 実メッセージを取得（存在しないIDはクリア）
    fetched: Dict[int, Optional[discord.Message]] = {}
    for it in sched_items_with_msg:
        mid = it["copy_msg_id"]
        try:
            msg = await ch.fetch_message(mid)  # type: ignore
            fetched[mid] = msg
        except Exception:
            fetched[mid] = None

    async with SCHEDULE_LOCK:
        for it in sched_items_with_msg:
            if not fetched.get(it["copy_msg_id"]):
                it["copy_msg_id"] = None

    # 4) 失敗したものをもう一度補完
    async with SCHEDULE_LOCK:
        missing_items2 = [it for it in SCHEDULE if not it.get("copy_msg_id")]
    if missing_items2:
        await _insert_copy_sorted(missing_items2)

    # 5) 並びの修正：CH内のメッセージ送信時刻順 と SCHEDULE の when 順を対応付け
    async with SCHEDULE_LOCK:
        # SCHEDULE は when 昇順前提
        sched_in_order = [it for it in SCHEDULE if it.get("copy_msg_id")]
        msg_ids = [it["copy_msg_id"] for it in sched_in_order]

    # 現在のメッセージを取得＆送信時刻順に
    msg_objs: List[discord.Message] = []
    for mid in msg_ids:
        try:
            msg = await ch.fetch_message(mid)  # type: ignore
            msg_objs.append(msg)
        except Exception:
            pass

    if not msg_objs:
        return

    msg_objs.sort(key=lambda m: m.created_at)
    L = min(len(msg_objs), len(sched_in_order))

    # 6) 編集で整列（必ず COPY_LOCK → SCHEDULE_LOCK の順で取得）
    async with COPY_LOCK:
        async with SCHEDULE_LOCK:
            for i in range(L):
                it = sched_in_order[i]
                msg = msg_objs[i]
                desired = _fmt_copy_line(it)
                if msg.content != desired or it["copy_msg_id"] != msg.id:
                    try:
                        await msg.edit(content=desired)
                    except Exception as e:
                        print(f"[guard] copy edit failed: {e}")
                    it["copy_msg_id"] = msg.id

async def _cleanup_copy_orphans(max_scan: int = 300):
    """
    コピーチャンネルの「ボットが送った自動行（server-place-HH:MM:SS）」のうち、
    SCHEDULE から参照されていない（copy_msg_id に含まれない）ものを削除する。
    """
    if not COPY_CHANNEL_ID:
        return
    ch = await _get_text_channel(COPY_CHANNEL_ID)
    if not ch or not bot.user:
        return

    async with SCHEDULE_LOCK:
        valid_ids: Set[int] = {it["copy_msg_id"] for it in SCHEDULE if it.get("copy_msg_id")}

    try:
        async for msg in ch.history(limit=max_scan):
            if msg.author.id != bot.user.id:
                continue
            if msg.pinned:
                continue
            if not RE_COPY_LINE.match(msg.content or ""):
                continue
            if msg.id in valid_ids:
                continue
            try:
                await msg.delete()
            except Exception as e:
                print(f"[cleanup] delete failed: {e}")
    except Exception as e:
        print(f"[cleanup] history failed: {e}")

@tasks.loop(seconds=60.0)
async def order_guard_tick():
    """
    1分おきに：
      - SCHEDULE を厳密に when 昇順へ（timestrもwhen由来で正規化）
      - スキップフラグ再計算
      - 通知ボード再描画
      - コピー用チャンネルの整列補修（欠落補完＆内容入れ替え）
      - コピーチャンネルの孤児メッセージ掃除
    """
    try:
        async with SCHEDULE_LOCK:
            SCHEDULE.sort(key=lambda x: x["when"])
            for it in SCHEDULE:
                # timestr を when に合わせて正規化（ズレの蓄積を防止）
                it["timestr"] = it["when"].astimezone(TIMEZONE).strftime("%H:%M:%S")
            _recompute_skip2m_flags()
    except Exception as e:
        print(f"[guard] schedule sort failed: {e}")

    await _refresh_board()
    try:
        await _reorder_copy_channel_to_match_schedule()
    except Exception as e:
        print(f"[guard] reorder copy channel failed: {e}")

    try:
        await _cleanup_copy_orphans(max_scan=300)
    except Exception as e:
        print(f"[guard] cleanup failed: {e}")

@order_guard_tick.before_loop
async def before_guard():
    await bot.wait_until_ready()

# ---- ここまで：1分おきの監視・並び替えガード ----

async def add_events_and_refresh_board(pairs: List[Tuple[str, int, str]]):
    """
    pairs: [(server, place, timestr)]
    - (server, place) が重複する場合は“遅い時間”を採用（既存を上書き/保持）
    - 完全重複（server, place, timestr） はスキップ
    - 追加して**時間順**に整列
    - 通知ボードを更新
    - コピー専用チャンネルへは**時間順で割り込み**
    """
    if not pairs:
        print("[add] no pairs")
        return

    # --- 入力バッチ内の完全重複除去 ---
    dedup_pairs: List[Tuple[str, int, str]] = []
    seen_in_batch = set()
    for server, place, timestr in pairs:
        key = (server, place, timestr)
        if key in seen_in_batch:
            continue
        seen_in_batch.add(key)
        dedup_pairs.append(key)

    # --- (server, place) 被りは“遅い時間”を採用（同一バッチ内）---
    latest_by_place: Dict[Tuple[str, int], Tuple[str, datetime]] = {}
    for server, place, timestr in dedup_pairs:
        when = _next_occurrence_today_or_tomorrow(timestr)
        k = (server, place)
        prev = latest_by_place.get(k)
        if (not prev) or (when > prev[1]):
            latest_by_place[k] = (timestr, when)

    new_items: List[Dict] = []
    replaced_items: List[Dict] = []

    async with SCHEDULE_LOCK:
        existing_exact = { (it["server"], it["place"], it["timestr"]) for it in SCHEDULE }
        for (server, place), (timestr, when) in latest_by_place.items():
            key = (server, place, timestr)
            if key in existing_exact:
                continue  # 完全重複は何もしない

            # 既存に (server, place) があれば “遅い方” を採用
            same_place_items = [it for it in SCHEDULE if it["server"] == server and it["place"] == place]
            if same_place_items:
                current_latest = max(same_place_items, key=lambda x: x["when"])
                if when > current_latest["when"]:
                    # 新しい方が遅い → 既存を全て削除して置き換え
                    for it in same_place_items:
                        if it in SCHEDULE:
                            SCHEDULE.remove(it)
                            replaced_items.append(it)
                else:
                    # 既存の方が遅い（または同時刻）→ 追加不要
                    continue

            item = {
                "when": when, "server": server, "place": place, "timestr": timestr,
                "key": (server, place, timestr), "skip2m": False, "sent_2m": False, "sent_15s": False,
                "copy_msg_id": None
            }
            SCHEDULE.append(item)
            new_items.append(item)

        if not new_items and not replaced_items:
            print("[add] no changes (all duplicates or earlier than existing)")
            return

        SCHEDULE.sort(key=lambda x: x["when"])
        _recompute_skip2m_flags()

    # コピーCH：置換で消したメッセージをまず消す
    for it in replaced_items:
        await _delete_copy_message_if_exists(it)

    # コピーCH：安全に割り込み挿入（内部で時間順へソート）
    await _insert_copy_sorted(new_items)

    # 通知ボード更新
    await _refresh_board()

async def _clear_all_schedules() -> int:
    """全スケジュールを削除（コピーCHのメッセージも個別に削除）"""
    async with SCHEDULE_LOCK:
        items = list(SCHEDULE)
        SCHEDULE.clear()
        _recompute_skip2m_flags()
    for it in items:
        await _delete_copy_message_if_exists(it)
    await _refresh_board()
    return len(items)

async def _delete_events(server: Optional[str] = None, place: Optional[int] = None, timestr: Optional[str] = None) -> int:
    """
    指定条件に一致する予定を削除し件数を返す。
    timestrがNoneなら server+place の全件を削除。
    """
    to_delete: List[Dict] = []
    async with SCHEDULE_LOCK:
        for it in list(SCHEDULE):
            if server is not None and it["server"] != server:
                continue
            if place is not None and it["place"] != place:
                continue
            if timestr is not None and it["timestr"] != timestr:
                continue
            SCHEDULE.remove(it)
            to_delete.append(it)
        if to_delete:
            SCHEDULE.sort(key=lambda x: x["when"])
            _recompute_skip2m_flags()
    # コピーメッセ削除
    for it in to_delete:
        await _delete_copy_message_if_exists(it)
    # ボード更新
    if to_delete:
        await _refresh_board()
    return len(to_delete)

def _parse_time_str(s: str) -> Optional[str]:
    """'H:MM' / 'HH:MM' / 'HH:MM:SS' を 'HH:MM:SS' に整形"""
    if not s:
        return None
    n = unicodedata.normalize("NFKC", s).replace("：", ":").strip()
    m3 = re.fullmatch(r"(\d{1,2}):(\d{1,2}):(\d{1,2})", n)
    if m3:
        h, m, s = (int(m3.group(1)), int(m3.group(2)), int(m3.group(3)))
        if 0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return None
    m2 = re.fullmatch(r"(\d{1,2}):(\d{1,2})", n)
    if m2:
        h, m = (int(m2.group(1)), int(m2.group(2)))
        if 0 <= h < 24 and 0 <= m < 60:
            return f"{h:02d}:{m:02d}:00"
    return None

def _parse_server_token(tok: str) -> Optional[str]:
    """
    's123' '[S123]' 'S123' '123' -> '123' を返す
    （ユーザー入力では s なしの「1234-1-17:00:00」形式も許可）
    """
    if not tok:
        return None
    n = unicodedata.normalize("NFKC", tok)
    # まずは S付きの既存パターン
    m = RE_SERVER.search(n)
    if m:
        return m.group(1)
    # 素の数字だけでもOK（2〜5桁）
    m2 = re.fullmatch(r"\D*([0-9]{2,5})\D*", n)
    return m2.group(1) if m2 else None

def _parse_place_token(tok: str) -> Optional[int]:
    if not tok:
        return None
    n = unicodedata.normalize("NFKC", tok)
    m = re.fullmatch(r"\D*(\d{1,3})\D*", n)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def _parse_spec_tokens(args: List[str]) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    入力の解釈（例）:
      - s123 5 12:34[:56]
      - 1234 5 12:34[:56]
      - s123-5-12:34[:56]
      - 1234-5-12:34[:56]
      - s123-5       （!del で timestr 省略可）
      - 1234-5       （!del で timestr 省略可）
    """
    one = " ".join(args).strip()
    m = re.fullmatch(r"\s*([^\s\-]+)\s*-\s*(\d{1,3})(?:\s*-\s*([0-9:：]{3,8}))?\s*", one)
    server = place = timestr = None
    if m:
        server = _parse_server_token(m.group(1))
        place  = _parse_place_token(m.group(2))
        timestr = _parse_time_str(m.group(3)) if m.group(3) else None
        return server, place, timestr

    if len(args) >= 2:
        server = _parse_server_token(args[0])
        place  = _parse_place_token(args[1])
        timestr = _parse_time_str(args[2]) if len(args) >= 3 else None
        return server, place, timestr

    # 単体で s123 / 1234 などは不可
    return None, None, None

# ---------------------------
# スケジューラ（アラート＋消込み）
# ---------------------------

@tasks.loop(seconds=1.0)
async def scheduler_tick():
    """
    毎秒チェックして：
      - 2分前/15秒前をアラートチャンネルに通知（5秒後削除）
        ※ 次の予定が5分以内なら2分前は通知しない
      - 本番時刻到達で一覧から削除→ボード編集
      - その際、コピー専用チャンネルの個別メッセージも削除
      - （本番時の ⏰ 通知メッセージ送信は行わない）
    """
    now = datetime.now(TIMEZONE)

    # アラートチャンネル取得
    alert_ch: Optional[discord.TextChannel] = None
    if ALERT_CHANNEL_ID:
        alert_ch = await _get_text_channel(ALERT_CHANNEL_ID)

    fired: List[Dict] = []
    to_alert_2m: List[Dict] = []
    to_alert_15s: List[Dict] = []

    async with SCHEDULE_LOCK:
        for it in SCHEDULE:
            dt = (it["when"] - now).total_seconds()

            # 2分前（抑制フラグが True の場合は送らない）
            if not it.get("sent_2m", False) and not it.get("skip2m", False) and 0 < dt <= 120:
                it["sent_2m"] = True
                to_alert_2m.append(it)

            # 15秒前
            if not it.get("sent_15s", False) and 0 < dt <= 15:
                it["sent_15s"] = True
                to_alert_15s.append(it)

            # 本番
            if it["when"] <= now:
                fired.append(it)

        if fired:
            keys_fired = {tuple(x["key"]) for x in fired}
            SCHEDULE[:] = [x for x in SCHEDULE if tuple(x["key"]) not in keys_fired]
            SCHEDULE.sort(key=lambda x: x["when"])
            _recompute_skip2m_flags()

    # アラート送信（5秒後削除）※ロールメンションのみ
    if alert_ch is not None:
        for it in to_alert_2m:
            await _send_temp_alert(alert_ch, f"⏳ **2分前**: {it['server']}-{it['place']}-{it['timestr']}")
        for it in to_alert_15s:
            await _send_temp_alert(alert_ch, f"⏱️ **15秒前**: {it['server']}-{it['place']}-{it['timestr']}")

    # 本番：コピー専用チャンネルの個別メッセージを削除（通知送信はしない）
    if fired:
        for it in fired:
            await _delete_copy_message_if_exists(it)
        # ボード更新（過ぎたものを消した状態に）
        await _refresh_board()

@scheduler_tick.before_loop
async def before_scheduler():
    await bot.wait_until_ready()

# ---------------------------
# Helpers（画像系）
# ---------------------------

def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s).replace(" ", "")

def load_image_from_bytes(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    im = ImageOps.exif_transpose(im)  # EXIFの向きを補正
    return im.convert("RGBA")

def resize_to_width(im: Image.Image, width: int = TARGET_WIDTH) -> Image.Image:
    if im.width == width:
        return im
    h = int(round(im.height * width / im.width))
    return im.resize((width, h), Image.LANCZOS)

def slice_exact_7(im: Image.Image, cuts_pct: List[float]) -> List[Image.Image]:
    """境界％から7ブロックに分割（1..7）"""
    w, h = im.size
    boundaries = [int(round(h * p / 100.0)) for p in cuts_pct]
    y = [0] + boundaries + [h]
    parts = []
    for i in range(7):
        parts.append(im.crop((0, y[i], w, y[i+1])))
    return parts

def trim_lr_percent(im: Image.Image, left_pct: float, right_pct: float) -> Image.Image:
    w, h = im.size
    left = int(round(w * left_pct / 100.0))
    right = w - int(round(w * right_pct / 100.0))
    left = max(0, min(left, w - 1))
    right = max(left + 1, min(right, w))
    return im.crop((left, 0, right, h))

def google_ocr_word_boxes(pil_im: Image.Image) -> List[Tuple[str, Tuple[int,int,int,int]]]:
    """Google Vision で word 単位の文字とバウンディングボックスを返す"""
    buf = io.BytesIO()
    pil_im.convert("RGB").save(buf, format="JPEG", quality=95)
    image = vision.Image(content=buf.getvalue())

    response = gcv_client.document_text_detection(image=image)
    if response.error.message:
        raise RuntimeError(response.error.message)

    words: List[Tuple[str, Tuple[int,int,int,int]]] = []
    if not response.full_text_annotation.pages:
        return words

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for para in block.paragraphs:
                for word in para.words:
                    txt = "".join([s.text for s in word.symbols])
                    xs = [v.x for v in word.bounding_box.vertices]
                    ys = [v.y for v in word.bounding_box.vertices]
                    x1, x2 = min(xs), max(xs)
                    y1, y2 = min(ys), max(ys)
                    words.append((txt, (x1, y1, x2, y2)))
    return words

def google_ocr_line_boxes(pil_im: Image.Image, y_tol: int = 18) -> List[Tuple[str, Tuple[int,int,int,int]]]:
    """wordをY座標で行グループ化して (line_text, (x1,y1,x2,y2)) を返す。"""
    words = google_ocr_word_boxes(pil_im)
    if not words:
        return []

    # 中心Yでソート
    items = []
    for txt, (x1, y1, x2, y2) in words:
        cy = (y1 + y2) / 2.0
        items.append((cy, x1, y1, x2, y2, txt))
    items.sort(key=lambda t: (t[0], t[1]))

    lines: List[List[Tuple[int,int,int,int,str]]] = []
    for cy, x1, y1, x2, y2, txt in items:
        if not lines:
            lines.append([(x1, y1, x2, y2, txt)])
            continue
        last = lines[-1]
        ly1 = min(a[1] for a in last)
        ly2 = max(a[3] for a in last)
        lcy = (ly1 + ly2) / 2.0
        if abs(cy - lcy) <= y_tol:
            last.append((x1, y1, x2, y2, txt))
        else:
            lines.append([(x1, y1, x2, y2, txt)])

    line_boxes: List[Tuple[str, Tuple[int,int,int,int]]] = []
    for chunks in lines:
        chunks.sort(key=lambda a: a[0])  # x1
        text = "".join(c[4] for c in chunks)  # スペース無しで連結
        x1 = min(c[0] for c in chunks)
        y1 = min(c[1] for c in chunks)
        x2 = max(c[2] for c in chunks)
        y2 = max(c[3] for c in chunks)
        line_boxes.append((text, (x1, y1, x2, y2)))
    return line_boxes

def compact_7_by_removing_sections(pil_im: Image.Image) -> Image.Image:
    """
    7番ブロック内で、タイトル～免戦/時間の直下まで残し、それ以降を詰める
    """
    im = pil_im.copy()
    w, h = im.size

    lines = google_ocr_line_boxes(im, y_tol=18)

    titles: List[Tuple[int,int]] = []
    candidates: List[Tuple[int,int]] = []

    for text, (x1, y1, x2, y2) in lines:
        t = _norm(text)
        if RE_TITLE_COMPACT.search(t):
            titles.append((y1, y2))
        # 「免戦中」1文字OK + 時刻っぽい も候補に
        if _has_time_like(text) or _is_immune_line(text):
            candidates.append((y1, y2))

    titles.sort(key=lambda p: p[0])
    candidates.sort(key=lambda p: p[0])

    if not titles:
        return im

    keep_slices: List[Tuple[int,int]] = []
    for i, (t_y1, t_y2) in enumerate(titles):
        start = t_y1
        end = titles[i + 1][0] if i + 1 < len(titles) else h
        if end <= start:
            continue

        cand_bottom = None
        for cy1, cy2 in candidates:
            if start <= cy1 < end:
                cand_bottom = cy2
        if cand_bottom is not None:
            cut_at = cand_bottom
        else:
            cut_at = min(end, int(round(start + (end - start) * FALLBACK_KEEP_TOP_RATIO)))
            cut_at = max(cut_at, t_y2)

        if cut_at > start:
            keep_slices.append((start, cut_at))

    if not keep_slices:
        return im

    segments = [im.crop((0, a, w, b)) for (a, b) in keep_slices]
    out_h = sum(seg.height for seg in segments)
    out = Image.new("RGBA", (w, out_h), (0, 0, 0, 0))
    y = 0
    for seg in segments:
        out.paste(seg, (0, y))
        y += seg.height
    return out

def hstack(im_left: Image.Image, im_right: Image.Image, gap: int = 8, bg=(0,0,0,0)) -> Image.Image:
    """左右結合（高さは大きい方に合わせ中央寄せ）"""
    h = max(im_left.height, im_right.height)
    w = im_left.width + gap + im_right.width
    canvas = Image.new("RGBA", (w, h), bg)
    y1 = (h - im_left.height)//2
    y2 = (h - im_right.height)//2
    canvas.paste(im_left, (0, y1))
    canvas.paste(im_right, (im_left.width + gap, y2))
    return canvas

def vstack(images: List[Image.Image], gap: int = 8, bg=(0,0,0,0)) -> Image.Image:
    """縦結合（幅は最大に合わせ中央寄せ）"""
    if not images:
        raise ValueError("no images to stack")
    max_w = max(img.width for img in images)
    total_h = sum(img.height for img in images) + gap*(len(images)-1)
    canvas = Image.new("RGBA", (max_w, total_h), bg)
    y = 0
    for img in images:
        x = (max_w - img.width)//2
        canvas.paste(img, (x, y))
        y += img.height + gap
    return canvas

def vstack_uniform_width(images: List[Image.Image], width: int) -> Image.Image:
    """幅をそろえてから縦結合（デバッグで複数画像返す用）"""
    resized = []
    for im in images:
        if im.width != width:
            h = int(round(im.height * width / im.width))
            im = im.resize((width, h), Image.LANCZOS)
        resized.append(im)
    return vstack(resized, gap=12, bg=(0,0,0,0))

# ---------------------------
# OpenAI OCR
# ---------------------------

def openai_ocr_png(pil_im: Image.Image) -> Tuple[str, bytes]:
    """OpenAI へ画像OCR依頼。返り値: (テキスト, 送ったPNGバイト列)"""
    buf = io.BytesIO()
    pil_im.convert("RGB").save(buf, format="PNG")
    png_bytes = buf.getvalue()

    b64 = base64.b64encode(png_bytes).decode("utf-8")
    resp = oai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "以下の画像に写っている日本語テキストを、そのまま読み取ってください（改行と数字も保持）。"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]
        }],
        temperature=0.0,
    )
    text = resp.choices[0].message.content.strip()
    return text, png_bytes

# ---------------------------
# 計算 & フォーマット（理由付き）
# ---------------------------

def _time_to_seconds(t: str, *, prefer_mmss: bool = False) -> int:
    """
    時刻/時間文字列を秒に。
    prefer_mmss=True のとき 2 区切りは MM:SS と解釈（免戦中向け）。
    """
    t = _norm(t).replace("：", ":")
    m3 = re.match(r"^(\d{1,2}):(\d{2}):(\d{2})$", t)
    if m3:
        h, m, s = map(int, m3.groups())
        return h*3600 + m*60 + s
    m2 = re.match(r"^(\d{1,2}):(\d{2})$", t)
    if m2:
        a, b = map(int, m2.groups())
        if prefer_mmss:
            return a*60 + b      # MM:SS
        return a*3600 + b*60     # HH:MM
    return 0

def _seconds_to_hms(sec: int) -> str:
    sec %= 24*3600
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def _build_failure_message(diag: Dict) -> str:
    reasons = []

    if not diag.get("base_found"):
        reasons.append(f"基準時刻が見つかりませんでした。")
    if diag.get("place_total", 0) == 0:
        reasons.append("「越域/戦闘 駐◯場」のタイトル行が見つかりませんでした。")
    else:
        missing = diag.get("place_missing_immune", 0)
        if missing == diag.get("place_total"):
            reasons.append(f"駐騎場は {diag['place_total']} 件検出しましたが、免戦時間(MM:SS)が見つかりませんでした。")
        elif missing > 0:
            reasons.append(f"{missing} 件の駐騎場で免戦時間(MM:SS)が見つかりませんでした。")
    if not diag.get("server_found"):
        reasons.append("サーバー番号（例: [s1234]）が見つかりませんでした。")

    msg = "⚠️ 解析完了… ですが計算できませんでした。\n\n原因:\n"
    if reasons:
        msg += "".join(f"- {r}\n" for r in reasons)
    else:
        msg += "- 原因を特定できませんでした（OCR結果に十分な情報がありませんでした）。\n"

    # 未使用テキスト（当てはまらなかった行）
    unused = diag.get("unused_lines", [])
    if unused:
        # Discordのメッセ長対策で40行まで
        head = "未使用テキスト（パースに使えなかった行）:"
        block = "\n".join(unused[:40])
        if len(unused) > 40:
            block += "\n…（省略）"
        msg += f"\n{head}\n```\n{block}\n```"
    return msg

def parse_and_compute(oai_text: str) -> Tuple[Optional[str], Optional[str], Optional[str], List[Tuple[int, str]], Dict]:
    """
    OCRテキストから
      server(str), base_time(HH:MM:SS), ceasefire(HH:MM:SS), results[(place, time_str)], diag(dict)
    を返す。diag は失敗理由と「未使用テキスト」生成に使う。
    """
    raw_lines = [ln.rstrip() for ln in oai_text.splitlines()]
    lines = [ln.strip() for ln in raw_lines if ln.strip()]
    if not lines:
        return None, None, None, [], {"unused_lines": [], "server_found": False, "base_found": False,
                                      "place_total": 0, "place_missing_immune": 0}

    used_idx: Set[int] = set()

    server = None
    base_time_sec: Optional[int] = None
    ceasefire_sec: Optional[int] = None

    # pair: {'place':int,'place_idx':int,'immune_sec':Optional[int],'immune_idx':Optional[int]}
    pairs: List[Dict] = []

    # 1周: サーバー / 基準 / 停戦終了 / 越域駐〇場 + 免戦 を順に拾う
    for i, raw in enumerate(lines):
        n = _norm(raw)

        # server
        if server is None:
            m = RE_SERVER.search(n)
            if m:
                server = m.group(1)
                used_idx.add(i)

        # ceasefire
        if "停戦" in n:
            tt = _extract_time_like(raw)
            if tt:
                ceasefire_sec = _time_to_seconds(tt, prefer_mmss=False)
                used_idx.add(i)

        # base_time（免戦/停戦行は除外）
        if base_time_sec is None and (not _is_immune_line(raw)) and ("停戦" not in n):
            tt = _extract_time_like(raw)
            if tt:
                base_time_sec = _time_to_seconds(tt, prefer_mmss=False)
                used_idx.add(i)

        # place（タイトルや同一行の「…場 4 免戦中 …」にも反応）
        pl = _extract_place(raw)
        if pl is not None:
            pairs.append({"place": pl, "place_idx": i, "immune_sec": None, "immune_idx": None})

        # immune time
        if _is_immune_line(raw):
            # 免戦行に場番号が含まれているか（タイトル式 or 行頭数字式）
            imm_pl: Optional[int] = pl
            if imm_pl is None:
                m_bare = RE_BARE_PLACE_BEFORE_IMMUNE.search(unicodedata.normalize("NFKC", raw))
                if m_bare:
                    try:
                        imm_pl = int(m_bare.group(1))
                    except Exception:
                        imm_pl = None

            tt = _extract_time_like(raw)
            if tt:
                tsec = _time_to_seconds(tt, prefer_mmss=True)

                assigned = False
                if imm_pl is not None:
                    # 同じ場番号で未割当の最新ペアを優先して割当
                    for j in range(len(pairs)-1, -1, -1):
                        if pairs[j]["place"] == imm_pl and pairs[j]["immune_sec"] is None:
                            pairs[j]["immune_sec"] = tsec
                            pairs[j]["immune_idx"] = i
                            assigned = True
                            break
                    # まだ無ければ、当該場の新規ペアをこの行で作成
                    if not assigned:
                        pairs.append({"place": imm_pl, "place_idx": i, "immune_sec": tsec, "immune_idx": i})
                        assigned = True
                if not assigned:
                    # フォールバック：直近の未割当ペアに付与
                    for j in range(len(pairs)-1, -1, -1):
                        if pairs[j]["immune_sec"] is None:
                            pairs[j]["immune_sec"] = tsec
                            pairs[j]["immune_idx"] = i
                            assigned = True
                            break
                # used は「有効ペア」確定後に加算する（後でまとめて）

    # 計算
    calc: List[Tuple[int, int]] = []
    for pr in pairs:
        if pr["immune_sec"] is None or base_time_sec is None:
            continue
        sec = (base_time_sec + int(pr["immune_sec"])) % (24*3600)
        calc.append((pr["place"], sec))
        # このペアは有効に使われたので参照行を used に入れる
        used_idx.add(pr["place_idx"])
        used_idx.add(pr["immune_idx"])

    # 停戦補正
    if calc and ceasefire_sec is not None:
        delta = (ceasefire_sec - calc[0][1])
        calc = [(pl, (sec + delta) % (24*3600)) for (pl, sec) in calc]
        calc[0] = (calc[0][0], ceasefire_sec % (24*3600))

    # 出力整形
    results: List[Tuple[int, str]] = [(pl, _seconds_to_hms(sec)) for (pl, sec) in calc]
    base_str = _seconds_to_hms(base_time_sec) if base_time_sec is not None else None
    cease_str = _seconds_to_hms(ceasefire_sec) if ceasefire_sec is not None else None

    # 診断情報
    place_total = len(pairs)
    place_missing_immune = sum(1 for pr in pairs if pr["immune_sec"] is None)
    unused_lines = [lines[i] for i in range(len(lines)) if i not in used_idx]

    diag = {
        "server_found": server is not None,
        "base_found": base_time_sec is not None,
        "place_total": place_total,
        "place_missing_immune": place_missing_immune,
        "unused_lines": unused_lines
    }
    return server, base_str, cease_str, results, diag

def build_result_message(server: Optional[str],
                         base_str: Optional[str],
                         cease_str: Optional[str],
                         results: List[Tuple[int, str]],
                         diag: Dict) -> str:
    # 成功
    if base_str and results and server:
        head = f"✅ 解析完了！⏱️ 基準時間:{base_str}" + (f" ({cease_str})" if cease_str else "")
        body_lines = [f"{server}-{pl}-{t}" for (pl, t) in results]
        return head + "\n" + "\n".join(body_lines)
    # 失敗（理由つき）
    return _build_failure_message(diag)

# ---------------------------
# パイプライン
# ---------------------------

def process_image_pipeline(pil_im: Image.Image) -> Tuple[Image.Image, str, str, List[Tuple[int, str]], str]:
    """
    リサイズ→スライス→トリム→（7を詰め処理）→合成→OpenAI OCR→計算
    戻り値: (最終合成画像, 結果メッセージ, server, results, ocr_text)
    """
    base = resize_to_width(pil_im, TARGET_WIDTH)
    parts = slice_exact_7(base, CUTS)

    kept: Dict[int, Image.Image] = {}
    for idx in KEEP:
        block = parts[idx - 1]
        l_pct, r_pct = TRIM_RULES[idx]
        kept[idx] = trim_lr_percent(block, l_pct, r_pct)

    kept[7] = compact_7_by_removing_sections(kept[7])
    top_row = hstack(kept[6], kept[2], gap=8)
    final_img = vstack([top_row, kept[4], kept[7]], gap=10)

    oai_text, _ = openai_ocr_png(final_img)

    server, base_str, cease_str, results, diag = parse_and_compute(oai_text)
    message = build_result_message(server, base_str, cease_str, results, diag)

    return final_img, message, (server or ""), results, oai_text

# ---------------------------
# 共通実行（複数画像対応）
# ---------------------------

IMAGE_MIME_PREFIXES = ("image/",)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

def _is_image_attachment(a: discord.Attachment) -> bool:
    if a.content_type and any(a.content_type.startswith(p) for p in IMAGE_MIME_PREFIXES):
        return True
    return a.filename.lower().endswith(IMAGE_EXTS)

async def run_pipeline_for_attachments(
    atts: List[discord.Attachment],
    *,
    want_image: bool
) -> Tuple[Optional[discord.File], str, List[Tuple[str, int, str]], str]:
    """
    複数画像を処理。
    return:
      - fileobj: 画像を返す場合は1枚（縦結合）
      - message: 全結果の連結テキスト＋末尾に
                 「登録リスト（重複は遅い時刻のみ／並べ替えなし=OCR順で表示）」を付与
      - pairs:   スケジュール登録用（同一(server,place)は遅い時刻のみ→時間順で整列）
      - ocr_joined: すべてのOCRテキストを連結（!oaiocr用デバッグ表示）
    """
    images: List[Image.Image] = []
    messages: List[str] = []
    # 表示用の“生”候補（OCR検出順を保持）
    raw_pairs_all: List[Tuple[str, int, str]] = []
    ocr_texts: List[str] = []

    loop = asyncio.get_event_loop()

    for idx, a in enumerate(atts, start=1):
        data = await a.read()
        pil = load_image_from_bytes(data)
        final_img, msg, server, results, ocr_text = await loop.run_in_executor(None, process_image_pipeline, pil)
        images.append(final_img)
        messages.append(msg)
        ocr_texts.append(f"# 画像{idx}\n{ocr_text}")

        # OCR順のまま収集（表示用）
        for place, tstr in results:
            if server:
                raw_pairs_all.append((server, place, tstr))

    # ---- スケジュール登録用：同一(server,place)は“遅い時刻のみ”→時間順に整列 ----
    latest_by_place: Dict[Tuple[str, int], Tuple[str, datetime]] = {}
    for server, place, timestr in raw_pairs_all:
        when = _next_occurrence_today_or_tomorrow(timestr)
        k = (server, place)
        prev = latest_by_place.get(k)
        if (not prev) or (when > prev[1]):
            latest_by_place[k] = (timestr, when)

    sorted_items: List[Tuple[str, int, str, datetime]] = sorted(
        ((srv, plc, ts, when) for (srv, plc), (ts, when) in latest_by_place.items()),
        key=lambda x: x[3]
    )
    pairs_all: List[Tuple[str, int, str]] = [(srv, plc, ts) for (srv, plc, ts, _w) in sorted_items]

    # ---- 表示用：重複は“遅い時刻”だけを1件表示、順番は並べ替えなし（OCR順） ----
    latest_timestr_map: Dict[Tuple[str, int], str] = {k: v[0] for k, v in latest_by_place.items()}
    reg_lines: List[str] = []
    already_emitted: Set[Tuple[str, int]] = set()  # 同時刻重複対策（片方だけ表示）
    for srv, plc, ts in raw_pairs_all:
        want_ts = latest_timestr_map.get((srv, plc))
        if want_ts is None:
            continue
        if ts != want_ts:
            continue  # “遅い時刻”だけ表示
        if (srv, plc) in already_emitted:
            continue  # 完全重複は片方だけ
        reg_lines.append(f"{srv}-{plc}-{ts}")
        already_emitted.add((srv, plc))

    reg_block = ""
    if reg_lines:
        reg_block = "📌 登録リスト（重複は“遅い時刻”のみ／並べ替えなし）\n" + "\n".join(reg_lines)

    # 連結テキスト
    full_message = "\n\n".join(messages) if messages else "⚠️ 結果がありませんでした。"
    if reg_block:
        full_message = f"{full_message}\n\n{reg_block}"

    # OCR原文（!oaiocr のみ付与）
    ocr_joined = "\n\n".join(ocr_texts) if ocr_texts else ""

    # 画像は1枚にまとめる or 返さない
    fileobj: Optional[discord.File] = None
    if want_image and images:
        merged = vstack_uniform_width(images, width=TARGET_WIDTH)
        out = io.BytesIO()
        merged.convert("RGB").save(out, format="PNG")
        out.seek(0)
        fileobj = discord.File(out, filename="result.png")

    return fileobj, full_message, pairs_all, ocr_joined

# ---------------------------
# Commands
# ---------------------------

@bot.command(name="oaiocr", help="画像を添付して実行。処理→詰め→OpenAI OCR→計算（複数画像OK）。OCR原文も返します。※このコマンドはスケジュール登録しません。")
async def oaiocr(ctx: commands.Context):
    try:
        atts = [a for a in ctx.message.attachments if _is_image_attachment(a)]
        if not atts:
            await ctx.reply("画像を添付して `!oaiocr` を実行してください。")
            return

        # まずは即レス（のちに編集）
        placeholder = await ctx.reply("解析中…🔎")

        fileobj, message, pairs, ocr_all = await run_pipeline_for_attachments(atts, want_image=True)

        # 結果＋OCR原文（コードブロック）に編集差し替え。画像は別送（1枚に統合）
        if ocr_all:
            message = f"{message}\n\n🧾 OpenAI OCR 原文:\n```\n{ocr_all}\n```"

        await placeholder.edit(content=message)
        if fileobj:
            await ctx.send(file=fileobj)

        # ❌ `!oaiocr` はスケジュール登録しない
        # if pairs:
        #     await add_events_and_refresh_board(pairs)

    except Exception as e:
        await ctx.reply(f"エラー: {e}")

# ------------- 手動追加 / 削除 / 全リセット --------------

def _has_manage_perm(ctx: commands.Context) -> bool:
    gp = getattr(ctx.author, "guild_permissions", None)
    return bool(gp and (gp.administrator or gp.manage_guild or gp.manage_messages))

@bot.command(
    name="add",
    aliases=["a"],
    help="手動追加: !add s123 5 12:34[:56] / !add 1234 5 12:34[:56] / !add s123-5-12:34[:56] / !add 1234-5-12:34[:56]\n同一駐騎場が既にあれば遅い時間を採用（置換）"
)
async def cmd_add(ctx: commands.Context, *args):
    try:
        server, place, timestr = _parse_spec_tokens(list(args))
        if not server or place is None or not timestr:
            await ctx.reply("使い方: `!add 1234-1-17:00:00` など（`!add s123 1 17:00` 形式も可）")
            return

        # 置換ロジックは add_events_and_refresh_board に内蔵
        await add_events_and_refresh_board([(server, place, timestr)])
        await ctx.reply(f"追加（または置換）しました: S{server}-{place}-{timestr}")
    except Exception as e:
        await ctx.reply(f"エラー: {e}")

@bot.command(
    name="del",
    help="削除: !del 1234-1-17:00:00 / !del 1234-1（この場合は該当駐騎場の全時刻を削除） / `s123` 形式も可"
)
async def cmd_del(ctx: commands.Context, *args):
    try:
        if not _has_manage_perm(ctx):
            await ctx.reply("権限がありません。（サーバー管理/メッセージ管理）")
            return
        server, place, timestr = _parse_spec_tokens(list(args))
        if not server or place is None:
            await ctx.reply("使い方: `!del 1234-1-17:00:00` または `!del 1234-1`（全件）")
            return

        n = await _delete_events(server=server, place=place, timestr=timestr)
        if n == 0:
            await ctx.reply("該当する予定は見つかりませんでした。")
        else:
            if timestr:
                await ctx.reply(f"削除しました: S{server}-{place}-{timestr}（{n}件）")
            else:
                await ctx.reply(f"削除しました: S{server}-{place}-*（{n}件）")
    except Exception as e:
        await ctx.reply(f"エラー: {e}")

@bot.command(name="reset", help="全スケジュールをリセット（要権限）")
async def cmd_reset(ctx: commands.Context):
    try:
        if not _has_manage_perm(ctx):
            await ctx.reply("権限がありません。（サーバー管理/メッセージ管理）")
            return
        n = await _clear_all_schedules()
        await ctx.reply(f"🧹 全スケジュールをリセットしました（{n}件削除）")
    except Exception as e:
        await ctx.reply(f"エラー: {e}")

# ---- ±1秒 調整コマンド ----------------------------------------------

async def _shift_items(places: Optional[Set[int]], delta_seconds: int) -> int:
    """
    places が None の場合は全件、そうでなければ place が一致するものだけ
    'when' を±deltaし、'timestr' もTZに合わせて再計算する。
    コピーCHの個別メッセージ内容も更新（順序は維持）。
    """
    changed: List[Dict] = []
    async with SCHEDULE_LOCK:
        for it in SCHEDULE:
            if (places is None) or (it["place"] in places):
                it["when"] = it["when"] + timedelta(seconds=delta_seconds)
                it["timestr"] = it["when"].astimezone(TIMEZONE).strftime("%H:%M:%S")
                changed.append(it)
        if changed:
            SCHEDULE.sort(key=lambda x: x["when"])
            _recompute_skip2m_flags()

    # コピーCHの内容を更新（順序はそのまま）
    if changed and COPY_CHANNEL_ID:
        ch = await _get_text_channel(COPY_CHANNEL_ID)
        if ch:
            for it in changed:
                mid = it.get("copy_msg_id")
                if not mid:
                    continue
                try:
                    msg_obj = await ch.fetch_message(mid)
                    await msg_obj.edit(content=_fmt_copy_line(it))
                except Exception as e:
                    print(f"[shift] copy edit failed: {e}")

    # ボード更新
    if changed:
        await _refresh_board()
    return len(changed)

def _parse_places_from_args(args: Tuple[str, ...]) -> Optional[Set[int]]:
    if not args:
        return None
    s: Set[int] = set()
    for tok in args:
        m = re.search(r"(\d{1,3})", tok)
        if m:
            try:
                s.add(int(m.group(1)))
            except Exception:
                pass
    return s or None

@bot.command(name="1", help="駐騎ナンバーを +1 秒。例: `!1 1 4 5 6`（指定なしで全件）")
async def cmd_plus1(ctx: commands.Context, *args):
    places = _parse_places_from_args(args)
    n = await _shift_items(places, +1)
    if places:
        await ctx.reply(f"＋1秒しました（対象:{sorted(places)} / {n}件）", allowed_mentions=discord.AllowedMentions.none())
    else:
        await ctx.reply(f"＋1秒しました（全件 / {n}件）", allowed_mentions=discord.AllowedMentions.none())

@bot.command(name="-1", help="駐騎ナンバーを -1 秒。例: `!-1 1 4 5 6`（指定なしで全件）")
async def cmd_minus1(ctx: commands.Context, *args):
    places = _parse_places_from_args(args)
    n = await _shift_items(places, -1)
    if places:
        await ctx.reply(f"−1秒しました（対象:{sorted(places)} / {n}件）", allowed_mentions=discord.AllowedMentions.none())
    else:
        await ctx.reply(f"−1秒しました（全件 / {n}件）", allowed_mentions=discord.AllowedMentions.none())

# ---- Role ID helper commands ----------------------------------------------

@bot.command(
    name="roleid",
    help="ロールIDを表示: !roleid @ロール / !roleid ロール名 / !roleid 123456789012345678"
)
@commands.guild_only()
async def roleid(ctx: commands.Context, *, query: str = ""):
    """@メンション / 名前 / 数字ID のいずれかでロールを指定 → IDを返す"""
    guild = ctx.guild
    role = None

    # 1) メッセージにロールメンションが含まれていれば最優先
    if ctx.message.role_mentions:
        role = ctx.message.role_mentions[0]

    # 2) 数字（ID）っぽいものがあれば ID として解決
    if role is None:
        m = re.search(r"(\d{17,20})", query)
        if m:
            role = guild.get_role(int(m.group(1)))

    # 3) 名前で検索（完全一致→部分一致）
    if role is None and query:
        # 完全一致（大文字小文字区別）
        for r in guild.roles:
            if r.name == query:
                role = r
                break
        # 部分一致（大文字小文字無視）
        if role is None:
            q = query.casefold()
            candidates = [r for r in guild.roles if q in r.name.casefold()]
            if len(candidates) == 1:
                role = candidates[0]
            elif len(candidates) > 1:
                lines = [f"{r.name} : {r.id}" for r in candidates[:10]]
                txt = "候補が複数あります。より正確に指定してください：\n```\n" + "\n".join(lines) + "\n```"
                await ctx.reply(txt, allowed_mentions=discord.AllowedMentions.none())
                return

    if role is None:
        await ctx.reply("ロールが見つかりませんでした。`!roles` で一覧を確認してください。", 
                        allowed_mentions=discord.AllowedMentions.none())
        return

    await ctx.reply(f"{role.name} : {role.id}", allowed_mentions=discord.AllowedMentions.none())


@bot.command(name="roles", help="このサーバーのロール一覧とIDを表示")
@commands.guild_only()
async def roles(ctx: commands.Context):
    """ロール一覧を上位順で表示"""
    guild = ctx.guild
    roles_sorted = sorted(guild.roles, key=lambda r: r.position, reverse=True)
    lines = [f"{r.position:02d}  {r.name} : {r.id}" for r in roles_sorted]
    text = "ロール一覧（上=上位）\n```\n" + "\n".join(lines) + "\n```"

    # 万一 2000 文字超える場合は少しだけ丸める
    if len(text) > 1900:
        text = text[:1850] + "\n...（省略）\n```"

    await ctx.reply(text, allowed_mentions=discord.AllowedMentions.none())
# ---------------------------------------------------------------------------

# ---------------------------
# 自動解析（送信専用チャンネル）
# ---------------------------

@bot.event
async def on_message(message: discord.Message):
    try:
        # 自分や他Botは無視
        if message.author.bot:
            return

        # コマンドは先に処理
        if message.content.startswith("!"):
            await bot.process_commands(message)
            return

        # 対象チャンネルかつ画像が含まれているか
        if INPUT_CHANNEL_IDS and message.channel.id in INPUT_CHANNEL_IDS:
            atts = [a for a in message.attachments if _is_image_attachment(a)]
            if not atts:
                return

            # まずは同チャンネルにプレースホルダ
            placeholder = await message.channel.send("解析中…🔎")

            # 解析（画像は返さない / OCR原文は自動モードでは省略）
            _, result_text, pairs, _ = await run_pipeline_for_attachments(atts, want_image=False)

            # プレースホルダを編集（解析完了通知）
            await placeholder.edit(content=result_text)

            # ✅ 自動モードはスケジュール登録する（従来どおり）
            if pairs:
                await add_events_and_refresh_board(pairs)

            return  # ここで終了

        # その他はそのまま
        await bot.process_commands(message)

    except Exception as e:
        try:
            await message.channel.send(f"エラー: {e}")
        except Exception:
            pass

# ---------------------------
# Ping / Ready
# ---------------------------

@bot.command(name="ping")
async def ping(ctx: commands.Context):
    await ctx.reply("pong 🏓")

@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user} (tz={TIMEZONE.key})")
    # 起動直後に通知チャンネルへ「今後のスケジュール」ボードを表示（既存があれば編集）
    try:
        if NOTIFY_CHANNEL_ID:
            ch = await _get_text_channel(NOTIFY_CHANNEL_ID)
            if ch:
                await _ensure_schedule_message(ch)
    except Exception as e:
        print(f"[on_ready] ensure board failed: {e}")

    if not scheduler_tick.is_running():
        scheduler_tick.start()
    if not order_guard_tick.is_running():
        order_guard_tick.start()

# ---------------------------
# Run
# ---------------------------

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)