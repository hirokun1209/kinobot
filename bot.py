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
from PIL import ImageOps  # ← 追加（EXIFの回転を正しく適用）
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse  # ← ここでまとめて import
import uvicorn
import struct
import json
from google.cloud import vision
from google.oauth2 import service_account
from pathlib import Path
from PIL.ExifTags import TAGS
import base64
from openai import OpenAI  # ← 追加
from collections import deque
import random


# 共通の時刻パターン（HH:MM:SS / HH:MM）
TIME_HHMMSS = re.compile(r"\b(\d{1,2})[:：](\d{2})[:：](\d{2})\b")
TIME_HHMM   = re.compile(r"\b(\d{1,2})[:：](\d{2})\b")

# ← ここから追加
# [s1234] の有無・「越域」の有無・「駐騎/駐车/駐車」どれでも許容
PLACE_RE = re.compile(
    r"(?:\[\s*[sS]\d{3,4}\s*\])?\s*越域?\s*駐[騎骑车車]場\s*(\d+)"
)
# [s1234] / s1234 / 1234 を全部「1234」に正規化
SERVER_NUM_RE = re.compile(r"[sS]?\s*(\d{3,4})")

# --- (!oaiocr ヘッダ帯の3エンジン比較用 追加) ---
import unicodedata
from typing import Optional, Tuple, Dict

# 英字の紛れを数字へ寄せる（O→0, l/I→1, S→5 など）
# O→0 / lI→1 / S→5 などを数字に寄せる
_CHAR2DIGIT = str.maketrans({'O':'0','o':'0','I':'1','l':'1','S':'5','s':'5','Z':'2','z':'2','B':'8'})

def _normalize_server4(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    t = unicodedata.normalize("NFKC", str(text)).translate(_CHAR2DIGIT).lower()
    t = re.sub(r'[^0-9\[\]s-越域駐騎場]+', '', t)
    for pat in (r'\[s\s*(\d{4,5})\]', r's\D*(\d{4,5})', r'(\d{4,5})'):
        m = re.search(pat, t)
        if m:
            d = re.sub(r'\D', '', m.group(1))
            if len(d) >= 4:
                return d[:4]
    return None

def _normalize_server(x) -> str | None:
    if x is None:
        return None
    m = SERVER_NUM_RE.search(str(x))
    return m.group(1) if m else None
EXIF_DT_KEYS = ("DateTimeOriginal", "DateTimeDigitized", "DateTime")  # 優先順

POST_CROP_TOP=0.00

# === JSONの曖昧な出力を吸収する正規化 ===
def _coerce_str_lines(x) -> list[str]:
    """str / list[str] / list[dict{text|value|content|line}] / dict を全部 list[str] に揃える"""
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    out = []
    if isinstance(x, dict):
        for k in ("text", "value", "content", "line"):
            v = x.get(k)
            if isinstance(v, str) and v.strip():
                out.append(v.strip())
        return out
    if isinstance(x, (list, tuple)):
        for it in x:
            if isinstance(it, str) and it.strip():
                out.append(it.strip())
            elif isinstance(it, dict):
                for k in ("text", "value", "content", "line"):
                    v = it.get(k)
                    if isinstance(v, str) and v.strip():
                        out.append(v.strip())
                        break
    return out

def _first_str(x) -> str | None:
    """単一値を取りたいとき用の安全版"""
    ls = _coerce_str_lines(x)
    return ls[0] if ls else (x if isinstance(x, str) else None)

def _get_exif_datetime_strings(img_bytes: bytes) -> dict:
    """
    画像のEXIFから日時文字列（"YYYY:MM:DD HH:MM:SS" 等）を拾って返す。
    戻り値: {"DateTimeOriginal": "...", "DateTimeDigitized": "...", "DateTime": "..."} のうち存在するキーのみ
    """
    out = {}
    try:
        img = Image.open(io.BytesIO(img_bytes))
        exif = getattr(img, "_getexif", lambda: None)()
        if not exif:
            return out
        for tag_id, value in exif.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag in EXIF_DT_KEYS and isinstance(value, str) and value.strip():
                out[tag] = value.strip()
    except Exception:
        pass
    return out

def _parse_exif_dt_to_jst(s: str) -> str | None:
    """
    EXIFの典型書式 'YYYY:MM:DD HH:MM:SS' をJST文字列 'YYYY-MM-DD HH:MM:SS' に。
    失敗時は None を返す。
    """
    try:
        # 一部端末で 'YYYY-MM-DD HH:MM:SS' のこともあるので、':'→'-'補正は日付部だけに限定
        # 基本ケース
        if re.fullmatch(r"\d{4}:\d{2}:\d{2}\s+\d{2}:\d{2}:\d{2}", s):
            dt_naive = datetime.strptime(s, "%Y:%m:%d %H:%M:%S")
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}", s):
            dt_naive = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        else:
            return None
        # EXIFはタイムゾーン情報を持たないことが多いので「端末ローカル=JST想定」で扱う
        dt_jst = dt_naive.replace(tzinfo=JST)
        return dt_jst.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None
# --- PNGメタも見る撮影時刻推定ヘルパー ---
def _parse_str_to_jst(s: str) -> datetime | None:
    """よくある文字列日時をJST datetimeに（タイムゾーン無しはJSTとみなす）"""
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            dt = datetime.strptime(s.strip(), fmt)
            return dt.astimezone(JST) if dt.tzinfo else dt.replace(tzinfo=JST)
        except Exception:
            pass
    return None

def get_taken_time_from_image_bytes(img_bytes: bytes) -> tuple[datetime|None, str, str]:
    """
    画像バイトから撮影/作成時刻を推定。
    戻り値: (dt, how, raw)  howは取得元の説明、rawは元の文字列
    優先: EXIF(DateTimeOriginal→Digitized→DateTime) → PNG(info['timestamp'等])
    """
    # 1) EXIF
    try:
        img = Image.open(io.BytesIO(img_bytes))
        exif = getattr(img, "_getexif", lambda: None)()
        if exif:
            from PIL.ExifTags import TAGS
            tag_map = {TAGS.get(k, k): v for k, v in exif.items()}
            for key in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
                v = tag_map.get(key)
                if isinstance(v, str) and v.strip():
                    dt = _parse_str_to_jst(v)  # EXIFはTZ無し前提→JST扱い
                    if dt: return dt, f"EXIF:{key}", v
    except Exception:
        pass

    # 2) PNG/tIMEなど（Pillowは info に入ることがある）
    try:
        img = Image.open(io.BytesIO(img_bytes))
        info = getattr(img, "info", {}) or {}
        for k in ("timestamp", "creation_time", "date:create", "date:modify"):
            if k in info:
                v = info[k]
                if isinstance(v, datetime):
                    dt = v.astimezone(JST) if v.tzinfo else v.replace(tzinfo=JST)
                    return dt, f"PNG:{k}", v.isoformat(sep=" ")
                if isinstance(v, str) and v.strip():
                    dt = _parse_str_to_jst(v)
                    if dt: return dt, f"PNG:{k}", v
    except Exception:
        pass

    return None, "meta:none", ""

def base_time_from_metadata(img_bytes: bytes) -> str | None:
    """
    EXIF/PNGなどのメタ日時から、時刻部だけを HH:MM:SS で返す。
    取れないときは None。
    """
    dt, _how, _raw = get_taken_time_from_image_bytes(img_bytes)
    if dt:
        return dt.strftime("%H:%M:%S")

    # PNG tIME の素フォールバック（文字列から時刻部だけ抜く）
    t = _extract_png_time(img_bytes)
    if t:
        m = re.search(r"\b(\d{2}:\d{2}:\d{2})\b", t)
        if m:
            return m.group(1)
    return None

def _extract_png_time(raw: bytes) -> str | None:
    """
    PNGの tIME チャンク（作成/更新時刻）を生バイトから直接読む。
    返り値: 'YYYY-MM-DD HH:MM:SS' or None
    """
    sig = b"\x89PNG\r\n\x1a\n"
    if not raw.startswith(sig):
        return None
    i = len(sig)
    try:
        while i + 12 <= len(raw):
            length = struct.unpack(">I", raw[i:i+4])[0]
            ctype  = raw[i+4:i+8]
            if ctype == b"tIME" and i+8+length <= len(raw):
                data = raw[i+8:i+8+length]
                if len(data) == 7:
                    year, mon, day, hh, mm, ss = struct.unpack(">H5B", data)
                    return f"{year:04d}-{mon:02d}-{day:02d} {hh:02d}:{mm:02d}:{ss:02d}"
                return None
            i += 12 + length  # length + type(4) + crc(4)
    except Exception:
        pass
    return None

def _extract_xmp(img: Image.Image) -> dict | None:
    """
    Pillow 9.2+ なら Image.getxmp() が使える。
    返り値: dict or None
    """
    try:
        x = img.getxmp()
        return x or None
    except Exception:
        return None

def _bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# 色域（必要なら後で微調整）
HSV_BLUE_RANGES = [((90, 60, 60), (125, 255, 255))]     # 盾（青）
HSV_RED_RANGES  = [((0, 100, 80), (10, 255, 255)),
                   ((170, 100, 80), (180, 255, 255))]   # 剣（赤）

def _in_range_mask(hsv, ranges):
    m = None
    for lo, hi in ranges:
        cur = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        m = cur if m is None else (m | cur)
    return m if m is not None else np.zeros(hsv.shape[:2], np.uint8)

# Google Vision クライアント（環境変数に埋めたJSONから作成）
GV_CLIENT = None
_creds_json = os.getenv("GOOGLE_CLOUD_VISION_JSON")
if _creds_json:
    try:
        _creds = service_account.Credentials.from_service_account_info(json.loads(_creds_json))
        GV_CLIENT = vision.ImageAnnotatorClient(credentials=_creds)
        print("✅ Google Vision client ready")
    except Exception as e:
        print(f"⚠️ Vision init failed: {e}")
        
# OpenAI クライアント
OA_CLIENT = None
_openai_key = os.getenv("OPENAI_API_KEY")
if _openai_key:
    try:
        OA_CLIENT = OpenAI(api_key=_openai_key)
        print(f"✅ OpenAI client ready (OCR model: {os.getenv('OPENAI_OCR_MODEL','gpt-4o-mini')})")
    except Exception as e:
        print(f"⚠️ OpenAI init failed: {e}")
# === OpenAI Async クライアント（無ければ None で動く）===
try:
    from openai import AsyncOpenAI
    OA_ASYNC = AsyncOpenAI(api_key=_openai_key) if _openai_key else None
except Exception:
    OA_ASYNC = None

# Responses API を使えるか？
OA_SUPPORTS_RESPONSES = bool(getattr(OA_ASYNC, "responses", None))

# ---- レートリミット/RPM(60秒窓) & バックオフ設定 ----
OPENAI_RPM = int(os.getenv("OPENAI_RPM", "3"))  # 例: gpt-4o-mini は 3RPM
OPENAI_MODEL = os.getenv("OPENAI_OCR_MODEL", "gpt-4o-mini")
_oa_calls = deque()
_oa_lock = asyncio.Lock()
_oa_circuit_until = 0.0  # quota 切れ時の休止期限（event loop monotonic 秒）

def _now_mono():
    return asyncio.get_event_loop().time()

async def _ensure_openai_slot():
    """60秒窓で RPM を守る。枠が空くまで await で待機。"""
    async with _oa_lock:
        now = _now_mono()
        while _oa_calls and now - _oa_calls[0] > 60:
            _oa_calls.popleft()
        if len(_oa_calls) >= OPENAI_RPM:
            wait = 60 - (now - _oa_calls[0]) + 0.01
            await asyncio.sleep(max(0.0, wait))
            now = _now_mono()
            while _oa_calls and now - _oa_calls[0] > 60:
                _oa_calls.popleft()
        _oa_calls.append(_now_mono())

def google_ocr_from_np(np_bgr) -> list[str]:
    """
    OpenCV(BGR)画像を PNG にして Vision API へ。
    返り値は行ごと（Paddle の戻りに近づける）。
    """
    if GV_CLIENT is None:
        return []
    ok, buf = cv2.imencode(".png", np_bgr)
    if not ok:
        return []
    image = vision.Image(content=buf.tobytes())
    resp = GV_CLIENT.text_detection(image=image)
    if getattr(resp, "error", None) and resp.error.message:
        print("Vision error:", resp.error.message)
        return []
    if not resp.text_annotations:
        return []
    full = resp.text_annotations[0].description
    return [line for line in full.splitlines() if line.strip()]

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
FORM_NOTIFY_CHANNEL_ID = int(os.getenv("FORM_NOTIFY_CHANNEL_ID", str(NOTIFY_CHANNEL_ID)))
if not TOKEN:
    raise ValueError("❌ DISCORD_TOKEN が設定されていません！")
DISCORD_LOOP = None
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# =======================
# FastAPI HTTP サーバー（スリープ防止）
# =======================
app = FastAPI()

def _build_health_meta():
    return {
        "gv_ready": GV_CLIENT is not None,
        "notify_channel_id": NOTIFY_CHANNEL_ID,
        "copy_channel_id": COPY_CHANNEL_ID,
        "pre_notify_channel_id": PRE_NOTIFY_CHANNEL_ID,
        "allowed_channels": READABLE_CHANNEL_IDS,
        "ocr_lang": "japan",
        "tz": "JST(+09:00)",
    }

@app.get("/")
@app.get("/ping")
@app.get("/ping/")
async def root():
    return JSONResponse(content={"status": "ok", "meta": _build_health_meta()})

# フォーム用の GET をちゃんとルーティング
@app.get("/form", response_class=HTMLResponse)
async def upload_form():
    return """
    <html>
      <head><meta name="viewport" content="width=device-width, initial-scale=1"></head>
      <body style="font-family: system-ui; padding: 16px;">
        <h2>画像アップロード</h2>
        <form action="/upload" method="post" enctype="multipart/form-data">
          <input type="file" name="files" accept="image/*" multiple><br><br>
          <button type="submit">送信</button>
        </form>
        <p>送信すると、EXIF/PNG/XMP を解析して Discord に通知します。</p>
      </body>
    </html>
    """
import tempfile

from typing import List
from fastapi.responses import RedirectResponse

@app.post("/upload")
async def upload_image(
    background: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    for f in files:
        raw = await f.read()

        dt_meta, how, raw_str = get_taken_time_from_image_bytes(raw)
        png_time = _extract_png_time(raw)
        exif_dt_map = _get_exif_datetime_strings(raw)

        meta = {"exif_dt_map": exif_dt_map, "png_time": png_time}
        if dt_meta:
            meta["taken_guess"] = {
                "when": dt_meta.strftime("%Y-%m-%d %H:%M:%S"),
                "how": how,
                "raw": raw_str
            }

        # 既存: メタ情報をDiscordに通知
        background.add_task(
            notify_discord_upload_meta_threadsafe, f.filename, meta, FORM_NOTIFY_CHANNEL_ID
        )

        # 画像もOCR→登録（フォーム経由）
        background.add_task(
            register_from_bytes_threadsafe, raw, f.filename, FORM_NOTIFY_CHANNEL_ID
        )

    return RedirectResponse(url="/form", status_code=303)

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
# 1時間（3600秒）後に削除
GLIST_TTL = int(os.getenv("GLIST_TTL", "3600"))  # 既定 1時間

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

# 画像バイト列を Discord 添付にするだけの薄いヘルパー
def _file_from_bytes(filename: str, byts: bytes):
    return discord.File(io.BytesIO(byts), filename=filename)

def _rect_from_ratio(H, W, L, T, R, B):
    x1 = max(0, min(int(W*L), W-1))
    y1 = max(0, min(int(H*T), H-1))
    x2 = max(x1+1, min(int(W*R), W))
    y2 = max(y1+1, min(int(H*B), H))
    return (x1, y1, x2, y2)

def _crop_rect(img, rect):
    x1, y1, x2, y2 = rect
    return img[y1:y2, x1:x2]

def _pick_last_4digit(texts):
    """配列/文字列から 4桁連続数字の『最後の出現』を返す"""
    if not texts: return ""
    if isinstance(texts, str): texts = [texts]
    import re
    last = ""
    for t in texts:
        for m in re.finditer(r"(?<!\d)(\d{4})(?!\d)", t):
            last = m.group(1)
    return last

def _ocr_lines_pp_gv(img_bgr):
    """ヘッダ帯を Paddle / Google で読む（行配列を返す）"""
    lines_pp = ocr_center_paddle(img_bgr) or []
    lines_gv = extract_text_from_image_google(img_bgr) if GV_CLIENT else []
    return lines_pp, lines_gv

def _triage_read_server_from_head(head_img_bgr):
    """3エンジン（Paddle/Google/OpenAI）比較し、最後の4桁を採用"""
    pp, gv = _ocr_lines_pp_gv(head_img_bgr)
    # OpenAI 単発問い合わせ（debug用・短文プロンプト）
    oai_lines = []
    try:
        if OA_CLIENT:
            uri = _bgr_to_data_uri_np(shrink_long_side(head_img_bgr, 768))
            res = OA_CLIENT.responses.create(
                model=os.getenv("OPENAI_OCR_MODEL", "gpt-4o-mini"),
                input=[{"role":"user","content":[
                    {"type":"input_text","text":"ヘッダ帯から s#### / [s####] / #### のサーバ番号候補を列挙。1行に1つ。説明不要。"},
                    {"type":"input_image","image_url":uri,"detail":"high"},
                ]}],
                max_output_tokens=48, temperature=0
            )
            txt = (res.output_text or "").strip()
            oai_lines = [t.strip() for t in txt.splitlines() if t.strip()]
    except Exception as e:
        oai_lines = [f"(OpenAI error: {e})"]

    # 正規化（sや[]を落として4桁取り）
    cand_pp  = _pick_last_4digit(pp)
    cand_gv  = _pick_last_4digit(gv)
    cand_oai = _pick_last_4digit(oai_lines)

    # 優先度：OpenAI → Google → Paddle
    final = cand_oai or cand_gv or cand_pp

    dbg = {"raw":{"pp":pp,"gv":gv,"oai":oai_lines},
           "norm":{"pp":cand_pp,"gv":cand_gv,"oai":cand_oai},
           "winner":final}
    return final, dbg

def _calc_regions(full_bgr):
    """フル画像基準で head / clock / center / cease を一括決定"""
    H, W = full_bgr.shape[:2]
    head   = _rect_from_ratio(H, W, 0.0, HEAD_TOP_RATIO,   HEAD_RIGHT_RATIO,   HEAD_BOTTOM_RATIO)
    clock  = _rect_from_ratio(H, W, CLOCK_LEFT_RATIO,      CLOCK_TOP_RATIO,    CLOCK_RIGHT_RATIO, CLOCK_BOTTOM_RATIO)
    center = _rect_from_ratio(H, W, 0.0, CENTER_TOP_RATIO, CLOCK_LEFT_RATIO,   CENTER_BOTTOM_RATIO)  # 右端＝時計左端
    cease  = _rect_from_ratio(H, W, 0.0, CEASE_TOP_RATIO,  1.0,                CEASE_BOTTOM_RATIO)
    return {"head": head, "clock": clock, "center": center, "cease": cease}

# --- 合成PNG(bytes) → BGR(ndarray) ---
def _bgr_from_png_bytes(png_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(png_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

# --- 合成画像内のヘッダ帯の割合（必要なら環境変数で微調整） ---
COMP_HEAD_TOP    = float(os.getenv("COMP_HEAD_TOP", "0.00"))
COMP_HEAD_BOTTOM = float(os.getenv("COMP_HEAD_BOTTOM", "0.25"))
COMP_HEAD_RIGHT  = float(os.getenv("COMP_HEAD_RIGHT", "1.00"))

# --- スイッチ：合成からヘッダを読む（1でON / 既定OFF） ---
USE_COMPOSITE_FOR_HEADER = os.getenv("USE_COMPOSITE_FOR_HEADER", "1") == "1"

def cleanup_old_entries():
    now = now_jst()
    for k in list(pending_places):
        if (now - pending_places[k]["created_at"]) > timedelta(hours=6):
            del pending_places[k]

HHMMSS_RE   = r'(?<!\d)(?:[01]?\d|2[0-3]):[0-5]\d:[0-5]\d(?!\d)'
DATE_LIKE_RE= r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}'

def _read_clock_from_roi_strict(full_bgr, clock_rect=None):
    """右上時計のROIだけから HH:MM:SS を抽出。日付や『停戦』混じりは排除。"""
    roi = _crop_rect(full_bgr, clock_rect) if clock_rect else full_bgr
    pp = ocr_center_paddle(roi) or []
    gv = extract_text_from_image(roi) or []
    texts = []
    for s in (pp if isinstance(pp, list) else [pp]) + (gv if isinstance(gv, list) else [gv]):
        if not s: 
            continue
        t = str(s)
        if "停戦" in t or re.search(DATE_LIKE_RE, t):
            continue
        texts.append(t)
    for t in texts:
        m = re.search(HHMMSS_RE, t)
        if m:
            return m.group(0)
    return None

def parse_txt_fields(txt: str):
    m = re.fullmatch(r"(奪取|警備)\s+(\d{4})-(\d+)-(\d{2}:\d{2}:\d{2})", txt)
    return m.groups() if m else None

def _ocr_lines_pp_gv(bgr: np.ndarray):
    """Paddle + Google で本文行を拾って統合（重複除去）"""
    pp = ocr_center_paddle(bgr) or []
    if isinstance(pp, str): pp = [pp]
    gv = extract_text_from_image(bgr) or []
    if isinstance(gv, str): gv = [gv]
    merged = []
    for arr in (pp, gv):
        for s in arr:
            if s and s not in merged:
                merged.append(s)
    return merged

def extract_places_from_center(center_txts):
    """中央本文から『越域 駐騎/駐車場 N』の N を列挙して返す（駐騎/駐車 両対応）"""
    text = "\n".join(center_txts or [])
    nums = re.findall(r'越域\s*駐[騎骑車车]場\s*([0-9]{1,2})', text)
    return [int(n) for n in nums]

# === 低コスト化ヘルパー（追加） ===
def shrink_long_side(bgr: np.ndarray, max_side: int = 768) -> np.ndarray:
    """長辺を max_side に縮小（総ピクセル数を減らす）"""
    h, w = bgr.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return bgr
    r = max_side / s
    return cv2.resize(bgr, (int(w*r), int(h*r)), interpolation=cv2.INTER_AREA)

def _pick_last_server_from_lines(lines) -> Optional[str]:
    """
    OCR行から [s1234]/s1234/1234/s268 などを全部拾い、
    『一番後ろ』を返す。4桁があれば4桁を優先。
    """
    if not lines:
        return None
    if isinstance(lines, str):
        lines = [lines]
    found: list[str] = []
    for t in lines:
        s = unicodedata.normalize("NFKC", str(t))
        for m in re.finditer(r"\[?\s*[sS]?\s*(\d{3,5})\s*\]?", s):
            found.append(m.group(1))
    if not found:
        return None
    for d in reversed(found):
        if len(d) == 4:
            return d
    return found[-1]

# “英字っぽい紛れ”を数字へ寄せて4桁へ収束させる軽い正規化（必要箇所で使用）
_CHAR2DIGIT = str.maketrans({'O':'0','o':'0','I':'1','l':'1','S':'5','s':'5','Z':'2','z':'2','B':'8'})
def _normalize_server4(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    t = unicodedata.normalize("NFKC", str(text)).translate(_CHAR2DIGIT).lower()
    t = re.sub(r'[^0-9\[\]s-越域駐騎場]+', '', t)
    for pat in (r'\[s\s*(\d{4,5})\]', r's\D*(\d{4,5})', r'(\d{4,5})'):
        m = re.search(pat, t)
        if m:
            d = re.sub(r'\D', '', m.group(1))
            if len(d) >= 4:
                return d[:4]
    return None

def _triage_read_server_from_head(head_bgr: np.ndarray) -> Tuple[Optional[str], dict]:
    pp_lines = ocr_center_paddle(head_bgr) or []
    if isinstance(pp_lines, str): pp_lines = [pp_lines]
    gv_lines = extract_text_from_image(head_bgr) or []
    if isinstance(gv_lines, str): gv_lines = [gv_lines]
    oai_prompt = ("Read server id like [s1234] in header. Return ONLY the four digits; if unsure, return NONE.")
    try:
        oai_text = call_openai_vision(head_bgr, prompt=oai_prompt) or ""
    except Exception:
        oai_text = ""

    cand_raw = {"pp": "\n".join(pp_lines), "gv": "\n".join(gv_lines), "oai": oai_text}

    def _prefer_last(lines):
        lines = list(lines or [])
        # キーワード行を優先（越域/駐騎/駐車）
        for ln in lines:
            ln2 = unicodedata.normalize("NFKC", str(ln))
            if ("越域" in ln2) or ("駐騎" in ln2) or ("駐車" in ln2):
                n = _pick_last_server_from_lines([ln2])
                if n: return n
        # [s を含む行を次点
        for ln in lines:
            if "[s" in str(ln):
                n = _pick_last_server_from_lines([ln])
                if n: return n
        # 全行で最後
        return _pick_last_server_from_lines(lines)

    pick_pp  = _prefer_last(pp_lines)
    pick_gv  = _prefer_last(gv_lines)
    pick_oai = _normalize_server4(oai_text)

    cand_norm = {"pp": pick_pp, "gv": pick_gv, "oai": pick_oai}
    if cand_norm.get("oai"):  # OAIが4桁なら最優先
        return cand_norm["oai"], {"raw": cand_raw, "norm": cand_norm, "winner": "oai"}

    votes = {}
    for n in (pick_pp, pick_gv, pick_oai):
        if n: votes[n] = votes.get(n, 0) + 1
    if votes:
        winner_val = max(votes.items(), key=lambda kv: kv[1])[0]
        return winner_val, {"raw": cand_raw, "norm": cand_norm, "winner": "vote"}
    return None, {"raw": cand_raw, "norm": cand_norm, "winner": None}

# === SRVDEBUG: 可視化ユーティリティ ===
def _draw_box(img_bgr, rect, color=(0,255,255), thick=2, label=None):
    x1,y1,x2,y2 = rect
    cv2.rectangle(img_bgr, (x1,y1), (x2,y2), color, thick)
    if label:
        cv2.putText(img_bgr, label, (x1+6, max(14,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

# 任意で環境調整できる比率（なければ既定値）
CLOCK_TOP_RATIO    = float(os.getenv("CLOCK_TOP_RATIO", "0.20"))  # １ブロック下げ版
CLOCK_BOTTOM_RATIO = float(os.getenv("CLOCK_BOTTOM_RATIO", "0.38"))
CEASE_FALLBACK_TOP = float(os.getenv("CEASE_FALLBACK_TOP", "0.85"))
# CLOCK の左右を環境変数で調整可能に
CLOCK_LEFT_RATIO  = float(os.getenv("CLOCK_LEFT_RATIO",  "0.72"))
CLOCK_RIGHT_RATIO = float(os.getenv("CLOCK_RIGHT_RATIO", "0.98"))

# === Region ratios (0.0 - 1.0) ===
HEAD_TOP_RATIO    = 0.00
HEAD_BOTTOM_RATIO = 0.25
HEAD_RIGHT_RATIO  = 1.00

# ⬇ 右上時計のトリミング領域（上に持ち上げ）
# 以前より TOP を小さく、BOTTOM も同じぶん上げて高さはほぼ同じ
CLOCK_TOP_RATIO    = 0.02   # ← 上に上げる
CLOCK_BOTTOM_RATIO = 0.11   # ← 旧値が 0.16 前後なら 0.12 に（高さ ~0.10 を維持）
CLOCK_LEFT_RATIO   = 0.74
CLOCK_RIGHT_RATIO  = 0.98

# 中央リスト（必要なら既存値のままでOK）
CENTER_TOP_RATIO    = 0.25
CENTER_BOTTOM_RATIO = 0.78
CENTER_LEFT_RATIO   = 0.05
CENTER_RIGHT_RATIO  = 0.90  # ※下の関数で時計の左端までクランプします

# 停戦バナー
CEASE_TOP_RATIO    = 0.86
CEASE_BOTTOM_RATIO = 0.98


# 中央（青）：上＝緑の下端、下＝紫の上端、右端＝時計の左端
CENTER_TOP_RATIO    = HEAD_BOTTOM_RATIO
CENTER_BOTTOM_RATIO = CEASE_TOP_RATIO

# 合成画像に対する「ヘッダ帯」の比率（srvdebug と同じ）
COMP_HEAD_TOP    = 0.00
COMP_HEAD_BOTTOM = 0.25
COMP_HEAD_RIGHT  = 1.00

# 合成画像からヘッダを読むか（デフォルト=ON）
USE_COMPOSITE_FOR_HEADER = (os.getenv("USE_COMPOSITE_FOR_HEADER", "1") == "1")

def _mark_regions_on_full(full_bgr: np.ndarray):
    """フル画像座標系で領域を決めて可視化用画像と各矩形を返す"""
    H, W = full_bgr.shape[:2]

    # 各矩形（x1, y1, x2, y2）
    head_rect = (
        0,
        int(H * HEAD_TOP_RATIO),
        int(W * HEAD_RIGHT_RATIO),
        int(H * HEAD_BOTTOM_RATIO),
    )
    clock_rect = (
        int(W * CLOCK_LEFT_RATIO),
        int(H * CLOCK_TOP_RATIO),
        int(W * CLOCK_RIGHT_RATIO),
        int(H * CLOCK_BOTTOM_RATIO),
    )
    center_rect = (
        int(W * CENTER_LEFT_RATIO),
        int(H * CENTER_TOP_RATIO),
        int(W * CENTER_RIGHT_RATIO),
        int(H * CENTER_BOTTOM_RATIO),
    )
    # 中央の右端は時計の左端を超えないようにクランプ（右側の余計なパネルを避ける）
    center_rect = (center_rect[0], center_rect[1], min(center_rect[2], clock_rect[0]), center_rect[3])

    cease_rect = (
        0,
        int(H * CEASE_TOP_RATIO),
        W,
        int(H * CEASE_BOTTOM_RATIO),
    )

    # デバッグ描画
    dbg = full_bgr.copy()
    _draw_box(dbg, head_rect,  (0, 255,   0), 2, "HEAD")
    _draw_box(dbg, clock_rect, (0, 255, 255), 2, "CLOCK")   # 黄色
    _draw_box(dbg, center_rect,(255,  0,   0), 2, "CENTER")
    _draw_box(dbg, cease_rect, (255,  0, 255), 2, "CEASE")

    return dbg, head_rect, clock_rect, center_rect, cease_rect
    
def percent_crop(bgr: np.ndarray, l=0.0, t=0.0, r=0.0, b=0.0) -> np.ndarray:
    """左右上下を割合でトリミング"""
    h, w = bgr.shape[:2]
    x1 = int(w * l); y1 = int(h * t)
    x2 = w - int(w * r); y2 = h - int(h * b)
    x1 = max(0, min(x1, w-1)); y1 = max(0, min(y1, h-1))
    x2 = max(x1+1, min(x2, w)); y2 = max(y1+1, min(y2, h))
    return bgr[y1:y2, x1:x2]

def find_black_bands_rows(bgr: np.ndarray,
                          thr: int | None = None,
                          ratio: float | None = None,
                          min_h: int | None = None) -> list[tuple[int,int]]:
    """横一帯の黒塗り（行）を検出して [(y1,y2), ...] を返す"""
    # ← デフォルトはここで評価（定数は後で定義されていてもOKになる）
    if thr   is None:  thr   = BLACK_ROW_LUMA_THR
    if ratio is None:  ratio = BLACK_ROW_RATIO
    if min_h is None:  min_h = BLACK_MIN_H

    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    row_black = (g < thr).sum(axis=1) >= int(w * ratio)
    bands = []
    i = 0
    while i < h:
        if row_black[i]:
            j = i
            while j < h and row_black[j]:
                j += 1
            if (j - i) >= min_h:
                bands.append((i, j))
            i = j
        else:
            i += 1
    return bands

def compact_black_rows_and_cut_tail(bgr: np.ndarray) -> np.ndarray:
    """
    黒帯行は丸ごと削除して上に詰める。
    さらに『最後の黒帯の終端』より下は全部削除。
    """
    bands = find_black_bands_rows(bgr)
    if not bands:
        return bgr
    h = bgr.shape[0]
    last_end = bands[-1][1]
    keep = np.ones(h, dtype=bool)
    for y1, y2 in bands:
        keep[y1:y2] = False
    if DELETE_BELOW_LAST:
        keep[last_end:] = False
    return bgr[keep, :, :]

# ===== 1画像ルート：合成 → OpenAI OCR =====

SINGLEIMG_MAX_SIDE = int(os.getenv("SINGLEIMG_MAX_SIDE", "1280"))  # 送信用に縮小してトークン節約
CLOCK_SCALE_W      = float(os.getenv("CLOCK_SCALE_W", "0.32"))    # 右上時計の横幅(ベース比)
MARGIN_PX          = int(os.getenv("COMPOSE_MARGIN_PX", "10"))    # 貼り付けマージンpx
# ===== 合成オプション（前処理） =====
ATTACH_CEASE        = os.getenv("ATTACH_CEASE", "1") == "1"    # 「停戦終了」を下に接続（被せない）
COMPACT_BLACK       = os.getenv("COMPACT_BLACK", "1") == "1"   # 黒帯を詰める
DELETE_BELOW_LAST   = os.getenv("DELETE_BELOW_LAST", "1") == "1"  # 最下黒帯より下を削除

# 黒帯（横一帯）検出のしきい値
BLACK_ROW_LUMA_THR  = int(os.getenv("BLACK_ROW_LUMA_THR",  "22"))   # 0–255 この明度以下を黒扱い
BLACK_ROW_RATIO     = float(os.getenv("BLACK_ROW_RATIO",   "0.92"))  # 行内の黒ピクセル比がこれを超えたら黒行
BLACK_MIN_H         = int(os.getenv("BLACK_MIN_H", "10"))            # 黒帯最小高さ(px)

# 仕上げのパーセントトリミング（左20%・上15%）
POST_CROP_LEFT      = float(os.getenv("POST_CROP_LEFT",  "0.20"))
POST_CROP_TOP       = float(os.getenv("POST_CROP_TOP",   "0.00"))
POST_CROP_RIGHT     = float(os.getenv("POST_CROP_RIGHT", "0.00"))
POST_CROP_BOTTOM    = float(os.getenv("POST_CROP_BOTTOM","0.00"))

# ---- SRVDEBUG用：ヘッダ帯の想定範囲（端末により微調整）----
HEAD_TOP_RATIO    = float(os.getenv("HEAD_TOP_RATIO", "0.00"))  # 画面高さの上から何割〜
HEAD_BOTTOM_RATIO = float(os.getenv("HEAD_BOTTOM_RATIO", "0.25"))  # 〜下まで何割（初期: 上18%）
HEAD_RIGHT_RATIO  = float(os.getenv("HEAD_RIGHT_RATIO", "1.00"))  # 右端は少し余らせる
# ニュース帯を避ける“内側”クロップ（!srvdebug / !oaiocr 共通）
HEAD_INNER_TOP    = float(os.getenv("HEAD_INNER_TOP", "0.12"))
HEAD_INNER_BOTTOM = float(os.getenv("HEAD_INNER_BOTTOM", "0.28"))

# ヘッダ単体もOpenAIに投げて比較するか（True推奨）
DEBUG_ATTACH_TO_OPENAI = os.getenv("SRVDEBUG_ATTACH_OAI", "1") == "1"

def compose_center_with_clock_and_cease(bgr_full: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    1) 「免戦中」の直下を黒塗り（既存）
    2) 中央をベースに切り出し
    3) 黒帯を詰めて最下黒帯より下を削除
    4) 左20%・上15%をトリミング
    5) 「停戦終了」を“下に接続”（被せない）
    6) 右上の時計を右上へ貼り付け
    """
    # 免戦中の横帯は先に黒塗り（誤読抑止）
    bgr_masked, _ = auto_mask_ime(bgr_full)

    # 各パーツを切り出し
    center = crop_center_area(bgr_masked).copy()
    clock  = crop_top_right(bgr_masked)

    # 停戦終了帯（検出→無ければフォールバック）
    rects = find_ceasefire_regions_full_img(bgr_masked)
    if rects:
        x1,y1,x2,y2 = rects[0]
        cease = bgr_masked[y1:y2, x1:x2]
    else:
        cease = crop_cease_banner(bgr_masked)

    # 黒帯を詰める＋最下黒帯より下を削除
    if COMPACT_BLACK:
        center = compact_black_rows_and_cut_tail(center)

    # 左20%・上15%トリミング（値は環境変数で調整可）
#    if any([POST_CROP_LEFT, POST_CROP_RIGHT, POST_CROP_BOTTOM]):
        # 上(Top)は絶対に削らない
#        center = percent_crop(center, POST_CROP_LEFT, 0.0, POST_CROP_RIGHT, POST_CROP_BOTTOM)

    base = center

    # 「停戦終了」を“下に接続”（被せない）
    if ATTACH_CEASE and cease is not None and cease.size:
        target_w = base.shape[1]
        ratio = target_w / max(1, cease.shape[1])
        csz   = cv2.resize(cease, (target_w, max(12, int(cease.shape[0]*ratio))), interpolation=cv2.INTER_AREA)
        spacer = np.zeros((6, target_w, 3), dtype=np.uint8)  # 少しだけ間隔
        base   = np.vstack([base, spacer, csz])

    # 右上の時計を貼り付け（従来通り）
    H, W = base.shape[:2]
    if clock is not None and clock.size:
        tw = min(int(W * CLOCK_SCALE_W), clock.shape[1])
        th = max(12, int(clock.shape[0] * (tw / clock.shape[1])))
        clk = cv2.resize(clock, (tw, th), interpolation=cv2.INTER_AREA)
        y = max(MARGIN_PX, 2)
        x = W - tw - MARGIN_PX
        base[y:y+th, x:x+tw] = clk

    return base, {"base_center": base, "clock": clock, "cease": cease}

async def oai_ocr_oneimg_async(full_bgr: np.ndarray) -> dict | None:
    """
    合成1枚だけを OpenAI に送り、既存と互換の JSON を返す。
    返り値例:
      {"top_clock_lines":["22:18:42"], "center_lines":["[s1275]...", "..."], "ceasefire_end":"02:07:52", "structured":{...}}
    """
    if OA_ASYNC is None:
        return None

    comp_bgr, dbg = compose_center_with_clock_and_cease(full_bgr)

    # 送信用に縮小
    comp_small = shrink_long_side(comp_bgr, SINGLEIMG_MAX_SIDE)

    # Data URI 化
    png_bytes, data_uri = _bgr_to_png_bytes_and_data_uri(comp_small)
    if not data_uri:
        return None

    # 出力最小化のため指示は短く・JSONのみ
    instruction = (
        '{"top_clock_lines":[],"center_lines":[],"ceasefire_end":null,'
        '"structured":{"server":"","rows":[{"place":0,"status":"免戦中","duration":"00:00:00"}]}}'
        ' 上のJSONだけ返す。'
        ' 右上はゲーム内時計、中央は一覧、最下部は「停戦終了 HH:MM:SS」。'
        ' 配列要素は必ず文字列（str）にし、オブジェクトを入れないこと。'
        ' 数字/コロンは正規化して。'
    )

    # oai_ocr_oneimg_async 内：送信コンテンツの detail を high に
    content = [
        {"type": "input_text",  "text": instruction},
        {"type": "input_image", "image_url": data_uri, "detail": "high"},  # low→high
    ]

    await _ensure_openai_slot()
    try:
        if OA_SUPPORTS_RESPONSES:
            res = await OA_ASYNC.responses.create(
                model=OPENAI_MODEL,
                input=[{"role":"user","content":content}],
                max_output_tokens=200,
                temperature=0
            )
            txt = (res.output_text or "").strip()
        else:
            res = await OA_ASYNC.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role":"user","content":[
                    {"type":"text","text":instruction},
                    {"type":"image_url","image_url":{"url":data_uri,"detail":"high"}},  # ← low→high
                ]}],
                max_tokens=200,
                temperature=0
            )
            txt = (res.choices[0].message.content or "").strip()

        txt = _strip_code_fences(txt)
        out = json.loads(txt) if txt.startswith("{") else None
        if out is not None:
            # デバッグに使えるよう、送った合成PNGも同梱（任意）
            out["_echo"] = {"composite_png": png_bytes}
        return out
    except Exception as e:
        print(f"[OpenAI oneimg OCR] error: {e}")
        return None

def crop_cease_banner(bgr: np.ndarray) -> np.ndarray | None:
    """
    『停戦終了 HH:MM:SS』の帯あたりを大雑把に切り抜く。
    端末差がある場合は比率を微調整してください。
    """
    h, w = bgr.shape[:2]
    y1, y2 = int(h*0.25), int(h*0.30)
    x1, x2 = int(w*0.12), int(w*0.88)
    if y2 - y1 < 12 or x2 - x1 < 12:
        return None
    return bgr[y1:y2, x1:x2]

def choose_base_time(img_bytes: bytes) -> tuple[str|None, str]:
    """
    戻り値: (HH:MM:SS または None, "meta"|"ocr"|"none")
    メタ(EXIF/PNG/XMP)が取れなければ右上OCRにフォールバック。
    """
    dt_meta, _how, _raw = get_taken_time_from_image_bytes(img_bytes)
    if dt_meta:
        return dt_meta.strftime("%H:%M:%S"), "meta"
    dt_ocr, _raw_ocr = _ocr_clock_topright_to_jst(img_bytes)
    if dt_ocr:
        return dt_ocr.strftime("%H:%M:%S"), "ocr"
    return None, "none"

# 停戦終了での自動補正：許容誤差（既定1秒）
CEASEFIX_MAX_SEC = int(os.getenv("CEASEFIX_MAX_SEC", "1"))

# 免戦の最大想定時間（h）: これより長い時間は誤読として弾く
IMSEN_MAX_HOURS = int(os.getenv("IMSEN_MAX_HOURS", "5"))

def _sec_from_hhmmss(s: str) -> int:
    try:
        h, m, se = map(int, s.split(":"))
        return h*3600 + m*60 + se
    except Exception:
        return 10**9  # 大きめ

def pick_duration_from_group(lines: list[str]) -> str | None:
    """
    与えられた行グループから免戦時間を1つ選ぶ規則:
      1) '免戦中' を含む行の近傍（±2行）にある HH:MM:SS を最優先
      2) それでも見つからなければ、グループ内の HH:MM:SS のうち最小を採用
    ただし、h > IMSEN_MAX_HOURS は誤読として除外（例: 11:00:06 など）
    """
    fixed = [normalize_time_separators(x) for x in lines]
    im_idx = [i for i, t in enumerate(fixed) if "免戦中" in t]

    # グループ内の全候補 (行番号, "HH:MM:SS")
    cand: list[tuple[int, str]] = []
    for i, t in enumerate(fixed):
        s = force_hhmmss_if_six_digits(t)
        for m in list(TIME_HHMMSS.finditer(s)) + list(TIME_HHMM.finditer(s)):
            s = correct_imsen_text(m.group(0))
            if not s:
                continue
            try:
                h = int(s.split(":")[0])
            except:
                continue
            if h <= IMSEN_MAX_HOURS:
                cand.append((i, s))

    if not cand:
        return None

    # 免戦中の近傍（±2行）を最優先
    if im_idx:
        near: list[tuple[int, int, str]] = []  # (距離, 秒, 文字列)
        for i, s in cand:
            dist = min(abs(i - k) for k in im_idx)
            if dist <= 2:
                near.append((dist, _sec_from_hhmmss(s), s))
        if near:
            near.sort()
            return near[0][2]

    # フォールバック：一番短い時間
    return min(cand, key=lambda x: _sec_from_hhmmss(x[1]))[1]

def _extract_clock_from_top_txts(txts) -> str | None:
    """右上時計OCR（何型でもOK）から HH:MM:SS（なければ HH:MM→秒0）を返す"""
    for s in _coerce_str_lines(txts):
        s = force_hhmmss_if_six_digits(normalize_time_separators(s.strip()))
        m = TIME_HHMMSS.search(s) or TIME_HHMM.search(s)
        if m:
            h  = int(m.group(1))
            mi = int(m.group(2))
            se = int(m.group(3)) if m.re is TIME_HHMMSS else 0
            if 0 <= h < 24 and 0 <= mi < 60 and 0 <= se < 60:
                return f"{h:02}:{mi:02}:{se:02}"
    return None

def _parse_hhmmss_to_dt_jst(timestr: str) -> datetime | None:
    """'HH:MM:SS' もしくは 'HH:MM' を今日の日付の JST datetime に。深夜(〜05:59)は翌日扱い。"""
    s = force_hhmmss_if_six_digits(normalize_time_separators(timestr or ""))
    m = TIME_HHMMSS.search(s) or TIME_HHMM.search(s)
    if not m:
        return None
    h  = int(m.group(1))
    mi = int(m.group(2))
    se = int(m.group(3)) if m.re is TIME_HHMMSS else 0
    if not (0 <= h < 24 and 0 <= mi < 60 and 0 <= se < 60):
        return None
    base = datetime.combine(now_jst().date(), time(h, mi, se), tzinfo=JST)
    if base.time() < time(6, 0, 0):
        base += timedelta(days=1)
    return base

def _bgr_to_png_base64(np_bgr: np.ndarray) -> tuple[str|None, int]:
    ok, buf = cv2.imencode(".png", np_bgr)
    if not ok:
        return None, 0
    b = buf.tobytes()
    return base64.b64encode(b).decode("utf-8"), len(b)

def oai_ocr_lines(np_bgr: np.ndarray, purpose: str = "general") -> list[str]:
    """
    OpenAI (Responses API) でOCR。
    - 画像は Data URI を image_url に「文字列」で渡す
    - 原寸と2xシャープの2パス
    - gpt-5-mini → gpt-4o-mini の順でフォールバック
    """
    if OA_CLIENT is None:
        return []

    def _upsample_and_sharpen(img_bgr: np.ndarray) -> np.ndarray:
        up = cv2.resize(img_bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(up, (0, 0), 1.0)
        return cv2.addWeighted(up, 1.6, blur, -0.6, 0)

    def _bgr_to_data_uri(img_bgr: np.ndarray) -> str | None:
        ok, buf = cv2.imencode(".png", img_bgr)
        if not ok:
            return None
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    model_chain = [os.getenv("OPENAI_OCR_MODEL", "gpt-4o-mini")]
    variants = [np_bgr]

    user_text = (
        "画像から見える文字を行単位で抽出して返してください。時間は 05:00:15 / 55:12 のようにコロン区切り。説明は不要。"
        if purpose != "clock"
        else "画像の時計の時刻だけを抽出。可能なら HH:MM:SS を1行で返して。説明は不要。"
    )

    outputs: list[str] = []

    for var in variants:
        var_small = shrink_long_side(var, 768)
        data_uri = _bgr_to_data_uri(var_small)
        if not data_uri:
            continue

        content = [
            {"type": "input_text", "text": f"[目的:{purpose}] {user_text}"},
            {"type": "input_image", "image_url": data_uri, "detail": "high"},
        ]

        for model_name in model_chain:
            try:
                res = OA_CLIENT.responses.create(
                    model=model_name,
                    input=[{"role": "user", "content": content}],
                    max_output_tokens=64,
                )
                txt = (res.output_text or "").strip()
                if txt:
                    outputs.extend([t.strip() for t in txt.splitlines() if t.strip()])
                break  # このモデルで取れたら次のモデルへは行かない
            except Exception as e:
                print(f"[OpenAI OCR] {model_name} error: {e}")

    # 正規化＋重複排除
    out, seen = [], set()
    for t in outputs:
        t2 = normalize_time_separators(t)
        t2 = force_hhmmss_if_six_digits(t2)
        if t2 and t2 not in seen:
            seen.add(t2)
            out.append(t2)
    return out

def _bgr_to_data_uri_np(bgr: np.ndarray) -> str | None:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return None
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")

def _bgr_to_png_bytes_and_data_uri(img_bgr: np.ndarray) -> tuple[bytes | None, str | None]:
    """同じエンコード結果を OpenAI送信用(URI) と Discord添付用(bytes)の両方で返す"""
    if img_bgr is None:
        return None, None
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        return None, None
    by = buf.tobytes()
    uri = "data:image/png;base64," + base64.b64encode(by).decode("utf-8")
    return by, uri

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

# === oaiocr テキストレポート ===
def _fmt_block(lines: list[str] | None, empty="(なし)", max_lines=50) -> str:
    ls = (lines or [])[:max_lines]
    body = "\n".join(ls) if ls else empty
    return f"```\n{body}\n```"

def _only_txt_from_parsed(parsed: list[tuple]) -> list[str]:
    # parsed は [(dt, txt, raw_dur), ...]
    return [t for _, t, _ in parsed] if parsed else []

async def _send_oaiocr_text_report(
    ch,
    top_txts: list[str],
    center_txts: list[str],
    base_clock_str: str | None,
    cease_hhmmss: str | None,
    parsed_preview: list[tuple],
    parsed_final: list[tuple],
    durations: list[str],
    cease_fix_applied_sec: int = 0,
    cease_fix_threshold_sec: int | None = None,
    files: list[discord.File] | None = None,   # ← 追加
):
    thresh = cease_fix_threshold_sec if cease_fix_threshold_sec is not None else 0

    # 停戦終了の表示（例のフォーマット：YYYY:MM:DD HH:MM:SS）
    if cease_hhmmss:
        today = now_jst().strftime("%Y:%m:%d")
        cease_label = f"{today} {cease_hhmmss}"
    else:
        cease_label = "-"

    lines = []
    lines.append("🤖 **OpenAI OCR（1画像）**")

    lines.append("📸 上部（時計）:")
    lines.append(_fmt_block(top_txts or ["(検出なし)"]))

    lines.append("🧩 中央（本文）:")
    lines.append(_fmt_block(center_txts or ["(検出なし)"]))

    lines.append(f"🕒 基準(右上時計): `{base_clock_str or '---'}`")

    lines.append(f"🛡 停戦終了: `{cease_label}` / 自動補正: ±{abs(cease_fix_applied_sec)}秒（閾値±{thresh}s）")

    lines.append("📋 **補正前の予定プレビュー**:")
    lines.append(_fmt_block(_only_txt_from_parsed(parsed_preview)))

    lines.append("🧾 **最終出力（登録される行）**:")
    lines.append(_fmt_block(_only_txt_from_parsed(parsed_final)))

    lines.append("⏳ 免戦時間候補:")
    lines.append(_fmt_block(durations))

    await ch.send("\n".join(lines), files=files if files else None)

async def oai_ocr_all_in_one_async(top_bgr: np.ndarray, center_bgr: np.ndarray, full_bgr: np.ndarray | None = None) -> dict | None:
    """
    単一リクエストで 2〜3領域をOCRし、JSONで返す:
      {
        "top_clock_lines":[...],
        "center_lines":[...],
        "ceasefire_end":"HH:MM:SS" | null,
        "structured":{"server":"s####","rows":[{"place":<int>,"status":"免戦中","duration":"HH:MM:SS"}]}
      }
    """
    if OA_ASYNC is None:
        return None

    now = _now_mono()
    if now < _oa_circuit_until:
        return None  # quota休止中

    # 送る前に縮小（画素数＝コスト）
    top_small    = shrink_long_side(top_bgr,    640)
    center_small = shrink_long_side(center_bgr, 768)

    # oai_ocr_all_in_one_async 内の cease_small 作成部をこの形に
    cease_small = None
    # 修正後（ズラしを廃止 ＋ フォールバックを追加）:
    if full_bgr is not None:
        rects = find_ceasefire_regions_full_img(
            full_bgr,
            y_shift_ratio=0.70,
            pad_bottom=100,
            pad_x=CEASE_PAD_X,
            pad_top=CEASE_PAD_TOP,
        )
        if rects:
            x1, y1, x2, y2 = rects[0]
            band = full_bgr[y1:y2, x1:x2]
            cease_small = shrink_long_side(band, 512)
        else:
            # 検出できない端末向けに大ざっぱフォールバック
            fb = crop_cease_banner(full_bgr)
            if fb is not None:
                cease_small = shrink_long_side(fb, 512)


    # ★ ここを置換：OpenAI用の data_uri と Discord用の png bytes を同時に作成
    top_png,    img1 = _bgr_to_png_bytes_and_data_uri(top_small)
    center_png, img2 = _bgr_to_png_bytes_and_data_uri(center_small)
    cease_png,  img3 = (None, None)
    if cease_small is not None:
        cease_png, img3 = _bgr_to_png_bytes_and_data_uri(cease_small)

    if not img1 or not img2:
        return None

    # ⬇ 説明を短く（入力トークンも節約）
    instruction = (
        '{"top_clock_lines":[],"center_lines":[],"ceasefire_end":null,'
        '"structured":{"server":"","rows":[{"place":0,"status":"免戦中","duration":"00:00:00"}]}}'
        ' 以上のJSONだけを返す。数値とコロンは正規化。'
        ' 帯から「停戦終了 HH:MM:SS」を読めたら ceasefire_end に入れる。'
        ' 配列要素は必ず文字列（str）にし、オブジェクトを入れないこと。'
    )

    content_responses = [
        {"type": "input_text", "text": instruction},
        {"type": "input_image", "image_url": img1, "detail": "high"},
        {"type": "input_image", "image_url": img2, "detail": "high"},
    ]
    if img3:
        content_responses.append({"type": "input_image", "image_url": img3, "detail": "high"})

    content_chat = [
        {"type": "text", "text": instruction},
        {"type": "image_url", "image_url": {"url": img1, "detail": "high"}},  # ← low→high
        {"type": "image_url", "image_url": {"url": img2, "detail": "high"}},  # ← low→high
    ]
    if img3:
        content_chat.append({"type": "image_url", "image_url": {"url": img3, "detail": "high"}})  # ← low→high

    backoff = 1.0
    for _ in range(5):
        try:
            await _ensure_openai_slot()
            if OA_SUPPORTS_RESPONSES:
                res = await OA_ASYNC.responses.create(
                    model=OPENAI_MODEL,
                    input=[{"role": "user", "content": content_responses}],
                    # ⬇ 出力上限も小さく（JSONだけ）
                    max_output_tokens=200,
                )
                txt = (res.output_text or "").strip()
            else:
                res = await OA_ASYNC.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": content_chat}],
                    max_tokens=200,
                )
                txt = (res.choices[0].message.content or "").strip()

            txt = _strip_code_fences(txt)
            if not txt or not txt.startswith("{"):
                return None
            out = json.loads(txt)

            # ★ 追加：OpenAI に送った PNG bytes を同梱（ローカル利用用）
            out["_echo"] = {
                "top_png": top_png,
                "center_png": center_png,
            }
            if cease_png:
                out["_echo"]["cease_png"] = cease_png

            return out

        except Exception as e:
            msg = str(e).lower()
            if "insufficient_quota" in msg:
                cooldown = int(os.getenv("OPENAI_CIRCUIT_COOLDOWN", "600"))
                globals()["_oa_circuit_until"] = _now_mono() + cooldown
                print(f"[OpenAI] quota exhausted → circuit open for {cooldown}s")
                return None
            if "rate_limit" in msg or "429" in msg:
                await asyncio.sleep(backoff + random.random()*0.3)
                backoff = min(backoff*2, 8.0)
                continue
            raise
    return None

# === SRVDEBUG: 鯖番号読取デバッグ ===
async def _srvdebug_from_bytes(img_bytes: bytes, filename: str, channel_id: int):
    await client.wait_until_ready()
    ch = client.get_channel(channel_id)
    if not ch:
        return

    # 1) 入力画像のEXIF回転を正しく適用（重要）
    pil = Image.open(io.BytesIO(img_bytes))
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    full_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # 2) 領域の可視化（HEAD/CLOCK/CENTER/CEASE）: フル画像基準で一括ROI
    marked, head_rect, clock_rect, center_rect, cease_rect = _mark_regions_on_full(full_bgr)

    # 3) 各クロップ
    #    ヘッダ・時計は“黒塗り前の原画像”から取る
    x1, y1, x2, y2 = head_rect
    head_bgr = full_bgr[y1:y2, x1:x2]

    x1, y1, x2, y2 = clock_rect
    top_bgr = full_bgr[y1:y2, x1:x2]  # = CLOCK ROI（ファイル名に合わせて top_bgr のまま）

    #    CENTER/CEASE は前処理（免戦直下黒塗りなど）後の画像から
    masked_bgr, _ = auto_mask_ime(full_bgr)

    # center は「上=HEAD下端、下=CEASE上端、右=時計左端」で再構築（ズレ防止）
    cx1 = 0
    cy1 = head_rect[3]
    cx2 = clock_rect[0]
    cy2 = cease_rect[1]
    center_bgr = masked_bgr[cy1:cy2, cx1:cx2]

    # 停戦帯は動的検出があれば上書き（変数名は衝突させない）
    cease_bgr = masked_bgr[cease_rect[1]:cease_rect[3], cease_rect[0]:cease_rect[2]]
    cease_cands = find_ceasefire_regions_full_img(masked_bgr)
    if cease_cands:
        x1, y1, x2, y2 = cease_cands[0]
        cease_bgr = masked_bgr[y1:y2, x1:x2]

    # 4) 実際にOpenAIへ送っている「合成PNG」を作る（必ず full_bgr から）
    comp_bgr, _ = compose_center_with_clock_and_cease(full_bgr)
    comp_small = shrink_long_side(comp_bgr, SINGLEIMG_MAX_SIDE)

    # 5) 3エンジン比較（ヘッダ帯のみで確認）
    paddle_lines = ocr_center_paddle(head_bgr)
    gv_lines = extract_text_from_image_google(head_bgr) if GV_CLIENT else []
    oa_head = []
    if OA_CLIENT and DEBUG_ATTACH_TO_OPENAI:
        uri = _bgr_to_data_uri_np(shrink_long_side(head_bgr, 768))
        try:
            res = OA_CLIENT.responses.create(
                model=os.getenv("OPENAI_OCR_MODEL", "gpt-4o-mini"),
                input=[{"role": "user", "content": [
                    {"type": "input_text", "text": "タイトル帯からサーバ番号 [s####] / s#### / #### を1行で返して。説明は不要。"},
                    {"type": "input_image", "image_url": uri, "detail": "high"},
                ]}],
                max_output_tokens=32,
                temperature=0
            )
            txt = (res.output_text or "").strip()
            oa_head = [t.strip() for t in txt.splitlines() if t.strip()]
        except Exception as e:
            oa_head = [f"(OpenAI error: {e})"]

    # 6) 正規化（[s1234] / s1234 / 1234 → "1234"）※“最後の番号”ルール
    def _pick_num(lines):
        return _pick_last_server_from_lines(lines)

    raw_paddle = _pick_num(paddle_lines)
    raw_gv = _pick_num(gv_lines)
    raw_oa = _pick_num(oa_head)

    norm_paddle = _normalize_server(raw_paddle)
    norm_gv = _normalize_server(raw_gv)
    norm_oa = _normalize_server(raw_oa)

    # 7) Discordへ添付（実物を見ながら調整できる）
    files = []

    def _add(name, bgr, q=95):
        if bgr is None or not getattr(bgr, "size", 0):
            return
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            files.append(discord.File(io.BytesIO(buf.tobytes()), filename=name))

    _add("srvdebug_marked.jpg",         shrink_long_side(marked, 1280))
    _add("srvdebug_head.jpg",           shrink_long_side(head_bgr, 820))
    _add("srvdebug_clock.jpg",          shrink_long_side(top_bgr, 820))
    _add("srvdebug_center.jpg",         shrink_long_side(center_bgr, 1000))
    _add("srvdebug_cease.jpg",          shrink_long_side(cease_bgr, 820))
    _add("srvdebug_composite_sent.jpg", comp_small)

    hH, hW = head_bgr.shape[:2]
    lines = [
        f"🔎 **サーバ番号読取デバッグ** `{filename}`",
        "",
        f"🖼 合成送付サイズ: {comp_small.shape[1]}x{comp_small.shape[0]}  (SINGLEIMG_MAX_SIDE={SINGLEIMG_MAX_SIDE})",
        f"🎛 ヘッダ帯サイズ: {hW}x{hH}  HEAD_TOP={HEAD_TOP_RATIO} / BOTTOM={HEAD_BOTTOM_RATIO} / RIGHT={HEAD_RIGHT_RATIO}",
        "",
        "📚 **ヘッダ帯 OCR 結果**",
        f"・Paddle:  {paddle_lines[:4] or '(なし)'}  → raw={raw_paddle!r} / norm={norm_paddle!r}",
        f"・Google:  {gv_lines[:4]     or '(なし)'}  → raw={raw_gv!r} / norm={norm_gv!r}",
        f"・OpenAI:  {oa_head[:2]      or '(未送信)'} → raw={raw_oa!r} / norm={norm_oa!r}",
        "",
        "✅ 期待: いずれかで norm が4桁になっていること。",
        "❗ ダメなときの典型:",
        "  - 合成にヘッダが写っていない / 文字が小さすぎる（数字高さ<16px）",
        "  - EXIF回転未適用でクロップが外れている",
        "  - マスク/トリムでヘッダが欠けている",
        "  - プロンプトで『数字4桁だけ返す』が弱い",
    ]
    await ch.send("\n".join(lines), files=files)

def oai_extract_parking_json(center_bgr: np.ndarray) -> dict | None:
    """
    中央リストから『越域駐騎場<番号>』と『免戦中 HH:MM:SS』を JSON で抽出。
    返り値例:
      {"server":"s1296","rows":[{"place":2,"status":"免戦中","duration":"02:00:38"}]}
    """
    if OA_CLIENT is None:
        return None

    ok, buf = cv2.imencode(".png", center_bgr)
    if not ok:
        return None
    data_uri = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")

    prompt = (
        "ゲーム画面の中央リストから、各行の『越域駐騎場<番号>』と、その行にある『免戦中』の残り時間を抽出して、"
        "JSONで返してください。サーバー番号（例: [s1296]）が見えれば server として含めてください。"
        "フォーマットは厳守。追加の説明・前置き・コードフェンス禁止：\n"
        '{"server":"s####","rows":[{"place":<int>,"status":"免戦中","duration":"HH:MM:SS"}]}'
    )

    def _strip_code_fences(s: str) -> str:
        s = s.strip()
        if s.startswith("```"):
            # ```json ... ``` / ``` ... ```
            s = re.sub(r"^```(?:json)?\s*", "", s)
            s = re.sub(r"\s*```$", "", s)
        return s.strip()

    try:
        res = OA_CLIENT.responses.create(
            model=os.getenv("OPENAI_OCR_MODEL", "gpt-5-mini"),
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_uri, "detail": "high"},
                ],
            }],
            temperature=0,
            max_output_tokens=512,
        )
        txt = (res.output_text or "").strip()
        txt = _strip_code_fences(txt)
        return json.loads(txt) if txt.startswith("{") else None

    except Exception as e:
        print(f"[OpenAI OCR JSON] error: {e}")
        return None



async def ping_google_vision() -> tuple[bool, str]:
    """
    Visionクライアントの有無と、実際に tiny 画像で text_detection を叩いてみた結果を返す。
    戻り値: (成功したか, メッセージ)
    """
    if GV_CLIENT is None:
        return False, "GV_CLIENT=None（環境変数 GOOGLE_CLOUD_VISION_JSON 未設定/壊れている可能性）"

    # 1x1の黒画像で疎通テスト（文字は検出されなくてOK。例外が出ないかを見る）
    tiny = np.zeros((1, 1, 3), dtype=np.uint8)
    try:
        lines = google_ocr_from_np(tiny)  # 例外がなければOK
        # 成功。ただし返るテキストは通常空なので、その旨の文言にする
        return True, f"API呼び出し成功（text_annotations={len(lines)}行・通常は0で正常）"
    except Exception as e:
        return False, f"API呼び出しで例外: {e!r}"

# ===== 剣/盾検出 → 右側黒塗り ヘルパー =====
# HSV 色域（端末差でズレることがあるので必要なら微調整）
HSV_BLUE_LOW1  = (95,  80, 60)
HSV_BLUE_HIGH1 = (125,255,255)
HSV_BLUE_LOW2  = (85,  60, 60)   # 青の下側を少し拾う保険帯
HSV_BLUE_HIGH2 = (95, 255,255)

HSV_RED_LOW1   = (0,   100, 80)  # 赤は 0°/180° の両側を取る
HSV_RED_HIGH1  = (10,  255,255)
HSV_RED_LOW2   = (170, 100, 80)
HSV_RED_HIGH2  = (180, 255,255)

def detect_sword_shield_boxes(bgr: np.ndarray) -> list[tuple[int,int,int,int]]:
    """剣(赤系)・盾(青系)の色域で候補矩形を返す (x,y,w,h) のリスト"""
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    blue  = cv2.inRange(hsv, HSV_BLUE_LOW1,  HSV_BLUE_HIGH1) | cv2.inRange(hsv, HSV_BLUE_LOW2, HSV_BLUE_HIGH2)
    red   = cv2.inRange(hsv, HSV_RED_LOW1,   HSV_RED_HIGH1)  | cv2.inRange(hsv, HSV_RED_LOW2,  HSV_RED_HIGH2)
    mask  = blue | red

    # ノイズ整形
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    # 中央のリストパネル付近だけに限定（任意：薄いベージュ領域）
    panel = cv2.inRange(hsv, (0,0,180), (179,60,255))
    cnts,_ = cv2.findContours(panel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cand = [(cv2.boundingRect(c), cv2.contourArea(c)) for c in cnts]
        # 画面内でそこそこ大きい矩形だけ
        cand = [r for r,a in cand if (r[2]*r[3]) > (w*h)*0.05]
        if cand:
            x,y,ww,hh = sorted(cand, key=lambda r:r[1])[-1]
            crop = np.zeros_like(mask); crop[y:y+hh, x:x+ww] = mask[y:y+hh, x:x+ww]
            mask = crop

    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    # 画面解像度に応じたサイズ帯（だいたいアイコン30〜70px四方想定）
    min_wh = max(15, int(min(h,w)*0.03))
    max_wh = int(min(h,w)*0.12)
    for c in cnts:
        x,y,ww,hh = cv2.boundingRect(c)
        area = ww*hh
        if min_wh <= ww <= max_wh and min_wh <= hh <= max_wh and 400 < area < 8000:
            # 画面の中央～下部のリスト行だけ
            if h*0.18 < y < h*0.9:
                boxes.append((x,y,ww,hh))
    return boxes

def redact_right_of_boxes(bgr: np.ndarray, boxes: list[tuple[int,int,int,int]], right_width_px: int, pad: int = 6) -> np.ndarray:
    """各ボックスの右側を right_width_px ぶん黒塗り"""
    out = bgr.copy()
    H, W = out.shape[:2]
    for x,y,w,h in boxes:
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W, x + w + right_width_px)
        y2 = min(H, y + h + pad)
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,0,0), -1)
    return out



# ===== 「免戦中」の直下を右端まで黒塗りするヘルパー =====
# チューニング用の係数（必要に応じて微調整）
IME_ABOVE_RATIO = -0.10   # 上端を文字の下へ0.1文字ぶん下げる
IME_BELOW_RATIO = 1.10    # 帯の厚みを控えめに
IME_LEFT_MARGIN = 12      # ← 未定義なら追加（免戦中の右端からの余白px）

def find_ime_sen_rows_full_img(bgr: np.ndarray) -> list[tuple[int,int,int,int]]:
    """
    画像全体に対して PaddleOCR を走らせ、'免戦中' を含むテキストボックスを見つけ、
    その直下の横帯 (x1,y1,x2,y2) を返す。座標は画像そのもののピクセル座標。
    """
    rows = []
    try:
        result = ocr.ocr(bgr, cls=True)
    except Exception:
        result = None

    if not result or not result[0]:
        return rows

    H, W = bgr.shape[:2]
    for item in result[0]:
        box, (text, conf) = item
        if "免戦中" not in str(text):
            continue

        xs = [int(p[0]) for p in box]
        ys = [int(p[1]) for p in box]
        x_min, x_max = max(0, min(xs)), min(W, max(xs))
        y_min, y_max = max(0, min(ys)), min(H, max(ys))
        h_txt = max(8, y_max - y_min)

        # 免戦中ボックスの「すぐ下」を帯にする
        y1 = int(y_max - h_txt * IME_ABOVE_RATIO)
        y2 = int(y_max + h_txt * IME_BELOW_RATIO)
        x1 = 0
        x2 = W

        # 画面外クリップ
        y1 = max(0, min(H, y1))
        y2 = max(0, min(H, y2))
        if y2 - y1 < max(10, int(h_txt*0.6)):   # あまりに薄い帯はスキップ
            continue
        if x1 >= x2:
            continue

        rows.append((x1, y1, x2, y2))

    return rows

def fill_rects_black(bgr: np.ndarray, rects: list[tuple[int,int,int,int]]) -> np.ndarray:
    out = bgr.copy()
    for (x1, y1, x2, y2) in rects:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
    return out

def auto_mask_ime(bgr: np.ndarray) -> tuple[np.ndarray, int]:
    """
    '免戦中' を検出して、その直下の帯を右端まで黒塗り。
    戻り値: (黒塗り後画像, 見つかった数)
    """
    rects = find_ime_sen_rows_full_img(bgr)
    if not rects:
        return bgr, 0
    return fill_rects_black(bgr, rects), len(rects)

# ==== 停戦終了の検出＆切り取り（Y軸ちょい下寄せ） ====

# 調整用パラメータ（必要に応じて環境変数で上書き）
CEASE_PAD_X            = int(os.getenv("CEASE_PAD_X", "20"))      # 左右の余白(px)
CEASE_PAD_TOP          = int(os.getenv("CEASE_PAD_TOP", "6"))     # 上側の余白(px)
CEASE_PAD_BOTTOM       = int(os.getenv("CEASE_PAD_BOTTOM", "68")) # 下側の余白(px) ← 少し多め
CEASE_Y_SHIFT_RATIO    = float(os.getenv("CEASE_Y_SHIFT_RATIO", "0.45"))  # テキスト高さに対する下方向オフセット比

def find_ceasefire_regions_full_img(
    bgr: np.ndarray,
    y_shift_ratio: float | None = None,
    pad_bottom: int | None = None,
    pad_x: int | None = None,
    pad_top: int | None = None,
) -> list[tuple[int,int,int,int]]:
    """
    画像全体を PaddleOCR で走らせ、『停戦終了』を含むテキストボックスを見つけ、
    その周辺を“少し下寄せ”で切り取り矩形 (x1,y1,x2,y2) のリストで返す。
    """
    y_shift_ratio = CEASE_Y_SHIFT_RATIO if y_shift_ratio is None else y_shift_ratio
    pad_bottom    = CEASE_PAD_BOTTOM    if pad_bottom    is None else pad_bottom
    pad_x         = CEASE_PAD_X         if pad_x         is None else pad_x
    pad_top       = CEASE_PAD_TOP       if pad_top       is None else pad_top

    rects: list[tuple[int,int,int,int]] = []
    try:
        result = ocr.ocr(bgr, cls=True)
    except Exception:
        result = None

    if not result or not result[0]:
        return rects

    H, W = bgr.shape[:2]
    for item in result[0]:
        box, (text, conf) = item
        if "停戦終了" not in str(text):
            continue

        xs = [int(p[0]) for p in box]
        ys = [int(p[1]) for p in box]
        x_min, x_max = max(0, min(xs)), min(W, max(xs))
        y_min, y_max = max(0, min(ys)), min(H, max(ys))
        h_txt = max(8, y_max - y_min)

        # “Y軸をもう少し下へ”の肝：テキスト高さ割合で下方向にシフト
        y_shift = int(h_txt * y_shift_ratio)

        x1 = max(0, x_min - pad_x)
        x2 = min(W, x_max + pad_x)
        y1 = max(0, y_min - pad_top + y_shift)
        y2 = min(H, y_max + pad_bottom + y_shift)

        if (x2 - x1) >= 10 and (y2 - y1) >= 10:
            rects.append((x1, y1, x2, y2))

    return rects

async def _notify_discord_upload_meta(filename: str, meta: dict, channel_id: int):
    await client.wait_until_ready()
    ch = client.get_channel(channel_id)   # ← 渡されたチャンネルへ
    if not ch:
        return
    lines = [f"🗂 **アップロード解析** `{filename}`", ""]
    # EXIF
    exif = meta.get("exif_dt_map") or {}
    if exif:
        # 優先順で1つだけ強調表示
        picked = False
        for k in ("DateTimeOriginal","DateTimeDigitized","DateTime"):
            if k in exif:
                lines.append(f"📸 EXIF {k}: `{exif[k]}`")
                picked = True
                break
        if not picked:
            # 何かしらあるが上記3種なし
            for k, v in list(exif.items())[:3]:
                lines.append(f"📸 EXIF {k}: `{v}`")
    else:
        lines.append("📸 EXIF: なし/未取得")

    # PNG tIME
    if meta.get("png_time"):
        lines.append(f"🧩 PNG tIME: `{meta['png_time']}`")

    # XMP
    #if meta.get("xmp_short"):
    #    lines.append(f"📝 XMP: {meta['xmp_short']}")

    # 総合推定（既存ヘルパの get_taken_time_from_image_bytes）
    if meta.get("taken_guess"):
        dtg = meta["taken_guess"]
        lines.append(f"🕒 推定撮影/作成: `{dtg['when']}` 〔{dtg['how']} raw:{dtg['raw']}〕")

    await ch.send("\n".join(lines))

def notify_discord_upload_meta_threadsafe(filename: str, meta: dict, channel_id: int):
    try:
        loop = DISCORD_LOOP
        if loop is None:
            print("[notify_threadsafe] Discord loop not ready; skip")
            return
        asyncio.run_coroutine_threadsafe(
            _notify_discord_upload_meta(filename, meta, channel_id),
            loop
        )
    except Exception as e:
        print(f"[notify_threadsafe] failed: {e}")

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

async def auto_delete_after(msg, seconds: int):
    try:
        await asyncio.sleep(seconds)
        await msg.delete()
    except:
        pass

async def apply_adjust_for_server_place(server: str, place: str, sec_adj: int):
    """
    (server, place) で一番早い予定を sec_adj 秒ずらし、
    先に new_txt を登録→まとめ/コピー/通知へ反映→最後に旧や重複を掃除、の順で安全に更新する。
    戻り値: (old_txt, new_txt) または None
    """
    # 対象候補の収集
    candidates = []
    for txt, ent in list(pending_places.items()):
        g = parse_txt_fields(txt)
        if g and g[1] == server and g[2] == place:
            candidates.append((txt, ent))
    if not candidates:
        return None

    # 一番早い予定を基準
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

    # 既に new_txt が存在（同刻）→ old を消して統合
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

        # 同(server,place)の残骸掃除
        for txt, ent in list(pending_places.items()):
            g2 = parse_txt_fields(txt)
            if g2 and g2[1] == server and g2[2] == place and txt != new_txt:
                await retime_event_in_summary(txt, pending_places[new_txt]["dt"], new_txt, client.get_channel(NOTIFY_CHANNEL_ID))
                try:
                    if ent.get("copy_msg_id"):
                        ch_copy = client.get_channel(COPY_CHANNEL_ID)
                        if ch_copy:
                            msg = await ch_copy.fetch_message(ent["copy_msg_id"])
                            await msg.delete()
                except:
                    pass
                for key in [(txt, "2min"), (txt, "15s")]:
                    tsk = sent_notifications_tasks.pop(key, None)
                    if tsk:
                        tsk.cancel()
                pending_places.pop(txt, None)
        return (old_txt, new_txt)

    # ===== 先に new を追加 → 反映 → 最後に旧を掃除 =====

    # 1) new を追加（IDは引き継ぎ）
    old_main_id = entry.get("main_msg_id")
    old_copy_id = entry.get("copy_msg_id")
    pending_places[new_txt] = {
        "dt": new_dt,
        "txt": new_txt,
        "server": server,
        "created_at": entry.get("created_at", now_jst()),
        "main_msg_id": old_main_id,
        "copy_msg_id": old_copy_id,
    }

    # 2) まとめを old→new に差し替え
    await retime_event_in_summary(old_txt, new_dt, new_txt, client.get_channel(NOTIFY_CHANNEL_ID))

    # 3) コピーチャンネル：旧メッセがあれば内容だけ new に編集
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

    # 4) 通知再登録（奪取のみ）
    notify_ch = client.get_channel(NOTIFY_CHANNEL_ID)
    if notify_ch and new_txt.startswith("奪取"):
        await schedule_notification(new_dt, new_txt, notify_ch)

    # 5) 最後に同(server,place)で new_txt 以外を全部掃除（old含む）
    for txt, ent in list(pending_places.items()):
        g2 = parse_txt_fields(txt)
        if g2 and g2[1] == server and g2[2] == place and txt != new_txt:
            await retime_event_in_summary(txt, new_dt, new_txt, client.get_channel(NOTIFY_CHANNEL_ID))
            try:
                if ent.get("copy_msg_id"):
                    ch_copy = client.get_channel(COPY_CHANNEL_ID)
                    if ch_copy:
                        msg = await ch_copy.fetch_message(ent["copy_msg_id"])
                        await msg.delete()
            except:
                pass
            for key in [(txt, "2min"), (txt, "15s")]:
                tsk = sent_notifications_tasks.pop(key, None)
                if tsk:
                    tsk.cancel()
            pending_places.pop(txt, None)

    return (old_txt, new_txt)

def crop_top_right(img):
    h, w = img.shape[:2]
    x1 = int(w * CLOCK_LEFT_RATIO)
    x2 = int(w * CLOCK_RIGHT_RATIO)
    y1 = int(h * CLOCK_TOP_RATIO)
    y2 = int(h * CLOCK_BOTTOM_RATIO)
    # はみ出し防止
    x1 = max(0, min(x1, w-1)); x2 = max(x1+1, min(x2, w))
    y1 = max(0, min(y1, h-1)); y2 = max(y1+1, min(y2, h))
    return img[y1:y2, x1:x2]

def crop_center_area(img):
    h, w = img.shape[:2]

    # 上端＝HEADの下端
    y1 = int(h * HEAD_BOTTOM_RATIO)

    # 下端＝CEASEの上端（検出→なければフォールバック）
    rects = find_ceasefire_regions_full_img(img)
    if rects:
        cease_top = rects[0][1]  # (x1,y1,x2,y2) の y1
    else:
        cease_top = int(h * CEASE_FALLBACK_TOP)
    y2 = cease_top

    # 右端は“時計の左端”にロック
    x1 = 0
    x2 = int(w * CLOCK_LEFT_RATIO)

    # はみ出し防止
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    return img[y1:y2, x1:x2]
    
def extract_text_from_image(img):
    result = ocr.ocr(img, cls=True)
    return [line[1][0] for line in result[0]] if result and result[0] else []

def extract_text_from_image_google(np_bgr: np.ndarray) -> list[str]:
    """
    画像(np.ndarray BGR)を Google Vision だけでOCRして行ごとに返す。
    """
    if GV_CLIENT is None:
        return []
    try:
        lines = google_ocr_from_np(np_bgr)
        # 軽く整形
        out = []
        seen = set()
        for t in lines:
            t2 = normalize_time_separators(t)
            t2 = force_hhmmss_if_six_digits(t2)
            if t2 and t2 not in seen:
                seen.add(t2)
                out.append(t2)
        return out
    except Exception as e:
        print(f"[GV-OCR] error: {e}")
        return []

async def extract_text_from_image_google_async(np_bgr: np.ndarray) -> list[str]:
    """GVの同期APIをスレッドに逃がしてイベントループを止めない"""
    return await asyncio.to_thread(extract_text_from_image_google, np_bgr)

# ---- 中央OCR強化ユーティリティ ----
def preprocess_for_colon(img_bgr: np.ndarray) -> list[np.ndarray]:
    """
    コロン(:)の2点が消えないように複数前処理を作成して返す（BGRのまま）。
    """
    outs = []

    # 0) 原画像
    outs.append(img_bgr)

    # 1) 2倍拡大 + 軽いシャープ
    up = cv2.resize(img_bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(up, (0, 0), 1.0)
    sharp = cv2.addWeighted(up, 1.5, blur, -0.5, 0)
    outs.append(sharp)

    # 2) CLAHE + 自適応二値化（白黒両方）
    g = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    th = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 5)
    outs.append(cv2.cvtColor(th, cv2.COLOR_GRAY2BGR))
    outs.append(cv2.cvtColor(255 - th, cv2.COLOR_GRAY2BGR))  # 反転版

    # 3) 小粒点(:)が消えないようにclosing→opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    outs.append(cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR))

    return outs

def normalize_time_separators(s: str) -> str:
    """
    : を ; . ・ / などから復元。全角→半角、不要空白除去、1の誤読補正など。
    """
    s = s.replace("：", ":").replace("；", ";").replace("．", ".").replace("・", "・")
    s = s.replace("’", ":").replace("‘", ":").replace("ː", ":")
    s = s.replace("I", "1").replace("|", "1").replace("l", "1")
    s = s.replace(";", ":").replace("･", ":").replace("・", ":").replace(".", ":").replace("/", ":")
    s = re.sub(r"\s+", "", s)
    return s

def force_hhmmss_if_six_digits(s: str) -> str:
    """
    6桁数字だけ拾えたケースを HH:MM:SS に再構成。
    """
    digits = re.sub(r"\D", "", s)
    if len(digits) == 6:
        h, m, sec = digits[:2], digits[2:4], digits[4:]
        return f"{h}:{m}:{sec}"
    return s
    
def _ocr_clock_topright_to_jst(img_bytes: bytes) -> tuple[datetime|None, str]:
    """
    画像右上の画面内時計をOCRして、JSTの datetime を返す。
    戻り値: (datetime or None, raw_text)
    """
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # 「免戦中」直下は黒塗り（他処理と合わせる）
        bgr, _ = auto_mask_ime(bgr)

        # 右上トリム
        top = crop_top_right(bgr)

        # まず Paddle、弱ければ Google Vision にフォールバック
        texts = extract_text_from_image(top)
        if not texts and GV_CLIENT is not None:
            texts = extract_text_from_image_google(top)

        raw = texts[0] if texts else ""
        if not raw:
            return None, ""

        # 書式補正
        fixed = normalize_time_separators(raw)
        fixed = force_hhmmss_if_six_digits(fixed)

        # HH:MM:SS を抽出
        m = re.search(r"\b(\d{1,2}):(\d{2}):(\d{2})\b", fixed)
        if m:
            h, mi, se = map(int, m.groups())
        else:
            digits = re.sub(r"\D", "", fixed)
            if len(digits) < 6:
                return None, raw
            h, mi, se = int(digits[:2]), int(digits[2:4]), int(digits[4:6])

        # 今日の日付で JST datetime に
        today = now_jst().date()
        base = datetime.combine(today, time(h % 24, mi, se), tzinfo=JST)

        # 00:00〜05:59 は翌日扱いに寄せる（他ロジックと統一）
        if base.time() < time(6, 0, 0):
            base += timedelta(days=1)

        return base, raw
    except Exception:
        return None, ""

# === ここからコピペ（既存の center_ocr 関連の上に置いてOK）===

# 正規表現（そのまま流用）
IMSEN_RE = re.compile(r"免戦中")

def ocr_center_paddle(center_bgr: np.ndarray) -> list[str]:
    """中央領域をPaddleOCRだけで読む（前処理つき）"""
    candidates = preprocess_for_colon(center_bgr)
    results = []
    for cand in candidates:
        try:
            r = ocr.ocr(cand, cls=True)
            if not r or not r[0]:
                continue
            for line in r[0]:
                t = line[1][0]
                t = normalize_time_separators(t)
                t = force_hhmmss_if_six_digits(t)
                results.append(t)
        except Exception:
            pass

    # ユニーク化（出現順保持）
    seen, uniq = set(), []
    for t in results:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def ocr_center_google(center_bgr: np.ndarray) -> list[str]:
    """中央領域をGoogle Visionで読む（GV_CLIENTが無ければ空を返す）"""
    if GV_CLIENT is None:
        return []
    try:
        gv_lines = google_ocr_from_np(center_bgr)  # あなたの既存ヘルパー
    except Exception as e:
        print(f"[OCR] Vision error: {e}")
        return []
    # 軽く整形＋ユニーク化
    fixed, seen = [], set()
    for t in gv_lines:
        t2 = normalize_time_separators(t)
        t2 = force_hhmmss_if_six_digits(t2)
        if t2 and t2 not in seen:
            seen.add(t2)
            fixed.append(t2)
    return fixed


def ocr_center_with_fallback(center_bgr: np.ndarray) -> list[str]:
    """
    1) まずPaddle
    2) 弱ければGoogle Visionに自動フォールバック
    """
    paddle = ocr_center_paddle(center_bgr)
    if not center_ocr_is_poor(paddle):
        return paddle

    print("[OCR] fallback → Google Vision")
    google = ocr_center_google(center_bgr)
    # Visionが空ならPaddleの結果を返しておく
    return google or paddle

def center_ocr_is_poor(lines: list[str]) -> bool:
    """
    中央OCRの品質判定：
    - HH:MM:SS が1つも無い かつ 「免戦中」の出現が薄い ときは弱いとみなす
    """
    if any(TIME_RE.search(t) for t in lines):
        return False
    if sum(1 for t in lines if IMSEN_RE.search(t)) >= 1:
        # 免戦文字は見えてるならとりあえずOK
        return False
    return True

def extract_server_number(center_texts):
    for t in center_texts:
        # [] の有無どちらでも OK にする
        m = re.search(r"\[?[sS](\d{3,4})\]?", t)
        if m:
            return m.group(1)
    return None

def _extract_server_from_header(full_bgr: np.ndarray) -> Optional[str]:
    """まず内側クロップで抽出→失敗なら外側。複数なら『最後』を採用。"""
    H, W = full_bgr.shape[:2]
    x1, x2 = 0, int(W * HEAD_RIGHT_RATIO)

    inner_top    = float(globals().get("HEAD_INNER_TOP", 0.12))
    inner_bottom = float(globals().get("HEAD_INNER_BOTTOM", 0.28))

    # 1) 内側（ニュース帯を避ける＝タイトル行狙い）
    y1 = int(H * inner_top);  y2 = int(H * inner_bottom)
    head_inner = full_bgr[y1:y2, x1:x2]
    sid, _ = _triage_read_server_from_head(head_inner)
    if sid:
        return sid

    # 2) 外側（広め）
    y1 = int(H * HEAD_TOP_RATIO);  y2 = int(H * HEAD_BOTTOM_RATIO)
    head_outer = full_bgr[y1:y2, x1:x2]
    sid, _ = _triage_read_server_from_head(head_outer)
    return sid

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

TIME_RE = TIME_HHMMSS   # 後方互換のため

def _extract_clock_from_center_lines(lines):
    """
    center_lines に混ざった HH:MM:SS の '時刻' を拾う。
    ・免戦時間(通常 00:MM:SS, 01-05:MM:SS)は劣後、6時間以上を優先
    ・複数あれば一番大きい（見た目右上の時計に近い）を採用
    """
    if not lines:
        return None
    txt = "\n".join(str(s) for s in lines if s)
    import re
    cand = re.findall(r'\b([0-2]\d):([0-5]\d):([0-5]\d)\b', txt)
    if not cand:
        return None
    def to_sec(h,m,s): return int(h)*3600 + int(m)*60 + int(s)
    hi, lo = [], []
    for h,m,s in cand:
        (hi if int(h) >= 6 else lo).append((to_sec(h,m,s), f"{h}:{m}:{s}"))
    if hi:
        hi.sort(reverse=True)
        return hi[0][1]
    lo.sort(reverse=True)
    return lo[0][1] if lo else None


def _ocr_clock_from_roi(full_bgr):
    """
    右上の時計ROIをローカルOCR(Paddle/Google)で読む最終手段。
    """
    H, W = full_bgr.shape[:2]
    x1 = int(W * CLOCK_LEFT_RATIO)
    x2 = int(W * CLOCK_RIGHT_RATIO)
    y1 = int(H * CLOCK_TOP_RATIO)
    y2 = int(H * CLOCK_BOTTOM_RATIO)
    roi = full_bgr[y1:y2, x1:x2]

    pp = ocr_center_paddle(roi) or []
    gv = google_ocr_from_np(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)) or []

    # まとめてユニーク
    lines = []
    for arr in (pp if isinstance(pp, list) else [pp],
                gv if isinstance(gv, list) else [gv]):
        for s in arr:
            s = str(s).strip()
            if s and s not in lines:
                lines.append(s)

    import re
    for s in lines:
        m = re.search(r'\b([0-2]\d:[0-5]\d:[0-5]\d)\b', s)
        if m:
            return m.group(1)
    return None

def extract_imsen_durations(texts: list[str]) -> list[str]:
    durations = []
    for text in texts:
        t = normalize_time_separators(text)
        # ① 「免戦中 …」を優先
        for m in re.findall(r"免戦中\s*([0-9:：]{4,10})", t):
            durations.append(correct_imsen_text(m))
        # ② ①で取れなかったら、裸の時間をフォールバックで拾う
        if not durations:
            # 旧: for m in TIME_RE.findall(t): durations.append(correct_imsen_text(m))
            for m in TIME_RE.finditer(t):
                durations.append(correct_imsen_text(m.group(0)))
    return durations

def parse_multiple_places(center_texts, top_time_texts,
                          base_time_override: str | None = None,
                          server_override: str | None = None):
    res = []

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

    top_time = base_time_override or extract_top_time(top_time_texts)
    server = server_override or extract_server_number(center_texts)
    server = _normalize_server(server)
    if not top_time or not server:
        return []

    mode = "警備" if server == "1268" else "奪取"

    # --- ① 1行完結パターン（その行に「免戦中 HH:MM:SS」まで含まれる）を直取り ---
    seen = set()  # (server, place) 重複防止
    for line in center_texts:
        m = PLACE_RE.search(line)
        if not m:
            continue
        place = m.group(1)
        d = pick_duration_from_group([line])  # 同じ行から時間を拾う
        if d:
            dt, unlock = add_time(top_time, d)
            if dt and (server, place) not in seen:
                res.append((dt, f"{mode} {server}-{place}-{unlock}", d))
                seen.add((server, place))

    # --- ② ヘッダ行と別行のパターン（グループ化）。ヘッダ行も lines に含めるのがポイント ---
    groups = []
    current = None
    for line in center_texts:
        m = PLACE_RE.search(line)
        if m:
            if current and current["place"] and current["lines"]:
                groups.append(current)
            current = {"place": m.group(1), "lines": [line]}  # ← ヘッダ行も入れる！
        else:
            if current:
                current["lines"].append(line)
    if current and current["place"] and current["lines"]:
        groups.append(current)

    for g in groups:
        if (server, g["place"]) in seen:
            continue
        d = pick_duration_from_group(g["lines"])
        if not d:
            continue
        dt, unlock = add_time(top_time, d)
        if dt:
            res.append((dt, f"{mode} {server}-{g['place']}-{unlock}", d))
            seen.add((server, g["place"]))

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
    # --- 追加の保険: 時(HH)が23を超える場合、先頭が '5' 誤読なら 0 に補正 ---
    m = re.match(r"(\d{1,2}):(\d{2}):(\d{2})", normalize_time_separators(text))
    if m:
        h, mi, se = map(int, m.groups())
        s = normalize_time_separators(text)
        if h > 23 and s.startswith("5"):
            return f"0{h%10}:{mi:02d}:{se:02d}"
    return normalize_time_separators(text)

# ==== 盾テンプレ生成・検出・黒塗り ====

TEMPLATE_PATH = Path("shield_template.png")

def _autobuild_shield_template(bgr: np.ndarray) -> np.ndarray | None:
    """
    画像から青盾らしき領域を1つ切り出してテンプレート化。
    失敗時は None。
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # 青（やや広め）：端末差に備えて範囲広め
    mask = cv2.inRange(hsv, (90, 60, 60), (130, 255, 255))
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    H,W = bgr.shape[:2]
    # 盾は中央～下部のリスト行に出やすい＆サイズは中くらい
    cand = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        if H*0.15 < y < H*0.85 and 18 < w < 120 and 18 < h < 120 and 300 < area < 15000:
            # 楕円度で絞る（円に近いほど良い）
            peri = cv2.arcLength(c, True)
            if peri == 0: 
                continue
            circularity = 4*np.pi*cv2.contourArea(c)/(peri*peri)
            cand.append((circularity, (x,y,w,h)))
    if not cand:
        return None

    # 円っぽい順に
    cand.sort(key=lambda t: t[0], reverse=True)
    x,y,w,h = cand[0][1]
    # 周囲に少しマージン
    pad = 4
    x1 = max(0, x-pad); y1 = max(0, y-pad)
    x2 = min(W, x+w+pad); y2 = min(H, y+h+pad)
    tpl = bgr[y1:y2, x1:x2].copy()
    try:
        cv2.imwrite(str(TEMPLATE_PATH), tpl)
    except Exception:
        pass
    return tpl

def _load_or_make_template(bgr_for_fallback: np.ndarray) -> np.ndarray | None:
    if TEMPLATE_PATH.exists():
        tpl = cv2.imread(str(TEMPLATE_PATH))
        if tpl is not None:
            return tpl
    # 無ければ作る
    return _autobuild_shield_template(bgr_for_fallback)

def _nms_points(points: list[tuple[int,int]], min_dist: int) -> list[tuple[int,int]]:
    """近接する検出点をまとめる簡易NMS（最小距離で間引く）"""
    kept = []
    for (x,y) in sorted(points, key=lambda p:(p[1], p[0])):
        if all((abs(x-x0) > min_dist or abs(y-y0) > min_dist) for (x0,y0) in kept):
            kept.append((x,y))
    return kept

def find_shields_by_template(bgr: np.ndarray, tpl: np.ndarray, thr: float = 0.78) -> list[tuple[int,int,int,int]]:
    """
    テンプレマッチで盾位置を返す [(x,y,w,h), ...]
    解像度は固定前提（リサイズなし）。
    """
    th, tw = tpl.shape[:2]
    # TM_CCOEFF_NORMED が安定しやすい
    res = cv2.matchTemplate(bgr, tpl, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(res >= thr)
    pts = list(zip(xs, ys))
    if not pts:
        return []
    # 近い重複を削る
    pts = _nms_points(pts, min_dist=max(8, min(th, tw)//2))
    return [(x, y, tw, th) for (x, y) in pts]

def mask_row_right(bgr: np.ndarray, boxes: list[tuple[int,int,int,int]], pad_y: int = 6) -> np.ndarray:
    """
    各盾ボックスの行を、画面右端まで黒塗り（縦は盾±pad_y）。
    """
    H, W = bgr.shape[:2]
    out = bgr.copy()
    for (x,y,w,h) in boxes:
        y1 = max(0, y - pad_y)
        y2 = min(H, y + h + pad_y)
        cv2.rectangle(out, (x, y1), (W-1, y2), (0,0,0), -1)
    return out

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
    target_block["min"] = min(d for d, _ in target_block["events"])
    target_block["max"] = max(d for d, _ in target_block["events"])

    # 3) まとめメッセージを更新
    if target_block.get("msg"):
        try:
            await target_block["msg"].edit(content=format_block_msg(target_block, with_footer=True))
        except:
            pass
    else:
        try:
            target_block["msg"] = await channel.send(content=format_block_msg(target_block, with_footer=True))
        except:
            target_block["msg"] = None

    # 4) main_msg_id を再ひも付け
    for d, t in target_block["events"]:
        if t == new_txt and target_block.get("msg"):
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

async def _register_from_image_bytes(img_bytes: bytes, filename: str, channel_id: int):
    global last_groups_seq, last_groups  # ← 先頭に移動
    await client.wait_until_ready()
    ch = client.get_channel(channel_id)
    if not ch:
        return

    # 解析中プレースホルダー
    status_msg = await ch.send(f"🔄 解析中… `{filename}`")

    try:
        # ---- 画像前処理 ----
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        np_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        np_img, _ = auto_mask_ime(np_img)  # 「免戦中」直下を黒塗り

        # トリム
        top = crop_top_right(np_img)        # 画面内時計エリア
        center = crop_center_area(np_img)   # 本文エリア

        # OCR（中央は Paddle→弱ければGV に自動フォールバック）
        top_txts_ocr = extract_text_from_image(top)      # 右上時計（Paddle）
        center_txts  = ocr_center_with_fallback(center)  # 中央（Paddle/GV）

        # 基準時刻（メタ優先 → OCR → 無し）
        base_time, base_kind = choose_base_time(img_bytes)  # ("HH:MM:SS"|None, "meta"|"ocr"|"none")
        parsed = parse_multiple_places(center_txts, top_txts_ocr, base_time_override=base_time)

        # ---- 登録処理 & !g 用グループ構築 ----
        image_results = []
        structured_entries_for_this_image = []
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
                await auto_dedup()
                pending_copy_queue.append((dt, txt))
                image_results.append(f"{txt} ({raw})")

                # 通知まとめ・事前通知のスケジュール
                task = asyncio.create_task(handle_new_event(dt, txt, client.get_channel(NOTIFY_CHANNEL_ID)))
                active_tasks.add(task); task.add_done_callback(lambda t: active_tasks.discard(t))
                if txt.startswith("奪取"):
                    t2 = asyncio.create_task(schedule_notification(dt, txt, client.get_channel(NOTIFY_CHANNEL_ID)))
                    active_tasks.add(t2); t2.add_done_callback(lambda t: active_tasks.discard(t))

        # !g グループ採番
        gid = None
        if structured_entries_for_this_image:
            last_groups_seq += 1
            gid = last_groups_seq
            last_groups[gid] = structured_entries_for_this_image

        # ---- 綺麗なEmbedで通知 ----
        color_ok   = 0x2ECC71
        color_none = 0x95A5A6

        kind_label = {"meta": "メタ", "ocr": "OCR", "none": "未取得"}[base_kind]
        base_label = (base_time or "??:??:??") + f"（{kind_label}）"

        if image_results:
            emb = discord.Embed(
                title="✅ 解析完了（フォーム）",
                description=f"`{filename}`",
                color=color_ok
            )
            if gid is not None:
                emb.add_field(name="グループ", value=f"G{gid}", inline=True)
            emb.add_field(name="基準時間", value=base_label, inline=True)

            # 予定一覧（最大1024文字対策して分割）
            joined = "\n".join(f"・{t}" for t in image_results)
            if len(joined) > 1024:
                chunk = []
                cur = ""
                for line in image_results:
                    line = f"・{line}"
                    if len(cur) + 1 + len(line) > 1000:
                        chunk.append(cur)
                        cur = line
                    else:
                        cur = (cur + "\n" + line) if cur else line
                if cur:
                    chunk.append(cur)
                for i, c in enumerate(chunk):
                    title = "登録された予定" if i == 0 else f"登録された予定（続き {i}）"
                    emb.add_field(name=title, value=c, inline=False)
            else:
                emb.add_field(name="登録された予定", value=joined, inline=False)

            emb.add_field(
                name="ヒント",
                value="`!g <grp>` で±秒の微調整 / `!a` で時刻を直接修正\n"
                      "実際の時間と違う場合はスクショを撮り直して再送してね",
                inline=False
            )
            emb.set_footer(text="OCR: Paddle + GV fallback（免戦中は自動黒塗り）")

            await ch.send(embed=emb)
        else:
            emb = discord.Embed(
                title="⚠️ 解析は完了しましたが新規登録はありませんでした",
                description=f"`{filename}`\n基準時間: {base_label}",
                color=color_none
            )
            emb.set_footer(text="OCR: Paddle + GV fallback（免戦中は自動黒塗り）")
            await ch.send(embed=emb)

        # 解析中メッセージは消す
        try:
            await status_msg.delete()
        except:
            pass

    except Exception as e:
        # 失敗したらプレースホルダーをエラーメッセに置換
        try:
            await status_msg.edit(content=f"❌ 解析失敗 `{filename}`: {e}")
        except:
            pass


def register_from_bytes_threadsafe(img_bytes: bytes, filename: str, channel_id: int):
    try:
        loop = DISCORD_LOOP
        if loop is None:
            print("[register_threadsafe] Discord loop not ready; skip")
            return
        asyncio.run_coroutine_threadsafe(
            _register_from_image_bytes(img_bytes, filename, channel_id),
            loop
        )
    except Exception as e:
        print(f"[register_threadsafe] failed: {e}")
# =======================
# 自動リセット処理（毎日02:00）
# =======================
async def daily_reset_task():
    global last_groups, last_groups_seq  # ← 先頭に移動
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
        last_groups.clear()
        last_groups_seq = 0
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
    global last_groups, last_groups_seq  # ← 先頭に移動

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
    last_groups.clear()
    last_groups_seq = 0

    await message.channel.send("✅ 全ての予定と通知をリセットしました")
# =======================
# Discordイベント
# =======================
@client.event
async def on_ready():
    global DISCORD_LOOP
    DISCORD_LOOP = asyncio.get_running_loop()  # Discordのイベントループを記録

    print("✅ Discord ログイン成功！")
    print(f"📌 通知チャンネル: {NOTIFY_CHANNEL_ID}")
    print(f"📌 読み取り許可チャンネル: {READABLE_CHANNEL_IDS}")

    # 起動時にバックグラウンドタスクを立ち上げる
    asyncio.create_task(daily_reset_task())      # 自動リセット（毎日02:00）
    asyncio.create_task(periodic_cleanup_task()) # 過去予定の定期削除（1分おき）
    asyncio.create_task(process_copy_queue())    # コピーキュー処理

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
    global last_groups, last_groups_seq   # ← これを最初に置く
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

    # ==== !gv ====
    if message.content.strip() == "!gv":
        ok, info = await ping_google_vision()
        status = "✅ OK" if ok else "❌ NG"
        # 初期化ログの補助情報
        init_hint = "（起動時に『✅ Google Vision client ready』が表示されていれば初期化は成功）"
        await message.channel.send(f"{status} Google Vision ステータス: {info}\n{init_hint}")
        return

    # ==== !1 駐騎場ナンバーで一括 -1 秒 ====
    # 例) "!1 1 12 11" → place が 1,12,11 の (server, place) それぞれの最も早い予定を -1 秒
    if message.content.strip().startswith("!1"):
        parts = message.content.strip().split()
        if len(parts) < 2:
            await message.channel.send("⚠️ 使い方: `!1 <駐騎場> <駐騎場> ...` 例: `!1 1 12 11`")
            return

        if not pending_places:
            await message.channel.send("⚠️ 登録された予定はありません")
            return

        target_places = set(parts[1:])  # 文字列のまま（txt内のplaceと一致させる）
        # 現在の pending から (server, place) のユニーク集合を作成（対象placeのみ）
        pairs = set()
        for txt in list(pending_places.keys()):
            g = parse_txt_fields(txt)
            if not g:
                continue
            _mode, server, place, _hhmmss = g
            if place in target_places:
                pairs.add((server, place))

        if not pairs:
            await message.channel.send("⚠️ 対象の駐騎場の予定が見つかりませんでした")
            return

        updated_pairs = []  # (old_txt, new_txt)
        skipped = 0

        # 各 (server, place) で最も早い予定を -1 秒
        for server, place in sorted(pairs):
            res = await apply_adjust_for_server_place(server, place, -1)
            if res:
                updated_pairs.append(res)
            else:
                skipped += 1  # 何らかの理由で該当が消えていた等

        # 手動まとめ(!s)があれば最新化
        await refresh_manual_summaries()
        # コピー用チャンネルは全体完全同期（差分指定なしでOKな実装）
        await upsert_copy_channel_sorted([])

        if not updated_pairs and skipped > 0:
            await message.channel.send("（変更なし）対象が見つからなかった/既に削除済みの予定がありました")
            return

        if not updated_pairs:
            await message.channel.send("⚠️ 対象の駐騎場の予定が見つかりませんでした")
            return

        # レポート
        lines = ["✅ -1秒を適用しました", ""]
        for o, n in updated_pairs:
            lines.append(f"・{o} → {n}")
        if skipped:
            lines.append("")
            lines.append(f"（対象外/見つからず: {skipped} 件）")
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

        # ここからレポート部分を置き換え
        await refresh_manual_summaries()
        await upsert_copy_channel_sorted([])  # 引数は無視される設計。全体を完全同期。

        updated_cnt = len(updated_pairs)
        skipped_cnt = skipped

        if updated_cnt > 0:
            msg = ["✅ !g の結果"]
            msg.append(f"　更新: {updated_cnt} 件")
            if skipped_cnt > 0:
                msg.append(f"　対象外/見つからず: {skipped_cnt} 件")
            # 変更一覧
            msg.append("")
            msg.append("🔧 変更一覧:")
            msg += [f"　{o} → {n}" for o, n in updated_pairs]
            await message.channel.send("\n".join(msg))

        elif skipped_cnt > 0:
            # 更新 0 件でも、対象が見つからなかった/既に消えていた等は明示する
            await message.channel.send(f"（変更なし）対象が見つからなかった/既に削除済み: {skipped_cnt} 件")

        else:
            # 本当に何もヒットしなかった（gID が不正、last_groups が空など）
            await message.channel.send("（変更なし）該当グループが空か、pending に一致がありませんでした")
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

    # ==== !maskime 免戦中の直下を右端まで黒塗り ====
    if message.content.strip().startswith("!maskime"):
        if not message.attachments:
            await message.channel.send("🖼 画像を添付して `!maskime` を実行してね")
            return

        # 係数の上書き（オプション）: 例) !maskime 1.6 0.06 12
        try:
            parts = message.content.strip().split()
            if len(parts) >= 2:  # BELOW_RATIO
                globals()["IME_BELOW_RATIO"] = float(parts[1])
            if len(parts) >= 3:  # ABOVE_RATIO
                globals()["IME_ABOVE_RATIO"] = float(parts[2])
            if len(parts) >= 4:  # LEFT_MARGIN(px)
                globals()["IME_LEFT_MARGIN"] = int(parts[3])
        except Exception:
            pass

        for att in message.attachments:
            data = await att.read()
            img  = Image.open(io.BytesIO(data)).convert("RGB")
            bgr  = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            rects = find_ime_sen_rows_full_img(bgr)
            if not rects:
                await message.channel.send("⚠️ '免戦中' が見つからず、黒塗りは行いませんでした。")
                continue

            out_bgr = fill_rects_black(bgr, rects)
            ok, buf = cv2.imencode(".jpg", out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if not ok:
                await message.channel.send("⚠️ 画像のエンコードに失敗しました")
                continue

            # デバッグ用に帯の本数も表示
            await message.channel.send(
                content=f"✅ 黒塗り完了（帯: {len(rects)} 本 / BELOW={IME_BELOW_RATIO} ABOVE={IME_ABOVE_RATIO} MARGIN={IME_LEFT_MARGIN}px）",
                file=discord.File(io.BytesIO(buf.tobytes()), filename=f"maskime_{att.filename.rsplit('.',1)[0]}.jpg")
            )
        return

    # ==== !maskshield [thr=0.78] [pad=6] ====
    if message.content.strip().startswith("!maskshield"):
        parts = message.content.strip().split()
        thr = 0.78
        pad = 6
        if len(parts) >= 2:
            try: thr = float(parts[1])
            except: pass
        if len(parts) >= 3 and parts[2].isdigit():
            pad = int(parts[2])

        if not message.attachments:
            await message.channel.send("🖼 画像を添付して `!maskshield [閾値=0.78] [pad=6]` を実行してね")
            return

        for att in message.attachments:
            data = await att.read()
            img  = Image.open(io.BytesIO(data)).convert("RGB")
            bgr  = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            tpl = _load_or_make_template(bgr)
            if tpl is None:
                await message.channel.send("⚠️ 盾テンプレが作れませんでした（青い盾が見つからず）。別の画像で試してください。")
                continue

            boxes = find_shields_by_template(bgr, tpl, thr=thr)
            if not boxes:
                await message.channel.send("⚠️ 盾が検出できませんでした。`!maskshield 0.72` のように閾値を下げて試してね。")
                continue

            out_bgr = mask_row_right(bgr, boxes, pad_y=pad)

            ok, buf = cv2.imencode(".jpg", out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not ok:
                await message.channel.send("⚠️ 画像のエンコードに失敗しました")
                continue
            await message.channel.send(
                content=f"✅ 黒塗り完了: 盾{len(boxes)}個 / thr={thr} / pad={pad}  （右端まで横塗り）",
                file=discord.File(io.BytesIO(buf.tobytes()), filename=f"masked_{att.filename.rsplit('.',1)[0]}.jpg")
            )
        return
    # ==== !time 画像の撮影時刻を推定表示 ====
    if message.content.strip().startswith("!time"):
        if not message.attachments:
            await message.channel.send("🖼 画像を添付して `!time` を実行してね（`!time ocr` で画面内時計も表示）")
            return

        mode = message.content.strip().split(maxsplit=1)
        mode = mode[1].lower() if len(mode) == 2 else ""
        want_ocr = mode in ("ocr", "all")
        show_all  = mode == "all"

        lines = ["🕒 **撮影(作成)時刻の推定**"]
        up_jst = message.created_at.replace(tzinfo=timezone.utc).astimezone(JST)

        for i, att in enumerate(message.attachments, start=1):
            try:
                b = await att.read()
            except Exception:
                lines.append(f"#{i}: 読み込み失敗（{att.filename}）")
                continue

            # 1) EXIF / PNG
            dt_meta, how, raw = get_taken_time_from_image_bytes(b)

            # 2) OCR（任意）
            dt_ocr, ocr_raw = (None, "")
            if want_ocr:
                dt_ocr, ocr_raw = _ocr_clock_topright_to_jst(b)

            # 3) Discord アップロード時刻（常に用意）
            dt_disc = up_jst

            # 出力
            head = f"#{i} {att.filename}"
            if dt_meta:
                lines.append(f"{head}\n　📸 `{dt_meta.strftime('%Y-%m-%d %H:%M:%S')}` 〔{how} raw:{raw}〕")
                if show_all:
                    lines.append(f"　🕒 Discord送信 `{dt_disc.strftime('%Y-%m-%d %H:%M:%S')}`")
                    if want_ocr and dt_ocr:
                        lines.append(f"　👀 OCR時計 `{dt_ocr.strftime('%Y-%m-%d %H:%M:%S')}` (raw:{ocr_raw})")
            else:
                # メタが無い → 代替を並べる
                lines.append(f"{head}\n　🕒 EXIF/PNGなし → Discord送信 `{dt_disc.strftime('%Y-%m-%d %H:%M:%S')}`")
                if want_ocr and dt_ocr:
                    lines.append(f"　👀 OCR時計 `{dt_ocr.strftime('%Y-%m-%d %H:%M:%S')}` (raw:{ocr_raw})")

        await message.channel.send("\n".join(lines))
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

        h, w = np_img.shape[:2]
        longer = max(h, w)
        if longer > 1024:
            scale = 1024 / longer
            np_img = cv2.resize(np_img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

        # OCR前に「免戦中」直下を黒塗り
        np_img_masked, _masked_cnt = auto_mask_ime(np_img)

        # トリミング
        top = crop_top_right(np_img_masked)
        center = crop_center_area(np_img_masked)

        # OCRテキスト抽出
        top_txts = extract_text_from_image(top)
        center_txts = ocr_center_with_fallback(center)

        # 補正関数
        def extract_and_correct_base_time(txts):
            if not txts:
                return "??:??:??"
            raw = normalize_time_separators(txts[0])

            # まず HH:MM:SS を素直に拾う
            m = re.search(r"\b(\d{1,2}):(\d{2}):(\d{2})\b", raw)
            if m:
                h, mi, se = map(int, m.groups())
                if 0 <= h < 24 and 0 <= mi < 60 and 0 <= se < 60:
                    return f"{h:02}:{mi:02}:{se:02}"

            # 次に 6桁数字だけを HH:MM:SS に再構成
            digits = re.sub(r"\D", "", raw)
            if len(digits) == 6:
                h, mi, se = int(digits[:2]), int(digits[2:4]), int(digits[4:6])
                if 0 <= h < 24 and 0 <= mi < 60 and 0 <= se < 60:
                    return f"{h:02}:{mi:02}:{se:02}"

            # 4桁だけ見える場合は 00:MM:SS 扱い
            if len(digits) == 4:
                m_, s_ = int(digits[:2]), int(digits[2:])
                if 0 <= m_ < 60 and 0 <= s_ < 60:
                    return f"00:{m_:02}:{s_:02}"
    
            return "??:??:??"

        # 予定抽出
        parsed_preview = parse_multiple_places(center_txts, top_txts)
        preview_lines = [f"・{txt}" for _, txt, _ in parsed_preview] if parsed_preview else ["(なし)"]
        preview_text = "\n".join(preview_lines)

        # 免戦時間抽出 ＋ 補正
        durations = extract_imsen_durations(center_txts)
        duration_text = "\n".join(durations) if durations else "(抽出なし)"

        # OCR結果文字列
        top_txts_str = "\n".join(top_txts) if top_txts else "(検出なし)"
        center_txts_str = "\n".join(center_txts) if center_txts else "(検出なし)"

        # 画像を添付ファイル化
        files = []

        def _attach(bgr_img, filename, quality=92):
            try:
                ok, buf = cv2.imencode(".jpg", bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                if ok:
                    files.append(discord.File(io.BytesIO(buf.tobytes()), filename=filename))
            except Exception:
                pass

        _attach(np_img_masked, f"ocrdebug_full_masked_{a.filename.rsplit('.',1)[0]}.jpg", quality=92)
        _attach(top,        f"ocrdebug_top_{a.filename.rsplit('.',1)[0]}.jpg",              quality=92)
        _attach(center,     f"ocrdebug_center_{a.filename.rsplit('.',1)[0]}.jpg",           quality=95)

        # 送信（テキスト + 画像3枚）
        await message.channel.send(
            content=(
                f"📸 **上部OCR結果（基準時刻）**:\n```\n{top_txts_str}\n```\n"
                f"🧩 **中央OCR結果（補正前）**:\n```\n{center_txts_str}\n```\n"
                f"📋 **補正後の予定一覧（奪取 or 警備）**:\n```\n{preview_text}\n```\n"
                f"⏳ **補正後の免戦時間一覧**:\n```\n{duration_text}\n```\n"
                f"🧽 maskime: {_masked_cnt} 本\n"
                f"🖼 添付: 全体(黒塗り済) / 上部トリム / 中央トリム"
            ),
            files=files if files else None
        )
        return
        
    # ==== !srvdebug ====
    if message.content.strip() == "!srvdebug":
        if not message.attachments:
            await message.channel.send("🖼 画像を添付して `!srvdebug` を実行してね")
            return
        for att in message.attachments:
            data = await att.read()
            await _srvdebug_from_bytes(data, att.filename, message.channel.id)
        return

    # ==== !oaiocr（OpenAI：合成1枚ルート） ====
    if message.content.strip() == "!oaiocr":
        if OA_ASYNC is None and OA_CLIENT is None:
            await message.channel.send("⚠️ OpenAI が未初期化です。OPENAI_API_KEY を設定して再起動してください。")
            return
        if not message.attachments:
            await message.channel.send("🖼 画像を添付して `!oaiocr` を実行してね")
            return

        att = message.attachments[0]
        raw = await att.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        full_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # —— 新ルート：OpenAI 1枚OCR（合成して1枚だけ送る）
        j = await oai_ocr_oneimg_async(full_bgr)

        # ======================
        # 失敗時（Paddle/GVにフォールバック）
        # ======================
        if not j:
            masked, _ = auto_mask_ime(full_bgr)
            top_bgr    = crop_top_right(masked)
            center_bgr = crop_center_area(masked)

            top_txts    = extract_text_from_image(top_bgr) or []
            center_txts = ocr_center_with_fallback(center_bgr) or []

            base_clock_str = _extract_clock_from_top_txts(top_txts) or base_time_from_metadata(raw)
            # === 失敗時も合成からヘッダを読む（任意） ===
            if USE_COMPOSITE_FOR_HEADER:
                try:
                    comp_bgr_local, _ = compose_center_with_clock_and_cease(full_bgr)
                    if comp_bgr_local is not None:
                        Hc, Wc = comp_bgr_local.shape[:2]
                        y1c = int(Hc * COMP_HEAD_TOP);   y2c = int(Hc * COMP_HEAD_BOTTOM)
                        x1c = 0;                         x2c = int(Wc * COMP_HEAD_RIGHT)
                        head_src_bgr = comp_bgr_local[y1c:y2c, x1c:x2c]
            
                        Hh, Wh = head_src_bgr.shape[:2]
                        y1 = int(Hh * HEAD_TOP_RATIO); y2 = int(Hh * HEAD_BOTTOM_RATIO)
                        x1 = 0;                        x2 = int(Wh * HEAD_RIGHT_RATIO)
                        head_img_bgr = head_src_bgr[y1:y2, x1:x2]
            
                        srv_fb, _ = _triage_read_server_from_head(head_img_bgr)
                    else:
                        srv_fb = _extract_server_from_header(full_bgr)
                except Exception:
                    srv_fb = _extract_server_from_header(full_bgr)
            else:
                srv_fb = _extract_server_from_header(full_bgr)
            # --- 3エンジン比較デバッグ（失敗フォールバック時） ---
            # ヘッダ帯のBGR画像を用意
            H, W = full_bgr.shape[:2]
            y1 = int(H * HEAD_TOP_RATIO); y2 = int(H * HEAD_BOTTOM_RATIO)
            x1 = 0; x2 = int(W * HEAD_RIGHT_RATIO)
            head_img_bgr = full_bgr[y1:y2, x1:x2]
            
            if os.getenv("OAI_HEADER_DEBUG") == "1":
                srv_dbg, dbg = _triage_read_server_from_head(head_img_bgr)
                lines = [
                    "📚 **ヘッダ帯 3エンジン比較（!oaiocr fallback）**",
                    f"・Paddle: {repr(dbg['raw'].get('pp'))} → norm={dbg['norm'].get('pp')!r}",
                    f"・Google: {repr(dbg['raw'].get('gv'))} → norm={dbg['norm'].get('gv')!r}",
                    f"・OpenAI: {repr(dbg['raw'].get('oai'))} → norm={dbg['norm'].get('oai')!r}",
                    f"➡️ 採用: {srv_dbg!r}",
                ]
                await message.channel.send("\n".join(lines))
            parsed_preview = parse_multiple_places(
                center_txts, top_txts,
                base_time_override=base_clock_str,
                server_override=srv_fb
            )
            parsed_final = list(parsed_preview)
            durations = extract_imsen_durations(center_txts)

            # ← 追加：参考用の合成画像をローカル生成して添付
            files = []
            try:
                comp_bgr, _ = compose_center_with_clock_and_cease(full_bgr)
                ok, buf = cv2.imencode(".png", comp_bgr)
                if ok:
                    files.append(discord.File(io.BytesIO(buf.tobytes()), filename="oaiocr_composite_fallback.png"))
            except Exception:
                pass

            # 📤 テキストレポート + 画像添付
            await _send_oaiocr_text_report(
                message.channel,
                top_txts=top_txts,
                center_txts=center_txts,
                base_clock_str=base_clock_str,
                cease_hhmmss=None,
                parsed_preview=parsed_preview,
                parsed_final=parsed_final,
                durations=durations,
                cease_fix_applied_sec=0,
                cease_fix_threshold_sec=CEASEFIX_MAX_SEC,
                files=files,  # ← 追加
            )
            return

        # ======================
        # 成功時（OpenAI結果を整形）
        # ======================
        top_txts    = _coerce_str_lines(j.get("top_clock_lines"))
        center_txts = _coerce_str_lines(j.get("center_lines"))
        cease_str   = _first_str(j.get("ceasefire_end"))
        # === 合成からヘッダ帯を読む（USE_COMPOSITE_FOR_HEADER=1 のとき優先） ===
        head_src_bgr = full_bgr  # 既定は原寸
        if USE_COMPOSITE_FOR_HEADER:
            comp_png = ((j.get("_echo") or {}).get("composite_png") or None)
            if comp_png:
                comp_bgr = _bgr_from_png_bytes(comp_png)
            else:
                comp_bgr, _ = compose_center_with_clock_and_cease(full_bgr)
            if comp_bgr is not None:
                Hc, Wc = comp_bgr.shape[:2]
                y1c = int(Hc * COMP_HEAD_TOP);    y2c = int(Hc * COMP_HEAD_BOTTOM)
                x1c = 0;                           x2c = int(Wc * COMP_HEAD_RIGHT)
                head_src_bgr = comp_bgr[y1c:y2c, x1c:x2c]
        
        # —— 3エンジン比較＋「一番後ろの番号」ルールで確定
        server_from_head, dbg = _triage_read_server_from_head(head_src_bgr)
        
        # OpenAI の structured.server でも補完して最終採用
        try:
            server_struct = _normalize_server4((j.get("structured") or {}).get("server"))
        except NameError:
            server_struct = _normalize_server((j.get("structured") or {}).get("server"))
        server_final = server_from_head or server_struct
        # まず全ROIをフル画像から一発計算（ネスト禁止）
        rects = _calc_regions(full_bgr)
        
        # 時計は必ず「フル画像の clock_rect」から読む
        base_clock_roi  = _read_clock_from_roi_strict(full_bgr, rects["clock"])
        
        # OpenAIの上部行（停戦/日付混じりは除外）
        base_clock_ocr  = _extract_clock_from_top_txts(top_txts)
        
        # 本文からの拾い上げ（あれば）
        base_clock_cent = _extract_clock_from_center_lines(center_txts) if ' _extract_clock_from_center_lines' in globals() else None
        
        # 最後はメタデータ
        base_clock_meta = base_time_from_metadata(raw)
        
        base_clock_str = base_clock_roi or base_clock_ocr or base_clock_cent or base_clock_meta
        # --- 時計の多段フォールバック（center→ROI） ---
        if not base_clock_str:
            # 1) center_lines に出てしまった HH:MM:SS を拾う（免戦的な 00:MM:SS などは劣後）
            clock_from_center = _extract_clock_from_center_lines(center_txts)
            if clock_from_center:
                base_clock_str = clock_from_center
                # 誤結合防止のため、見つけた時刻は center_lines から除去（任意）
                cleaned = []
                for ln in center_txts:
                    s = str(ln)
                    if clock_from_center in s:
                        s = s.replace(clock_from_center, "").strip()
                    if s:
                        cleaned.append(s)
                center_txts = cleaned
        
        if not base_clock_str:
            # 2) 右上ROIをローカルOCRで読む（Paddle/Google）
            base_clock_str = _ocr_clock_from_roi(full_bgr)
        # === 合成からヘッダ帯を読む（USE_COMPOSITE_FOR_HEADER=1 のとき優先） ===
        head_src_bgr = full_bgr  # 既定は原寸
        if USE_COMPOSITE_FOR_HEADER:
            comp_png = ((j.get("_echo") or {}).get("composite_png") or None)
            if comp_png:
                comp_bgr = _bgr_from_png_bytes(comp_png)
                if comp_bgr is not None:
                    Hc, Wc = comp_bgr.shape[:2]
                    y1c = int(Hc * COMP_HEAD_TOP);   y2c = int(Hc * COMP_HEAD_BOTTOM)
                    x1c = 0;                         x2c = int(Wc * COMP_HEAD_RIGHT)
                    head_src_bgr = comp_bgr[y1c:y2c, x1c:x2c]
        
        # （!srvdebugと同じ比率で最終クロップ）
        Hh, Wh = head_src_bgr.shape[:2]
        y1 = int(Hh * HEAD_TOP_RATIO); y2 = int(Hh * HEAD_BOTTOM_RATIO)
        x1 = 0;                        x2 = int(Wh * HEAD_RIGHT_RATIO)
        head_img_bgr = head_src_bgr[y1:y2, x1:x2]
        
        # 3エンジン比較でサーバ確定（“最後の番号”ルール込み）
        try:
            server_from_head, dbg = _triage_read_server_from_head(head_img_bgr)
        except NameError:
            server_from_head = _extract_server_from_header(full_bgr)
            dbg = {"raw": {}, "norm": {}, "winner": "fallback(_extract_server_from_header)"}
        
        try:
            server_struct = _normalize_server4((j.get("structured") or {}).get("server"))
        except NameError:
            server_struct = _normalize_server((j.get("structured") or {}).get("server"))
        
        server_final = server_from_head or server_struct
        # ----- Center 追OCR補完：OpenAI が場所を落としたときだけ実行 -----
        places0 = extract_places_from_center(center_txts)
        if not places0:
            aux_lines = []
            # OpenAIの合成PNGが返ってきていれば、それを優先してOCR
            comp_bgr = None
            try:
                comp_png = ((j.get("_echo") or {}).get("composite_png") or None)
                if comp_png:
                    arr = np.frombuffer(comp_png, np.uint8)
                    comp_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception:
                comp_bgr = None
        
            if comp_bgr is not None:
                aux_lines = _ocr_lines_pp_gv(comp_bgr)
            else:
                # 無ければ原寸のセンター帯で追OCR
                try:
                    center_bgr = crop_center_area(full_bgr)
                except Exception:
                    center_bgr = None
                if center_bgr is not None:
                    aux_lines = _ocr_lines_pp_gv(center_bgr)
        
            if aux_lines:
                # 追OCRの行を統合（重複除去）
                center_txts = list(dict.fromkeys(list(center_txts) + list(aux_lines)))
        
                if os.getenv("OAI_HEADER_DEBUG") == "1":
                    preview = "\n".join(aux_lines[:15])
                    await message.channel.send(
                        "🔁 追加OCR(center Paddle+Google)\n```\n" + preview + "\n```"
                    )
        
        if os.getenv("OAI_HEADER_DEBUG") == "1":
            lines = [
                "📚 **ヘッダ帯 3エンジン比較（!oaiocr success）**",
                f"・Paddle: {repr(dbg.get('raw',{}).get('pp'))} → norm={dbg.get('norm',{}).get('pp')!r}",
                f"・Google: {repr(dbg.get('raw',{}).get('gv'))} → norm={dbg.get('norm',{}).get('gv')!r}",
                f"・OpenAI: {repr(dbg.get('raw',{}).get('oai'))} → norm={dbg.get('norm',{}).get('oai')!r}",
                f"➡️ 採用(server_final): {server_final!r}",
            ]
            await message.channel.send("\n".join(lines))
        parsed_preview = parse_multiple_places(
            center_txts, top_txts,
            base_time_override=base_clock_str,
            server_override=server_final
        )
        parsed = list(parsed_preview)
            
        # rows / center からのフォールバック復元（ダミー生成なし）
        if not parsed:
            srv = server_final  # ヘッダ確定サーバ（なければ何もしない）
            rows = ((j.get("structured") or {}).get("rows") or [])
        
            # 1) structured.rows があれば使う
            if srv and base_clock_str and rows:
                mode = "警備" if srv == "1268" else "奪取"
                for r in rows:
                    place = _first_str(r.get("place"))
                    dur   = correct_imsen_text(_first_str(r.get("duration") or ""))
                    if not (place and dur):
                        continue
                    dt, unlock = add_time(base_clock_str, dur)
                    if dt:
                        parsed.append((dt, f"{mode} {srv}-{place}-{unlock}", dur))
        
            # 2) rows が無い/不足 → 中央本文から復元（場所×免戦を上から順にペアリング）
            if not parsed and srv and base_clock_str:
                mode = "警備" if srv == "1268" else "奪取"
                places = extract_places_from_center(center_txts)       # ← 駐騎/駐車 両対応にしておく
                durs   = extract_imsen_durations(center_txts) or []
                n = min(len(places), len(durs))
                for i in range(n):
                    place = str(places[i])
                    dur   = correct_imsen_text(durs[i])
                    dt, unlock = add_time(base_clock_str, dur)
                    if dt:
                        parsed.append((dt, f"{mode} {srv}-{place}-{unlock}", dur))
        # 3) それでも parsed が空なら、何も登録しない（ダミー禁止）
        # --- 最終ガード：server一致 & place>0 以外は捨てる（フォールバック後に適用） ---
        if server_final and parsed:
            filtered = []
            for dt, txt, raw_dur in parsed:
                m = re.search(r'^\S+\s+(\d{3,5})-([0-9]+)-', txt)
                if not m:
                    continue
                srv_txt, place_txt = m.group(1), m.group(2)
                if srv_txt != server_final:
                    continue
                if place_txt in ("0", ""):
                    continue
                filtered.append((dt, txt, raw_dur))
            parsed = filtered

        # ===== 登録処理（既存設計に合わせる）=====
        image_results = []
        structured_entries_for_this_image = []
        for dt, txt, raw_dur in parsed:
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
                await auto_dedup()
                pending_copy_queue.append((dt, txt))
                image_results.append(f"{txt} ({raw_dur})")

                task = asyncio.create_task(handle_new_event(dt, txt, client.get_channel(NOTIFY_CHANNEL_ID)))
                active_tasks.add(task); task.add_done_callback(lambda t: active_tasks.discard(t))
                if txt.startswith("奪取"):
                    t2 = asyncio.create_task(schedule_notification(dt, txt, client.get_channel(NOTIFY_CHANNEL_ID)))
                    active_tasks.add(t2); t2.add_done_callback(lambda t: active_tasks.discard(t))

        gid = None
        if structured_entries_for_this_image:
            last_groups_seq += 1
            gid = last_groups_seq
            last_groups[gid] = structured_entries_for_this_image

        durations = extract_imsen_durations(center_txts)

        # ← 追加：OpenAIが返した合成PNG（_echo）をそのまま添付、無ければローカル合成を添付
        files = []
        comp_png = ((j.get("_echo") or {}).get("composite_png") or None)
        if comp_png:
            files.append(_file_from_bytes("oaiocr_composite.png", comp_png))
        else:
            try:
                comp_bgr, _ = compose_center_with_clock_and_cease(full_bgr)
                ok, buf = cv2.imencode(".png", comp_bgr)
                if ok:
                    files.append(discord.File(io.BytesIO(buf.tobytes()), filename="oaiocr_composite_fallback.png"))
            except Exception:
                pass

        await _send_oaiocr_text_report(
            message.channel,
            top_txts=top_txts,
            center_txts=center_txts,
            base_clock_str=base_clock_str,
            cease_hhmmss=cease_str,
            parsed_preview=parsed_preview,
            parsed_final=parsed,
            durations=durations,
            cease_fix_applied_sec=0,
            cease_fix_threshold_sec=CEASEFIX_MAX_SEC,
            files=files,  # ← 追加
        )
        return

    # ==== !gvocr（Google VisionのみでOCRデバッグ表示） ====
    if message.content.strip() == "!gvocr":
        if GV_CLIENT is None:
            await message.channel.send("⚠️ Google Vision が未初期化です（環境変数 GOOGLE_CLOUD_VISION_JSON を確認）")
            return

        if not message.attachments:
            await message.channel.send("⚠️ 画像を添付してください（GVのみでOCRします）")
            return

        a = message.attachments[0]
        b = await a.read()
        img = Image.open(io.BytesIO(b)).convert("RGB")
        np_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # ← これが必要！

        # OCR前に「免戦中」直下を黒塗り（GV専用デバッグでも適用）
        np_img, _ = auto_mask_ime(np_img)

        # トリミング
        top = crop_top_right(np_img)
        center = crop_center_area(np_img)

        # ★ GV のみでOCR
        top_txts = await extract_text_from_image_google_async(top)
        center_txts = ocr_center_google(center)  # ここもGV専用（Paddle不使用）

        # 予定抽出（既存のパーサをそのまま利用）
        parsed_preview = parse_multiple_places(center_txts, top_txts)
        preview_lines = [f"・{txt}" for _, txt, _ in parsed_preview] if parsed_preview else ["(なし)"]
        preview_text = "\n".join(preview_lines)

        # 免戦時間（参考表示）
        durations = extract_imsen_durations(center_txts)
        duration_text = "\n".join(durations) if durations else "(抽出なし)"

        # 出力
        top_txts_str = "\n".join(top_txts) if top_txts else "(検出なし)"
        center_txts_str = "\n".join(center_txts) if center_txts else "(検出なし)"

        await message.channel.send(
            f"📸 **[GV] 上部OCR（基準時刻）**:\n```\n{top_txts_str}\n```\n"
            f"🧩 **[GV] 中央OCR（補正前）**:\n```\n{center_txts_str}\n```\n"
            f"📋 **[GV] 補正後の予定一覧**:\n```\n{preview_text}\n```\n"
            f"⏳ **[GV] 免戦時間候補**:\n```\n{duration_text}\n```"
        )
        return
        
    # ==== !glist 現在のグループ一覧表示 ====
    if message.content.strip() == "!glist":
        if not last_groups:
            sent = await message.channel.send("⚠️ 現在グループはありません。まず画像を送って解析してください。")
            asyncio.create_task(auto_delete_after(sent, GLIST_TTL))
            return

        lines = ["📸 現在の画像グループ:"]
        for gid, events in last_groups.items():
            lines.append(f"　G{gid}:")
            for e in events:
                lines.append(f"　　・{e['server']}-{e['place']}-{e['dt'].strftime('%H:%M:%S')}")

        sent = await message.channel.send("\n".join(lines))
        asyncio.create_task(auto_delete_after(sent, GLIST_TTL))
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

        # ---- new_txt が既に存在する場合はマージ（重複防止）----
        existed = new_txt in pending_places
        if existed:
            tgt = pending_places[new_txt]
            # 可能なら旧のIDを引き継ぎ（空欄のみ上書き）
            if old_main_msg_id and not tgt.get("main_msg_id"):
                tgt["main_msg_id"] = old_main_msg_id
            if old_copy_msg_id and not tgt.get("copy_msg_id"):
                tgt["copy_msg_id"] = old_copy_msg_id
            # 時刻は新指定で上書き（同一キーなので実質同時刻のはずだが念のため）
            tgt["dt"] = new_dt
        else:
            pending_places[new_txt] = {
                "dt": new_dt,
                "txt": new_txt,
                "server": server,
                "created_at": now_jst(),
                "main_msg_id": old_main_msg_id,  # まとめメッセのID（retime側で再設定されうる）
                "copy_msg_id": old_copy_msg_id,  # コピー用メッセのID（あれば編集で使う）
            }

        # ---- 通知チャンネルのまとめメッセ：古い行を削除→新行を時刻順に追加（編集で更新） ----
        await retime_event_in_summary(old_txt, new_dt, new_txt, channel)

        # ---- コピーチャンネル：旧メッセがあれば内容だけ new に編集（自動新規はしない）----
        if old_copy_msg_id:
            copy_ch = client.get_channel(COPY_CHANNEL_ID)
            if copy_ch:
                try:
                    msg = await copy_ch.fetch_message(old_copy_msg_id)
                    await msg.edit(content=new_txt.replace("🕒 ", ""))
                    pending_places[new_txt]["copy_msg_id"] = msg.id
                except discord.NotFound:
                    # 見つからない場合は諦める（同期タスクで再整合される）
                    pending_places[new_txt]["copy_msg_id"] = None
                except Exception:
                    pass

        # ---- 通知再登録（奪取のみ）----
        if mode == "奪取":
            # new_txt がすでに存在していた場合は既存予約が生きている可能性が高いので二重登録しない
            if not existed:
                await schedule_notification(new_dt, new_txt, channel)

        # 手動まとめ(!s)があれば編集で最新化。コピーチャンネルも定期同期で整合。
        await refresh_manual_summaries()
        await upsert_copy_channel_sorted([])

        await message.channel.send(f"🛠 修正しました: {old_txt} → {new_txt}")
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
        filenames = [att.filename for att in message.attachments]
        
        for a in message.attachments:
            structured_entries_for_this_image = []  # ← !g用
            b = await a.read()
            img = Image.open(io.BytesIO(b)).convert("RGB")
            np_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            # OCR前に「免戦中」直下を黒塗り
            np_img, _ = auto_mask_ime(np_img)

            # トリミング & OCR
            top = crop_top_right(np_img)
            center = crop_center_area(np_img)
            top_txts_ocr = extract_text_from_image(top)          # まず Paddle
            center_txts  = ocr_center_with_fallback(center)      # Paddle→GV

            # ★ 基準時刻：メタ優先
            meta_base = base_time_from_metadata(b)  # -> "HH:MM:SS" or None
            if meta_base:
                base_time = meta_base
                parsed = parse_multiple_places(center_txts, top_txts_ocr, base_time_override=base_time)
                base_annot = "(meta)"
            else:
                # 右上OCRの先頭行から HH:MM:SS を復元（フォールバック）
                def _extract_and_correct_base_time(txts):
                    if not txts: return "??:??:??"
                    raw = normalize_time_separators(txts[0].strip())
                    raw = force_hhmmss_if_six_digits(raw)
                    m = re.search(r"\b(\d{1,2}):(\d{2}):(\d{2})\b", raw)
                    if m:
                        h, mi, se = map(int, m.groups())
                        if 0 <= h < 24 and 0 <= mi < 60 and 0 <= se < 60:
                            return f"{h:02}:{mi:02}:{se:02}"
                    digits = re.sub(r"\D","", raw)
                    if len(digits) == 4:
                        m_, s_ = int(digits[:2]), int(digits[2:])
                        if 0 <= m_ < 60 and 0 <= s_ < 60:
                            return f"00:{m_:02}:{s_:02}"
                    return "??:??:??"

                base_time = _extract_and_correct_base_time(top_txts_ocr)
                parsed = parse_multiple_places(center_txts, top_txts_ocr)
                base_annot = "(ocr)"

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
                    # ✅ 自動重複除去（同じサーバー・駐騎場で後の時刻を削除）
                    await auto_dedup()
                    pending_copy_queue.append((dt, txt))
                    display_txt = f"{txt} ({raw})"
                    image_results.append(display_txt)

                    task = asyncio.create_task(handle_new_event(dt, txt, channel))
                    active_tasks.add(task); task.add_done_callback(lambda t: active_tasks.discard(t))
                    if txt.startswith("奪取"):
                        task2 = asyncio.create_task(schedule_notification(dt, txt, channel))
                        active_tasks.add(task2); task2.add_done_callback(lambda t: active_tasks.discard(t))

            if structured_entries_for_this_image:
                last_groups_seq += 1
                gid = last_groups_seq
                last_groups[gid] = structured_entries_for_this_image
                if image_results:
                    # 注記を表示に反映
                    grouped_results.append((gid, f"{base_time} {base_annot}", image_results))

        if grouped_results:
            lines = [
                f"✅ 解析完了！ `{' ,'.join(filenames)}` を登録しました",
                "",
                "🖼 実際の時間と異なる場合はスクショを撮り直してください",
                "⏱ 1秒程度のズレは 🔧 `!g` で修正可能",
                "🛠 大幅なズレは `!a` で修正してください",
                "",
            ]
            for gid, base_time_str, txts in grouped_results:
                lines.append(f"📸 [G{gid} | 基準時間: {base_time_str}]")
                lines += [f"・{txt}" for txt in txts]
                lines.append("")
            await status.edit(content="\n".join(lines))

# =======================
# 起動
# =======================
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