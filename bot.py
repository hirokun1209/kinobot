import os
import io
import re
import base64
import asyncio
import unicodedata
from typing import List, Tuple, Dict, Optional

import discord
from discord.ext import commands
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
# 通知を投げるチャンネル
NOTIFY_CHANNEL_ID = int(os.environ.get("NOTIFY_CHANNEL_ID", "0") or 0)
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

# 正規表現（緩め）
RE_IMMUNE = re.compile(r"免\s*戦\s*中")                                 # 「免戦中」
RE_TITLE  = re.compile(r"越\s*域\s*駐[\u4E00-\u9FFF]{1,3}\s*場")         # 「越域駐〇場」誤OCRも拾う
RE_TIME   = re.compile(r"\d{1,2}[:：]\d{2}(?:[:：]\d{2})?")              # 05:53 / 01:02:13 など
RE_SERVER = re.compile(r"\[?\s*[sS]\s*([0-9]{2,5})\]?")                 # [s1296] / s1296

# フォールバック：ブロック高さに対する“必ず残す”上部割合
FALLBACK_KEEP_TOP_RATIO = 0.35

# スケジュールタスク保持（GC防止）
SCHEDULED_TASKS: List[asyncio.Task] = []

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
    """Google Vision で word 単位の文字とバウンディングボックスを返す (text, (x1,y1,x2,y2))"""
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
    7番ブロック内で、
      ・各「越域駐〇場」行をタイトル
      ・タイトルiの下端〜タイトルi+1の上端を“ワンブロック”
      ・ブロック内に「免戦中」or 時間があればその下端まで“残す”、以降は削除（詰める）
    """
    im = pil_im.copy()
    w, h = im.size

    lines = google_ocr_line_boxes(im, y_tol=18)

    titles: List[Tuple[int,int]] = []
    candidates: List[Tuple[int,int]] = []

    for text, (x1, y1, x2, y2) in lines:
        t = _norm(text)
        if RE_TITLE.search(t):
            titles.append((y1, y2))
        if RE_IMMUNE.search(t) or RE_TIME.search(t):
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
# 計算 & フォーマット
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
            return a*60 + b      # MM:SS として扱う（例: 58:40 -> 00:58:40）
        return a*3600 + b*60     # HH:MM（基準/停戦）
    return 0

def _seconds_to_hms(sec: int) -> str:
    sec %= 24*3600
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def parse_and_compute(oai_text: str) -> Tuple[Optional[str], Optional[str], Optional[str], List[Tuple[int, str]]]:
    """
    OCRテキストから
      server(str), base_time(HH:MM:SS), ceasefire(HH:MM:SS), results[(place, time_str)]
    を返す。足りない場合は None。
    """
    lines = [ln.strip() for ln in oai_text.splitlines() if ln.strip()]
    if not lines:
        return None, None, None, []

    server = None
    base_time_sec: Optional[int] = None
    ceasefire_sec: Optional[int] = None

    pairs: List[Tuple[int, Optional[int]]] = []  # (place, immune_sec)

    def find_time_in_text(txt: str) -> Optional[str]:
        m = RE_TIME.search(txt)
        return m.group(0).replace("：", ":") if m else None

    # 1周: サーバー / 基準 / 停戦終了 / 越域駐〇場 + 免戦 を順に拾う
    for raw in lines:
        n = _norm(raw)
        # server
        if server is None:
            m = RE_SERVER.search(n)
            if m:
                server = m.group(1)

        # ceasefire: 行内に「停戦」があればその行の時刻
        if "停戦" in n:
            tt = find_time_in_text(raw)
            if tt:
                ceasefire_sec = _time_to_seconds(tt, prefer_mmss=False)

        # base_time: 「免戦」「停戦」を含まない最初の時刻
        if base_time_sec is None and ("免戦" not in n and "停戦" not in n):
            tt = find_time_in_text(raw)
            if tt:
                base_time_sec = _time_to_seconds(tt, prefer_mmss=False)

        # title: 越域駐〇場
        if RE_TITLE.search(n):
            m_num = re.search(r"場\s*([0-9]{1,3})", raw)
            if not m_num:
                m_num = re.search(r"([0-9]{1,3})\s*$", raw)
            if m_num:
                place = int(m_num.group(1))
                pairs.append((place, None))

        # immune time（2 区切りは MM:SS として解釈）
        if "免戦" in n:
            tt = find_time_in_text(raw)
            if tt:
                tsec = _time_to_seconds(tt, prefer_mmss=True)
                # 直近の未設定ペアに充当
                for i in range(len(pairs)-1, -1, -1):
                    if pairs[i][1] is None:
                        pairs[i] = (pairs[i][0], tsec)
                        break

    if base_time_sec is None or not pairs:
        return server, None, None, []

    # 計算
    calc: List[Tuple[int, int]] = []  # (place, sec_from_midnight)
    for place, immune in pairs:
        if immune is None:
            continue
        calc.append((place, (base_time_sec + immune) % (24*3600)))

    if not calc:
        return server, _seconds_to_hms(base_time_sec), _seconds_to_hms(ceasefire_sec) if ceasefire_sec is not None else None, []

    # 最上ブロック（最初のcalc）と停戦終了の差で補正
    if ceasefire_sec is not None:
        delta = (ceasefire_sec - calc[0][1])  # 正負OK
        calc = [(pl, (sec + delta) % (24*3600)) for (pl, sec) in calc]
        # 先頭は停戦終了に合わせる
        calc[0] = (calc[0][0], ceasefire_sec % (24*3600))

    # 出力整形
    results: List[Tuple[int, str]] = [(pl, _seconds_to_hms(sec)) for (pl, sec) in calc]
    base_str = _seconds_to_hms(base_time_sec)
    cease_str = _seconds_to_hms(ceasefire_sec) if ceasefire_sec is not None else None
    return server, base_str, cease_str, results

def build_result_message(server: Optional[str],
                         base_str: Optional[str],
                         cease_str: Optional[str],
                         results: List[Tuple[int, str]]) -> str:
    # 例） ✅ 解析完了！⏱️ 基準時間:17:26:45 (21:07:21)
    if not base_str or not results or not server:
        return "⚠️ 解析完了… ですが計算できませんでした。画像やOCR結果をご確認ください。"

    head = f"✅ 解析完了！⏱️ 基準時間:{base_str}"
    if cease_str:
        head += f" ({cease_str})"

    body_lines = [f"{server}-{pl}-{t}" for (pl, t) in results]
    return head + "\n" + "\n".join(body_lines)

# ---------------------------
# パイプライン
# ---------------------------

def process_image_pipeline(pil_im: Image.Image) -> Tuple[Image.Image, str, str, List[Tuple[int, str]]]:
    """
    リサイズ→スライス→トリム→（7を詰め処理）→合成→OpenAI OCR→計算
    戻り値: (最終合成画像, 結果メッセージ, server, results)
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

    server, base_str, cease_str, results = parse_and_compute(oai_text)
    message = build_result_message(server, base_str, cease_str, results)

    return final_img, message, server or "", results

# ---------------------------
# スケジューラ
# ---------------------------

def _next_occurrence_today_or_tomorrow(hms: str) -> datetime:
    """今日のその時刻、過ぎていれば翌日の同時刻（TZ考慮）"""
    now = datetime.now(TIMEZONE)
    hh, mm, ss = map(int, hms.split(":"))
    candidate = now.replace(hour=hh, minute=mm, second=ss, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate

async def _schedule_message(channel_id: int, when_dt: datetime, content: str):
    """指定時刻にメッセージを投げる"""
    async def _runner():
        try:
            while True:
                now = datetime.now(TIMEZONE)
                delay = (when_dt - now).total_seconds()
                if delay <= 0:
                    break
                await asyncio.sleep(min(delay, 60))  # 1分刻みで近づく（長時間sleepの安全策）
            channel = bot.get_channel(channel_id) or await bot.fetch_channel(channel_id)
            await channel.send(content)
        except Exception as e:
            print(f"[scheduler] error: {e}")

    task = asyncio.create_task(_runner())
    SCHEDULED_TASKS.append(task)

def schedule_notifications(server: str, results: List[Tuple[int, str]]):
    """結果の各時刻を通知専用チャンネルへスケジュール登録"""
    if NOTIFY_CHANNEL_ID <= 0 or not server or not results:
        return
    for place, hms in results:
        when_dt = _next_occurrence_today_or_tomorrow(hms)
        content = f"⏰ 通知: **{server}-{place}-{hms}** になりました！"
        _schedule_message(NOTIFY_CHANNEL_ID, when_dt, content)

# ---------------------------
# 共通実行（複数画像対応）
# ---------------------------

IMAGE_MIME_PREFIXES = ("image/",)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

def _is_image_attachment(a: discord.Attachment) -> bool:
    if a.content_type and any(a.content_type.startswith(p) for p in IMAGE_MIME_PREFIXES):
        return True
    return a.filename.lower().endswith(IMAGE_EXTS)

async def run_pipeline_for_attachments(atts: List[discord.Attachment], *, want_image: bool) -> Tuple[Optional[discord.File], str]:
    """複数画像を処理。want_image=True の時は画像1枚（縦結合）を返す。"""
    images: List[Image.Image] = []
    messages: List[str] = []
    merged_results: List[Tuple[int, str]] = []
    server_for_notify: Optional[str] = None

    loop = asyncio.get_event_loop()

    for a in atts:
        data = await a.read()
        pil = load_image_from_bytes(data)
        final_img, msg, server, results = await loop.run_in_executor(None, process_image_pipeline, pil)
        images.append(final_img)
        messages.append(msg)
        if server and not server_for_notify:
            server_for_notify = server
        # スケジュール登録
        schedule_notifications(server, results)

    # テキストは連結
    full_message = "\n\n".join(messages)

    # 画像は1枚にまとめる or 返さない
    if want_image and images:
        merged = vstack_uniform_width(images, width=TARGET_WIDTH)
        out = io.BytesIO()
        merged.convert("RGB").save(out, format="PNG")
        out.seek(0)
        return discord.File(out, filename="result.png"), full_message

    return None, full_message

# ---------------------------
# Commands（デバッグ）
# ---------------------------

@bot.command(name="oaiocr", help="画像を添付して実行。処理→詰め→OpenAI OCR→計算（複数画像OK）。")
async def oaiocr(ctx: commands.Context):
    try:
        atts = [a for a in ctx.message.attachments if _is_image_attachment(a)]
        if not atts:
            await ctx.reply("画像を添付して `!oaiocr` を実行してください。")
            return

        # まずは即レス（のちに編集）
        placeholder = await ctx.reply("解析中…🔎")

        fileobj, message = await run_pipeline_for_attachments(atts, want_image=True)

        # 結果に編集差し替え。画像は別送（1枚に統合）
        await placeholder.edit(content=message)
        if fileobj:
            await ctx.send(file=fileobj)

    except Exception as e:
        await ctx.reply(f"エラー: {e}")

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

            # 解析（画像は返さない）
            _, result_text = await run_pipeline_for_attachments(atts, want_image=False)

            # プレースホルダを編集
            await placeholder.edit(content=result_text)

            # on_message の最後にコマンド処理（通常は不要だが念のため）
            return

        # その他はそのまま
        await bot.process_commands(message)

    except Exception as e:
        try:
            await message.channel.send(f"エラー: {e}")
        except Exception:
            pass

# ---------------------------
# Ping
# ---------------------------

@bot.command(name="ping")
async def ping(ctx: commands.Context):
    await ctx.reply("pong 🏓")

# ---------------------------
# Run
# ---------------------------

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)