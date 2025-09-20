import os
import io
import re
import base64
import asyncio
from typing import List, Tuple, Dict, Optional

import discord
from discord.ext import commands
from PIL import Image, ImageDraw, ImageOps

# Google Vision
from google.cloud import vision

# OpenAI (official SDK v1)
from openai import OpenAI

# ---------------------------
# ENV/bootstrap
# ---------------------------
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GOOGLE_CLOUD_VISION_JSON = os.environ.get("GOOGLE_CLOUD_VISION_JSON", "")

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

# 正規表現（OCR解析兼 詰め処理）
RE_IMMUNE = re.compile(r"免\s*戦\s*中")
RE_TITLE  = re.compile(r"越\s*域\s*駐\s*[騎機車]\s*場")

# 余白（必要なら調整）
MARGIN_AFTER_IMMUNE = 0           # 免戦中の直下 +0px（＝下端から）
MARGIN_BEFORE_NEXT_TITLE = 0      # 次タイトル直上 -0px

# ---------------------------
# Helpers（画像）
# ---------------------------

def load_image_from_bytes(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    im = ImageOps.exif_transpose(im)  # EXIFの向きを補正
    return im.convert("RGBA")

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

def google_ocr_word_boxes(pil_im: Image.Image):
    """Google Vision で word 単位のテキストとBBoxを返す [(text,(x1,y1,x2,y2)), ...]"""
    buf = io.BytesIO()
    pil_im.convert("RGB").save(buf, format="JPEG", quality=95)
    image = vision.Image(content=buf.getvalue())
    response = gcv_client.document_text_detection(image=image)
    if response.error.message:
        raise RuntimeError(response.error.message)

    words = []
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

def google_ocr_line_boxes(pil_im: Image.Image, y_tol: int = 14):
    """wordをY座標で行グループ化して (line_text, (x1,y1,x2,y2)) を返す"""
    words = google_ocr_word_boxes(pil_im)
    if not words:
        return []

    items = []
    for txt, (x1, y1, x2, y2) in words:
        cy = (y1 + y2) / 2.0
        items.append((cy, x1, y1, x2, y2, txt))
    items.sort(key=lambda t: (t[0], t[1]))

    lines = []
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

    line_boxes = []
    for chunks in lines:
        chunks.sort(key=lambda a: a[0])  # x1
        text = "".join(c[4] for c in chunks)
        x1 = min(c[0] for c in chunks)
        y1 = min(c[1] for c in chunks)
        x2 = max(c[2] for c in chunks)
        y2 = max(c[3] for c in chunks)
        line_boxes.append((text, (x1, y1, x2, y2)))
    return line_boxes

def compute_remove_ranges_for_7(pil_im: Image.Image) -> List[Tuple[int, int]]:
    """
    7番ブロック内で削除(=詰め)すべき縦範囲を計算して返す [(top,bottom), ...]
     ・各「越域駐◯場」行をタイトル
     ・タイトルiの下端〜タイトルi+1の上端を“ワンブロック”
     ・ブロック内に「免戦中」があれば その下端〜次タイトル上端
     ・免戦中がなければ タイトル下端〜次タイトル上端
     ・最後のブロックは下端まで
    """
    w, h = pil_im.size
    lines = google_ocr_line_boxes(pil_im)

    titles: List[Tuple[int, int]] = []
    immunes: List[Tuple[int, int]] = []

    for text, (x1, y1, x2, y2) in lines:
        t = text.replace(" ", "")
        if RE_TITLE.search(t):
            titles.append((y1, y2))
        if RE_IMMUNE.search(t):
            immunes.append((y1, y2))

    titles.sort(key=lambda p: p[0])
    immunes.sort(key=lambda p: p[0])

    if not titles:
        return []  # 何も削除しない

    remove_ranges: List[Tuple[int, int]] = []

    for i, (t_y1, t_y2) in enumerate(titles):
        start = t_y2
        end = titles[i + 1][0] if i + 1 < len(titles) else h
        if end <= start:
            continue

        immune_bottom_in_block = None
        for iy1, iy2 in immunes:
            if start <= iy1 < end:
                immune_bottom_in_block = iy2  # 最後に出たもの
        if immune_bottom_in_block is not None:
            top = immune_bottom_in_block + MARGIN_AFTER_IMMUNE
        else:
            top = start

        bottom = max(top, end - MARGIN_BEFORE_NEXT_TITLE)
        if bottom > top:
            remove_ranges.append((top, bottom))

    # マージ（重なり/隣接）
    merged: List[Tuple[int, int]] = []
    for a, b in sorted(remove_ranges):
        if not merged or a > merged[-1][1]:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)
    return [(int(a), int(b)) for a, b in merged]

def remove_vertical_ranges(im: Image.Image, ranges: List[Tuple[int, int]]) -> Image.Image:
    """指定縦範囲を削除して上に詰める"""
    if not ranges:
        return im.copy()
    w, h = im.size
    keep_parts = []
    y = 0
    for top, bottom in ranges:
        top = max(0, min(top, h))
        bottom = max(0, min(bottom, h))
        if top > y:
            keep_parts.append(im.crop((0, y, w, top)))
        y = max(y, bottom)
    if y < h:
        keep_parts.append(im.crop((0, y, w, h)))
    new_h = sum(p.height for p in keep_parts)
    canvas = Image.new("RGBA", (w, new_h), (0, 0, 0, 0))
    yy = 0
    for p in keep_parts:
        canvas.paste(p, (0, yy))
        yy += p.height
    return canvas

def compress_7_block(pil_im: Image.Image) -> Image.Image:
    """7番ブロックの黒塗り対象領域を“削除して詰め”た画像を返す"""
    ranges = compute_remove_ranges_for_7(pil_im)
    return remove_vertical_ranges(pil_im, ranges)

def hstack(im_left: Image.Image, im_right: Image.Image, gap: int = 8, bg=(0,0,0,0)) -> Image.Image:
    h = max(im_left.height, im_right.height)
    w = im_left.width + gap + im_right.width
    canvas = Image.new("RGBA", (w, h), bg)
    y1 = (h - im_left.height)//2
    y2 = (h - im_right.height)//2
    canvas.paste(im_left, (0, y1))
    canvas.paste(im_right, (im_left.width + gap, y2))
    return canvas

def vstack(images: List[Image.Image], gap: int = 8, bg=(0,0,0,0)) -> Image.Image:
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

# ---------------------------
# Helpers（OCR→計算）
# ---------------------------

def _zen2han_num_colon(s: str) -> str:
    z2h = {ord('０') + i: ord('0') + i for i in range(10)}
    z2h.update({ord('：'): ord(':'), ord('［'): ord('['), ord('］'): ord(']')})
    return s.translate(z2h)

def _parse_hms_to_sec(s: str) -> int:
    """'HH:MM:SS' もしくは 'MM:SS' を秒に"""
    s = s.strip()
    parts = s.split(':')
    if len(parts) == 3:
        h, m, sec = map(int, parts)
        return h*3600 + m*60 + sec
    if len(parts) == 2:
        m, sec = map(int, parts)
        return m*60 + sec
    raise ValueError(f"time format not supported: {s}")

def _sec_to_hms(sec: int) -> str:
    sec %= 24*3600
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def parse_and_compute(oai_text: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    OCR文字列から:
      - サーバー番号
      - 基準時間
      - 停戦終了時間
      - 各「越域駐◯場 N」「免戦中 T」が並ぶブロックを走査
      - 基準時間 + 免戦T を計算
      - 先頭ブロックの計算結果と停戦終了が不一致なら差分(秒)を全結果に加算して補正
    戻り値: (出力テキスト, 基準時間, 停戦終了時間)
    """
    text = _zen2han_num_colon(oai_text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # サーバー番号
    m = re.search(r'\[?s(\d{2,5})\]?', text, flags=re.IGNORECASE)
    server = m.group(1) if m else "?"

    # 停戦終了（行ベースで取りやすい）
    cease: Optional[str] = None
    for ln in lines:
        if "停戦" in ln:
            m2 = re.search(r'(\d{1,2}:\d{2}:\d{2})', ln)
            if m2:
                cease = m2.group(1)
                break

    # 基準時間（免戦行でも停戦行でもない行にある HH:MM:SS）
    base: Optional[str] = None
    for ln in lines:
        if ("免戦" in ln) or ("停戦" in ln):
            continue
        m3 = re.search(r'(\d{1,2}:\d{2}:\d{2})', ln)
        if m3:
            base = m3.group(1)
            break

    # ブロック抽出（タイトル行→直後の免戦行）
    blocks: List[Tuple[int, int]] = []  # (駐◯場番号, 免戦秒)
    i = 0
    while i < len(lines):
        ln = lines[i]
        if RE_TITLE.search(ln):
            mnum = re.search(r'(\d{1,3})', ln)
            if mnum:
                num = int(mnum.group(1))
                # 次の行以降で免戦を探す
                j = i + 1
                dur_sec = None
                while j < len(lines):
                    ln2 = lines[j]
                    if RE_TITLE.search(ln2):
                        break
                    if "免戦" in ln2:
                        mtime = re.search(r'([0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?)', ln2)
                        if mtime:
                            dur_sec = _parse_hms_to_sec(mtime.group(1))
                            break
                    j += 1
                if dur_sec is not None:
                    blocks.append((num, dur_sec))
                i = j
                continue
        i += 1

    # 計算
    out_lines: List[str] = []
    if base and blocks:
        base_sec = _parse_hms_to_sec(base)
        result_secs = [base_sec + dur for (_, dur) in blocks]

        # 補正
        if cease:
            cease_sec = _parse_hms_to_sec(cease)
            diff = cease_sec - result_secs[0]
            if diff != 0:
                result_secs = [sec + diff for sec in result_secs]

        for (num, _dur), sec in zip(blocks, result_secs):
            out_lines.append(f"{server}-{num}-{_sec_to_hms(sec)}")

    result_text = "\n".join(out_lines) if out_lines else "(計算できませんでした)"
    return result_text, base, cease

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
# 画像パイプライン
# ---------------------------

def process_image_pipeline(pil_im: Image.Image) -> Tuple[Image.Image, str, bytes, str]:
    """
    指定のスライス→トリム→(7番ブロックを削除詰め)→合成→OpenAI OCR→時刻計算
    戻り値: (最終合成画像, OpenAI OCRテキスト, OpenAIへ送った画像bytes, 計算結果テキスト)
    """
    parts = slice_exact_7(pil_im, CUTS)

    kept: Dict[int, Image.Image] = {}
    for idx in KEEP:
        block = parts[idx - 1]
        l_pct, r_pct = TRIM_RULES[idx]
        kept[idx] = trim_lr_percent(block, l_pct, r_pct)

    # 7番は「黒塗り部分を削除して詰める」
    kept[7] = compress_7_block(kept[7])

    # 6の右隣に 2（時計）を並べる
    top_row = hstack(kept[6], kept[2], gap=8)

    # 縦に 4（停戦）→ 7（一覧）
    final_img = vstack([top_row, kept[4], kept[7]], gap=10)

    # OpenAI OCR → 計算
    oai_text, sent_png = openai_ocr_png(final_img)
    calc_text, base, cease = parse_and_compute(oai_text)

    # 付加情報（任意）
    header = []
    if base:
        header.append(f"基準: {base}")
    if cease:
        header.append(f"停戦: {cease}")
    header_line = " | ".join(header)
    calc_out = (header_line + "\n" if header_line else "") + calc_text

    return final_img, oai_text, sent_png, calc_out

# ---------------------------
# Discord command
# ---------------------------

@bot.command(name="oaiocr", help="画像を添付して実行。処理→詰め→OpenAI OCR→時間計算まで行います。")
async def oaiocr(ctx: commands.Context):
    try:
        if not ctx.message.attachments:
            await ctx.reply("画像を添付して `!oaiocr` を実行してください。")
            return

        # まず「解析中…」を即時返信
        progress_msg = await ctx.reply("解析中…⏳")

        att: Optional[discord.Attachment] = None
        for a in ctx.message.attachments:
            if a.content_type and a.content_type.startswith("image/"):
                att = a
                break
        if att is None:
            await progress_msg.edit(content="画像の添付が見つかりませんでした。")
            return

        data = await att.read()
        pil = load_image_from_bytes(data)

        loop = asyncio.get_event_loop()
        final_img, oai_text, sent_png, calc_out = await loop.run_in_executor(
            None, process_image_pipeline, pil
        )

        out_buf = io.BytesIO()
        final_img.convert("RGB").save(out_buf, format="PNG")
        out_buf.seek(0)

        sent_buf = io.BytesIO(sent_png); sent_buf.seek(0)

        files = [
            discord.File(out_buf, filename="result.png"),
            discord.File(sent_buf, filename="sent_to_openai.png"),
        ]

        await progress_msg.edit(
            content=f"OpenAI OCR 結果:\n```\n{oai_text}\n```\n計算結果:\n```\n{calc_out}\n```",
            attachments=files
        )

    except Exception as e:
        try:
            await ctx.reply(f"エラー: {e}")
        except Exception:
            pass

@bot.command(name="ping")
async def ping(ctx: commands.Context):
    await ctx.reply("pong 🏓")

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)