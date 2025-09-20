import os
import io
import re
import base64
import asyncio
import unicodedata
from typing import List, Tuple, Dict

import discord
from discord.ext import commands
from PIL import Image, ImageDraw, ImageOps

# Google Vision
from google.cloud import vision

# OpenAI (official SDK v1)
from openai import OpenAI

# ---- PIL resample (互換) ----
try:
    RESAMPLE = Image.Resampling.LANCZOS
except Exception:
    RESAMPLE = Image.LANCZOS

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

# 正規表現（緩め）
RE_IMMUNE = re.compile(r"免\s*戦\s*中")                                 # 「免戦中」
RE_TITLE  = re.compile(r"越\s*域\s*駐[\u4E00-\u9FFF]{1,3}\s*場")         # 「越域駐〇場」誤OCRも拾う
RE_TIME   = re.compile(r"\d{1,2}[:：]\d{2}(?:[:：]\d{2})?")              # 05:53 / 01:02:13 など

# フォールバック：ブロック高さに対する“必ず残す”上部割合
FALLBACK_KEEP_TOP_RATIO = 0.35

# ---------------------------
# Helpers
# ---------------------------

def load_image_from_bytes(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    im = ImageOps.exif_transpose(im)  # EXIFの向きを補正
    return im.convert("RGBA")

def resize_to_width(img: Image.Image, target_w: int = 708) -> Image.Image:
    """横幅を target_w に、縦は比率維持でリサイズ"""
    w, h = img.size
    if w == target_w:
        return img
    new_h = max(1, int(round(h * (target_w / float(w)))))
    return img.resize((target_w, new_h), RESAMPLE)

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
    """
    wordをY座標で行グループ化して (line_text, (x1,y1,x2,y2)) を返す。
    y_tol は同一行とみなす上下許容ピクセル。
    """
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
        text = "".join(c[4] for c in chunks)  # スペース無しで連結（日本語はこれでOK）
        x1 = min(c[0] for c in chunks)
        y1 = min(c[1] for c in chunks)
        x2 = max(c[2] for c in chunks)
        y2 = max(c[3] for c in chunks)
        line_boxes.append((text, (x1, y1, x2, y2)))
    return line_boxes

def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s).replace(" ", "")

def compact_7_by_removing_sections(pil_im: Image.Image) -> Image.Image:
    """
    7番ブロック内で、
      ・各「越域駐〇場」行をタイトルとみなす
      ・タイトルiの下端〜タイトルi+1の上端を“ワンブロック”
      ・ブロック内に「免戦中」または時間っぽい行(05:53/01:02:13等)があれば、
         その行の下端までを“残す”／それ以降〜次タイトル直上を削除（詰める）
      ・見つからなければブロック上部の一定割合(FALLBACK_KEEP_TOP_RATIO)は必ず残す
      ・最後のブロックは画像下端までを対象
    """
    im = pil_im.copy()
    w, h = im.size

    lines = google_ocr_line_boxes(im, y_tol=18)

    # タイトル行・免戦/時間行のY範囲抽出
    titles: List[Tuple[int,int]] = []   # (top, bottom)
    candidates: List[Tuple[int,int]] = []  # (top, bottom) for 免戦中 or 時間

    for text, (x1, y1, x2, y2) in lines:
        t = _norm(text)
        if RE_TITLE.search(t):
            titles.append((y1, y2))
        if RE_IMMUNE.search(t) or RE_TIME.search(t):
            candidates.append((y1, y2))

    titles.sort(key=lambda p: p[0])
    candidates.sort(key=lambda p: p[0])

    if not titles:
        # 何も検出できなければ原図を返す（安全側）
        return im

    # “残す”縦スライスを集める
    keep_slices: List[Tuple[int,int]] = []
    for i, (t_y1, t_y2) in enumerate(titles):
        start = t_y1
        end = titles[i + 1][0] if i + 1 < len(titles) else h

        if end <= start:
            continue

        # このブロック内で最後に出る候補のbottomを採用
        cand_bottom = None
        for cy1, cy2 in candidates:
            if start <= cy1 < end:
                cand_bottom = cy2

        if cand_bottom is not None:
            cut_at = cand_bottom  # 時間行の下端まで 残す
        else:
            # フォールバック：ブロック上部一定割合は残す（タイトルは必ず含める）
            cut_at = min(end, int(round(start + (end - start) * FALLBACK_KEEP_TOP_RATIO)))
            cut_at = max(cut_at, t_y2)

        if cut_at > start:
            keep_slices.append((start, cut_at))

    # スライスを縦に詰め直す
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
    """左右に結合（高さは大きい方に合わせ中央寄せ）"""
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

def process_image_pipeline(pil_im: Image.Image) -> Tuple[Image.Image, str, bytes]:
    """
    指定のスライス→トリム→（7を詰め処理）→合成→OpenAI OCR
    戻り値: (最終合成画像, OpenAI OCRテキスト, OpenAIへ送った画像bytes)
    """
    # ★ 受信直後に横幅708へ統一（縦は比率維持）
    pil_im = resize_to_width(pil_im, target_w=708)

    parts = slice_exact_7(pil_im, CUTS)

    # 1..7 のうち 2/4/6/7 を残す
    kept: Dict[int, Image.Image] = {}
    for idx in KEEP:
        block = parts[idx - 1]
        l_pct, r_pct = TRIM_RULES[idx]
        kept[idx] = trim_lr_percent(block, l_pct, r_pct)

    # 7番：黒塗りではなく“削除して詰める”
    kept[7] = compact_7_by_removing_sections(kept[7])

    # 6の右隣に 2 を横並び（時計はサーバー番号の隣）
    top_row = hstack(kept[6], kept[2], gap=8)

    # 縦に 4、7 を下へ
    final_img = vstack([top_row, kept[4], kept[7]], gap=10)

    # OpenAI OCR
    oai_text, sent_png = openai_ocr_png(final_img)
    return final_img, oai_text, sent_png

# ---------------------------
# Discord command
# ---------------------------

@bot.command(name="oaiocr", help="画像を添付して実行。処理→詰め→OpenAI OCR まで行います。")
async def oaiocr(ctx: commands.Context):
    try:
        if not ctx.message.attachments:
            await ctx.reply("画像を添付して `!oaiocr` を実行してください。")
            return

        att: discord.Attachment = None
        for a in ctx.message.attachments:
            if a.content_type and a.content_type.startswith("image/"):
                att = a
                break
        if att is None:
            await ctx.reply("画像の添付が見つかりませんでした。")
            return

        # まずは即レス
        await ctx.reply("解析中…")

        data = await att.read()
        pil = load_image_from_bytes(data)

        loop = asyncio.get_event_loop()
        final_img, oai_text, sent_png = await loop.run_in_executor(None, process_image_pipeline, pil)

        out_buf = io.BytesIO()
        final_img.convert("RGB").save(out_buf, format="PNG")
        out_buf.seek(0)

        sent_buf = io.BytesIO(sent_png)
        sent_buf.seek(0)

        files = [
            discord.File(out_buf, filename="result.png"),
            discord.File(sent_buf, filename="sent_to_openai.png"),
        ]
        await ctx.reply(content=f"OpenAI OCR 結果:\n```\n{oai_text}\n```", files=files)

    except Exception as e:
        await ctx.reply(f"エラー: {e}")

@bot.command(name="ping")
async def ping(ctx: commands.Context):
    await ctx.reply("pong 🏓")

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)