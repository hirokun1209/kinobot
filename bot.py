import os
import io
import re
import json
import base64
import asyncio
from typing import List, Tuple, Dict

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
print(f"✅ google-cloud-vision {vision.__version__} ready")

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

# ---- 黒塗り判定用 正規表現（強化版） ----
RE_IMMUNE = re.compile(r"免\s*戦\s*中")  # 「免戦中」
# 「越域駐騎場 / 越域駐機場」ゆるめ（OCRの取り違えや空白に強く）
RE_YUEYI = re.compile(r"(越\s*域\s*駐\s*[騎机]\s*場|駐\s*[騎机]\s*場)")
# 時刻（全角コロン対応）
RE_TIME = re.compile(r"[0-2]?\d[:：][0-5]\d[:：][0-5]\d")

TOP_MARGIN_AFTER_IMMUNE = 6   # 免戦行の直下 +6px
BOTTOM_MARGIN_BEFORE_NEXT = 2 # 次のタイトル行直上 -2px

# ---------------------------
# Helpers
# ---------------------------

def load_image_from_bytes(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    im = ImageOps.exif_transpose(im)
    return im.convert("RGBA")

def slice_exact_7(im: Image.Image, cuts_pct: List[float]) -> List[Image.Image]:
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
    left = max(0, min(left, w-1))
    right = max(left+1, min(right, w))
    return im.crop((left, 0, right, h))

def google_ocr_word_boxes(pil_im: Image.Image) -> List[Tuple[str, Tuple[int,int,int,int]]]:
    """Google Vision で word 単位の文字とバウンディングボックスを返す (text, (x1,y1,x2,y2))"""
    buf = io.BytesIO()
    pil_im.convert("RGB").save(buf, format="JPEG", quality=95)
    content = buf.getvalue()
    image = vision.Image(content=content)

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

# === ここから：行クラスタリング & 黒塗り ===

def _group_words_into_lines(words: List[Tuple[str, Tuple[int,int,int,int]]], img_h: int
                            ) -> List[Tuple[str, Tuple[int,int,int,int]]]:
    """
    単語を“行”にまとめて返す。
    返り値: [(行テキスト(空白除去), (x1,y1,x2,y2))...] を上から順に。
    """
    # yセンターが近いものを同一行としてまとめる
    thr = max(12, int(img_h * 0.012))  # 画像高さに応じてしきい値
    items = []
    for txt, (x1, y1, x2, y2) in words:
        cy = (y1 + y2) // 2
        items.append((cy, x1, txt, (x1, y1, x2, y2)))
    items.sort()  # cy, x1 でソート

    lines = []
    for cy, x1, txt, box in items:
        placed = False
        for ln in lines:
            if abs(cy - ln["cy"]) <= thr:
                ln["parts"].append((x1, txt, box))
                ln["top"] = min(ln["top"], box[1])
                ln["bot"] = max(ln["bot"], box[3])
                ln["cy"] = (ln["top"] + ln["bot"]) // 2
                placed = True
                break
        if not placed:
            lines.append({"parts":[(x1, txt, box)], "top":box[1], "bot":box[3], "cy":cy})

    out = []
    for ln in lines:
        ln["parts"].sort(key=lambda t: t[0])  # xでソート
        text = "".join(p[1] for p in ln["parts"]).replace(" ", "")
        x1 = ln["parts"][0][2][0]
        x2 = ln["parts"][-1][2][2]
        out.append((text, (x1, ln["top"], x2, ln["bot"])))
    out.sort(key=lambda t: t[1][1])  # y1で
    return out

def draw_black_bars_for_7(pil_im: Image.Image) -> Image.Image:
    """
    7番ブロック画像内で、
    免戦行（=「免戦中」または HH:MM:SS を含む行）の bottom+6px 〜
    次の タイトル行（=「越域駐騎/駐機場」）の top-2px を黒塗り。
    タイトル行が見当たらない場合は、次の免戦行の top-2px。
    それも無い場合は画像の一番下まで。
    """
    im = pil_im.copy()
    draw = ImageDraw.Draw(im)
    w, h = im.size

    words = google_ocr_word_boxes(im)
    lines = _group_words_into_lines(words, h)

    immune_lines: List[Tuple[int,int]] = []
    title_lines:  List[Tuple[int,int]] = []

    for text, (x1, y1, x2, y2) in lines:
        t = text  # 既に空白除去済み
        is_immune = RE_IMMUNE.search(t) is not None or RE_TIME.search(t) is not None
        is_title  = RE_YUEYI.search(t) is not None
        if is_immune:
            immune_lines.append((y1, y2))
        if is_title:
            title_lines.append((y1, y2))

    immune_lines.sort()
    title_lines.sort()

    for idx, (iy1, iy2) in enumerate(immune_lines):
        top = iy2 + TOP_MARGIN_AFTER_IMMUNE

        # 次のタイトル行
        next_title_top = next((ty1 for (ty1, ty2) in title_lines if ty1 > top), None)
        # 代替：次の免戦行
        next_immune_top = next((jy1 for (jy1, jy2) in immune_lines[idx+1:] if jy1 > top), None)

        if next_title_top is not None:
            bottom = max(top, next_title_top - BOTTOM_MARGIN_BEFORE_NEXT)
        elif next_immune_top is not None:
            bottom = max(top, next_immune_top - BOTTOM_MARGIN_BEFORE_NEXT)
        else:
            bottom = h

        if bottom > top:
            draw.rectangle([(0, top), (w, bottom)], fill=(0, 0, 0, 255))

    return im

# ---- ここまで黒塗り ----

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

def openai_ocr_png(pil_im: Image.Image) -> Tuple[str, bytes]:
    buf = io.BytesIO()
    pil_im.convert("RGB").save(buf, format="PNG")
    png_bytes = buf.getvalue()

    b64 = base64.b64encode(png_bytes).decode("utf-8")
    resp = oai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role":"user",
            "content":[
                {"type":"text", "text":"以下の画像に写っている日本語テキストを、そのまま読み取ってください（改行と数字も保持）。"},
                {"type":"image_url", "image_url":{"url": f"data:image/png;base64,{b64}"}}
            ]
        }],
        temperature=0.0,
    )
    text = resp.choices[0].message.content.strip()
    return text, png_bytes

def process_image_pipeline(pil_im: Image.Image) -> Tuple[Image.Image, str, bytes]:
    parts = slice_exact_7(pil_im, CUTS)

    kept: Dict[int, Image.Image] = {}
    for idx in KEEP:
        block = parts[idx-1]
        l_pct, r_pct = TRIM_RULES[idx]
        kept[idx] = trim_lr_percent(block, l_pct, r_pct)

    # 7番に黒塗り（改良版）
    kept[7] = draw_black_bars_for_7(kept[7])

    # 6の右隣に 2 を横並び（時計はサーバー番号の隣）
    top_row = hstack(kept[6], kept[2], gap=8)

    # 縦に並べて 6+2 の下に 4、7
    final_img = vstack([top_row, kept[4], kept[7]], gap=10)

    # OpenAI OCR
    oai_text, sent_png = openai_ocr_png(final_img)
    return final_img, oai_text, sent_png

# ---------------------------
# Discord command
# ---------------------------
@bot.command(name="oaiocr", help="画像を添付して実行。処理→黒塗り→OpenAI OCR まで行います。")
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

# ---------------------------
# run
# ---------------------------
if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)