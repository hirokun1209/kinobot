import os
import io
import re
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
CUTS = [7.51, 11.63, 25.21, 29.85, 33.62, 38.41, 61.90]  # 上からの境界％（1..7の7ブロック）

# 残すブロック番号（1始まり）
KEEP = [2, 4, 6, 7]

# 横方向トリミング（左は“左端からの残す開始％”、右は“右側の切り落とし率％”）
TRIM_RULES = {
    7: (20.0, 50.0),   # 駐騎ナンバー＋免戦時間  → 左20% / 右側50%をカット ⇒ 右端=50%
    6: (32.48, 51.50), # サーバー番号            → 左32.48% / 右51.5%をカット ⇒ 右端=48.5%
    4: (44.0, 20.19),  # 停戦終了                 → 左44% / 右20.19%をカット ⇒ 右端=79.81%
    2: (75.98, 10.73), # 時計                     → 左75.98% / 右10.73%をカット ⇒ 右端=89.27%
}

# 黒塗りアンカー検出（7番ブロック用）
RE_TIME   = re.compile(r"\b\d{1,2}:\d{2}:\d{2}\b")  # 01:02:13 など
RE_IMMUNE = re.compile(r"免\s*戦\s*中")               # 免戦中（フォールバック）
RE_YUEYI  = re.compile(r"越\s*域")                   # 「越域」をゆるく検出

TOP_MARGIN_AFTER_ANCHOR   = 6   # 上端のアンカー（時刻 or 免戦中）の直下 +6px
BOTTOM_MARGIN_BEFORE_NEXT = 2   # 次アンカー直上 -2px

# ---------------------------
# Helpers
# ---------------------------

def load_image_from_bytes(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    im = ImageOps.exif_transpose(im)  # EXIFの向きを補正
    return im.convert("RGBA")

def slice_exact_7(im: Image.Image, cuts_pct: List[float]) -> List[Image.Image]:
    """境界％から“高さ方向に7ブロック”へ分割し、1..7を返す"""
    w, h = im.size
    boundaries = [int(round(h * p / 100.0)) for p in cuts_pct]
    y = [0] + boundaries + [h]
    parts = []
    for i in range(7):
        parts.append(im.crop((0, y[i], w, y[i+1])))
    return parts

def trim_lr_percent(im: Image.Image, left_pct: float, right_cut_pct: float) -> Image.Image:
    """左％を起点に、右は“右側を何％切り落とすか”で解釈してトリム"""
    w, h = im.size
    left  = int(round(w * left_pct / 100.0))
    right = int(round(w * (1.0 - right_cut_pct / 100.0)))  # 右端=100%-right_cut
    left  = max(0, min(left, w - 1))
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

    words = []
    if not response.full_text_annotation.pages:
        return words

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for para in block.paragraphs:
                for word in para.words:
                    txt = "".join(s.text for s in word.symbols)
                    xs  = [v.x for v in word.bounding_box.vertices]
                    ys  = [v.y for v in word.bounding_box.vertices]
                    x1, x2 = min(xs), max(xs)
                    y1, y2 = min(ys), max(ys)
                    words.append((txt, (x1, y1, x2, y2)))
    return words

def _merge_close_rects(rects: List[Tuple[int,int]], min_gap: int = 4) -> List[Tuple[int,int]]:
    """近接する縦範囲(y1,y2)を結合（OCRの分割ブレ対策）"""
    if not rects:
        return []
    rects = sorted(rects)
    merged = [list(rects[0])]
    for y1, y2 in rects[1:]:
        if y1 - merged[-1][1] <= min_gap:  # ほぼ同じ行
            merged[-1][1] = max(merged[-1][1], y2)
        else:
            merged.append([y1, y2])
    return [(a, b) for a, b in merged]

def draw_black_bars_for_7_fullwidth(pil_im: Image.Image) -> Image.Image:
    """
    7番ブロック“フル幅”画像内の、
      時刻（01:02:13 など）の bottom+6px ～ 次の「越域」 top-2px を黒塗り。
    次の越域が無ければ「次の時刻 top-2px」、それも無ければ画像の最下端まで。
    ※「免戦中」が取れない場合でも“時刻”で確実に塗る。
    """
    im = pil_im.copy()
    draw = ImageDraw.Draw(im)
    w, h = im.size

    words = google_ocr_word_boxes(im)

    time_boxes  = []  # (y1,y2)
    yueyi_boxes = []  # (y1,y2)
    immune_boxes = [] # (y1,y2) フォールバック用

    for txt, (_, y1, _, y2) in words:
        t = txt.replace(" ", "")
        if RE_TIME.search(t):
            time_boxes.append((y1, y2))
        if RE_YUEYI.search(t):
            yueyi_boxes.append((y1, y2))
        if RE_IMMUNE.search(t):
            immune_boxes.append((y1, y2))

    # 近接する同一行のブレをまとめる
    time_boxes   = _merge_close_rects(time_boxes, min_gap=6)
    yueyi_boxes  = _merge_close_rects(yueyi_boxes, min_gap=6)
    immune_boxes = _merge_close_rects(immune_boxes, min_gap=6)

    # 時刻が一つも拾えない場合は「免戦中」の行をアンカーにする
    anchors = time_boxes[:] if time_boxes else [(y2, y2) for (_, y2) in immune_boxes]
    anchors.sort()

    for idx, (ay1, ay2) in enumerate(anchors):
        top = ay2 + TOP_MARGIN_AFTER_ANCHOR

        # 次の「越域」上端
        next_yueyi_top = None
        for y1, _ in yueyi_boxes:
            if y1 > top:
                next_yueyi_top = y1
                break

        # 代替：次のアンカー上端（次の時刻 or 免戦中）
        next_anchor_top = anchors[idx + 1][0] if idx + 1 < len(anchors) else None

        if next_yueyi_top is not None:
            bottom = max(top, next_yueyi_top - BOTTOM_MARGIN_BEFORE_NEXT)
        elif next_anchor_top is not None:
            bottom = max(top, next_anchor_top - BOTTOM_MARGIN_BEFORE_NEXT)
        else:
            bottom = h

        if bottom > top:
            draw.rectangle([(0, top), (w, bottom)], fill=(0, 0, 0, 255))

    return im

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
    縦スライス →（7番だけ先に黒塗り）→ 各ブロックを横トリム → 合成 → OpenAI OCR
    戻り値: (最終合成画像, OpenAI OCRテキスト, OpenAIへ送った画像bytes)
    """
    parts = slice_exact_7(pil_im, CUTS)  # 1..7

    # 7番は“フル幅”のまま先に黒塗りしてから横トリム（越域が切れずに検出できる）
    block7_full = parts[7 - 1]
    block7_black = draw_black_bars_for_7_fullwidth(block7_full)
    l7, r7 = TRIM_RULES[7]
    block7_trimmed = trim_lr_percent(block7_black, l7, r7)

    # 他はそのまま横トリム
    kept: Dict[int, Image.Image] = {7: block7_trimmed}
    for idx in (2, 4, 6):
        block = parts[idx - 1]
        l_pct, r_pct = TRIM_RULES[idx]
        kept[idx] = trim_lr_percent(block, l_pct, r_pct)

    # 6の右隣に 2（時計）を横並び
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

        # Discordへ送信
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

# ping 確認
@bot.command(name="ping")
async def ping(ctx: commands.Context):
    await ctx.reply("pong 🏓")

# ---------------------------
# run
# ---------------------------
if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)