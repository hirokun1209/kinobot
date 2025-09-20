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
CUTS = [7.51, 11.63, 25.21, 29.85, 33.62, 38.41, 61.90]  # 上からの境界％（合計7境界→7ブロック）

# 残すブロック番号（1始まり）
KEEP = [2, 4, 6, 7]

# 横方向トリミング（左/右を％でカット）可視用
TRIM_RULES = {
    7: (20.0, 50.0),   # 駐騎ナンバー＋免戦時間（表示用）
    6: (32.48, 51.50), # サーバー番号
    4: (44.0, 20.19),  # 停戦終了
    2: (75.98, 10.73), # 時計
}

# 7番ブロックのOCR用（見出しを確実に拾うため少し広め）
TRIM_RULES_7_OCR = (8.0, 35.0)

# 黒塗り判定（7番ブロック用）
RE_IMMUNE = re.compile(r"免\s*戦\s*中")  # 「免戦中」
# 見出しは「越域駐騎場 / 越域駐機場」など。ゆるく検出（越 / 越域 / 駐騎場 / 駐機場）
RE_HEADER_STRICT = re.compile(r"(越\s*域\s*駐\s*[騎機]\s*場)")
RE_HEADER_LAX = re.compile(r"(越\s*域|駐\s*[騎機]\s*場)")

TOP_MARGIN_AFTER_IMMUNE = 6     # 免戦中の直下 +6px
BOTTOM_MARGIN_BEFORE_NEXT = 2   # 次の見出し直上 -2px
HEADER_HEIGHT_GUESS = 24        # 見出しが見つからない時の保守的な控除量

# ---------------------------
# Helpers
# ---------------------------

def load_image_from_bytes(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    im = ImageOps.exif_transpose(im)  # EXIF補正
    return im.convert("RGBA")

def slice_exact_7(im: Image.Image, cuts_pct: List[float]) -> List[Image.Image]:
    """CUTSに基づいて画像を縦7分割（1..7）"""
    w, h = im.size
    y = [0] + [int(round(h * p / 100.0)) for p in cuts_pct] + [h]
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

def google_ocr_word_boxes(pil_im: Image.Image) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """Google Visionで word 単位の文字とバウンディングボックスを返す (text, (x1,y1,x2,y2))"""
    buf = io.BytesIO()
    pil_im.convert("RGB").save(buf, format="JPEG", quality=95)
    image = vision.Image(content=buf.getvalue())

    resp = gcv_client.document_text_detection(image=image)
    if resp.error.message:
        raise RuntimeError(resp.error.message)

    words: List[Tuple[str, Tuple[int, int, int, int]]] = []
    if not resp.full_text_annotation.pages:
        return words

    for page in resp.full_text_annotation.pages:
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

def _collect_y_positions_for_7(ocr_src: Image.Image) -> Tuple[List[Tuple[int,int]], List[int], int]:
    """
    7番ブロックOCRソースから
      - 免戦中の (y1,y2) リスト
      - 見出し候補（越域/駐◯場など）の y(top) リスト
      - 免戦中の推定行高（フォールバックで使用）
    を返す
    """
    words = google_ocr_word_boxes(ocr_src)

    immune: List[Tuple[int, int]] = []
    headers: List[int] = []

    immune_heights: List[int] = []

    for raw, (x1, y1, x2, y2) in words:
        t = raw.replace(" ", "")
        if RE_IMMUNE.search(t):
            immune.append((y1, y2))
            immune_heights.append(max(1, y2 - y1))
        # 見出しは厳密→ダメなら緩め、の両方を拾う
        if RE_HEADER_STRICT.search(t) or RE_HEADER_LAX.search(t):
            headers.append(y1)

    immune.sort()
    headers.sort()
    # 免戦中の行高の代表値（中央値）
    if immune_heights:
        immune_heights.sort()
        mid = immune_heights[len(immune_heights)//2]
    else:
        mid = HEADER_HEIGHT_GUESS
    return immune, headers, mid

def draw_black_bars_for_7(visible_im: Image.Image, ocr_src: Optional[Image.Image] = None) -> Image.Image:
    """
    7番ブロック画像内の
      免戦中 の bottom+6px 〜 次の 見出し(越域/駐◯場) の top-2px
    を横幅いっぱい黒塗り。
    見出しが検出できない場合は、次の「免戦中」行のかなり上（推定行高分）で止める。
    """
    im = visible_im.copy()
    draw = ImageDraw.Draw(im)
    w, h = im.size

    src = ocr_src if ocr_src is not None else im
    immune_boxes, header_tops, immune_line_h = _collect_y_positions_for_7(src)

    for idx, (iy1, iy2) in enumerate(immune_boxes):
        top = min(h, max(0, iy2 + TOP_MARGIN_AFTER_IMMUNE))

        # 1) 次の見出し（このtopより下）
        next_header_top = None
        for hy in header_tops:
            if hy > top:
                next_header_top = hy
                break

        # 2) 見出しが無い場合は、次の免戦中の上端から行高ぶん引いて見出し相当位置を近似
        bottom: int
        if next_header_top is not None:
            bottom = max(top, next_header_top - BOTTOM_MARGIN_BEFORE_NEXT)
        else:
            next_immune_top = immune_boxes[idx + 1][0] if (idx + 1) < len(immune_boxes) else None
            if next_immune_top is not None:
                approx_header_top = max(0, next_immune_top - immune_line_h)
                bottom = max(top, approx_header_top - BOTTOM_MARGIN_BEFORE_NEXT)
            else:
                bottom = h  # 最後まで

        if bottom > top:
            draw.rectangle([(0, top), (w, bottom)], fill=(0, 0, 0, 255))

    return im

def hstack(im_left: Image.Image, im_right: Image.Image, gap: int = 8, bg=(0, 0, 0, 0)) -> Image.Image:
    """左右に結合（高さは大きい方に合わせ中央寄せ）"""
    h = max(im_left.height, im_right.height)
    w = im_left.width + gap + im_right.width
    canvas = Image.new("RGBA", (w, h), bg)
    y1 = (h - im_left.height) // 2
    y2 = (h - im_right.height) // 2
    canvas.paste(im_left, (0, y1))
    canvas.paste(im_right, (im_left.width + gap, y2))
    return canvas

def vstack(images: List[Image.Image], gap: int = 8, bg=(0, 0, 0, 0)) -> Image.Image:
    """縦結合（幅は最大に合わせ中央寄せ）"""
    if not images:
        raise ValueError("no images to stack")
    max_w = max(img.width for img in images)
    total_h = sum(img.height for img in images) + gap * (len(images) - 1)
    canvas = Image.new("RGBA", (max_w, total_h), bg)
    y = 0
    for img in images:
        x = (max_w - img.width) // 2
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
    指定のスライス→トリム→黒塗り→合成→OpenAI OCR
    戻り値: (最終合成画像, OpenAI OCRテキスト, OpenAIへ送った画像bytes)
    """
    parts = slice_exact_7(pil_im, CUTS)

    kept: Dict[int, Image.Image] = {}

    # 2,4,6 は通常どおりトリム
    for idx in [2, 4, 6]:
        block = parts[idx - 1]
        l_pct, r_pct = TRIM_RULES[idx]
        kept[idx] = trim_lr_percent(block, l_pct, r_pct)

    # 7は表示用トリムとOCR用トリムを分ける
    block7 = parts[7 - 1]
    l_vis, r_vis = TRIM_RULES[7]
    l_ocr, r_ocr = TRIM_RULES_7_OCR
    trimmed_7_visible = trim_lr_percent(block7, l_vis, r_vis)
    trimmed_7_ocr = trim_lr_percent(block7, l_ocr, r_ocr)  # ←見出し検出を安定させるため広め

    kept[7] = draw_black_bars_for_7(trimmed_7_visible, ocr_src=trimmed_7_ocr)

    # 6の右隣に 2 を横並び（「時計はサーバー番号の隣」）
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

        att: Optional[discord.Attachment] = None
        for a in ctx.message.attachments:
            if a.content_type and a.content_type.startswith("image/"):
                att = a
                break
        if att is None:
            await ctx.reply("画像の添付が見つかりませんでした。")
            return

        data = await att.read()
        pil = load_image_from_bytes(data)

        # パイプライン処理（スレッドプールでブロッキング回避）
        loop = asyncio.get_running_loop()
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