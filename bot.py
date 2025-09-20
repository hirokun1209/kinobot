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
print(f"✅ google-cloud-vision {vision.__version__} ready")  # ← 追加

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

# 黒塗り判定（7番ブロック用）
RE_IMMUNE = re.compile(r"免\s*戦\s*中")  # 「免戦中」
RE_YUEYI = re.compile(r"(越\s*域\s*駐\s*[騎機]\s*場)")  # 「越域駐騎場 / 越域駐機場」ゆるめ

TOP_MARGIN_AFTER_IMMUNE = 6   # 免戦中の直下 +6px
BOTTOM_MARGIN_BEFORE_NEXT = 2 # 次のタイトル直上 -2px

# ---------------------------
# Helpers
# ---------------------------

def load_image_from_bytes(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    # EXIFの向きを補正
    im = ImageOps.exif_transpose(im)
    return im.convert("RGBA")

def slice_by_percent(im: Image.Image, cuts_pct: List[float]) -> List[Image.Image]:
    """高さ％のカット位置で上から順に7分割する（1..7のブロックを返す）"""
    w, h = im.size
    ys = [0] + [int(round(h * p / 100.0)) for p in cuts_pct] + [h]
    parts = []
    for i in range(len(ys)-1):
        parts.append(im.crop((0, ys[i], w, ys[i+1])))
    return parts  # len == 8になるが先頭が薄い帯にならないよう CUTS定義は7つ→8ブロック。今回「1..7」を扱うので最後は無視。

def slice_exact_7(im: Image.Image, cuts_pct: List[float]) -> List[Image.Image]:
    """問題文は7ブロック想定なので、境界から7つに揃える"""
    w, h = im.size
    boundaries = [int(round(h * p / 100.0)) for p in cuts_pct]
    y = [0] + boundaries + [h]
    # 1..7 だけ返す
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

def draw_black_bars_for_7(pil_im: Image.Image) -> Image.Image:
    """
    7番ブロック画像内の
      免戦中 の bottom+6px 〜 次の 越域駐騎(機)場 の top-2px
    を横幅いっぱい黒塗り。
    次が無い場合は「次の免戦中 top-2px」or「画像の一番下」へフォールバック。
    """
    im = pil_im.copy()
    draw = ImageDraw.Draw(im)
    w, h = im.size

    words = google_ocr_word_boxes(im)

    # y位置の候補を抽出
    immune_boxes = []
    yueyi_boxes = []

    for txt, (x1, y1, x2, y2) in words:
        t = txt.replace(" ", "")
        if RE_IMMUNE.search(t):
            immune_boxes.append((y1, y2))
        if RE_YUEYI.search(t):
            yueyi_boxes.append((y1, y2))

    immune_boxes.sort()
    yueyi_boxes.sort()

    # 黒塗り領域決定
    for idx, (iy1, iy2) in enumerate(immune_boxes):
        top = iy2 + TOP_MARGIN_AFTER_IMMUNE

        # 次の「越域駐騎(機)場」上端（この免戦より下の最初）
        next_yueyi_top = None
        for y1, y2 in yueyi_boxes:
            if y1 > top:
                next_yueyi_top = y1
                break

        # 代替：次の「免戦中」上端
        next_immune_top = None
        for j in range(idx+1, len(immune_boxes)):
            next_immune_top = immune_boxes[j][0]
            break

        if next_yueyi_top is not None:
            bottom = max(top, next_yueyi_top - BOTTOM_MARGIN_BEFORE_NEXT)
        elif next_immune_top is not None:
            bottom = max(top, next_immune_top - BOTTOM_MARGIN_BEFORE_NEXT)
        else:
            bottom = h  # 最後まで

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
    # GPT-4o-mini に画像を渡して日本語OCR
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
    """
    指定のスライス→トリム→黒塗り→合成→OpenAI OCR
    戻り値: (最終合成画像, OpenAI OCRテキスト, OpenAIへ送った画像bytes)
    """
    parts = slice_exact_7(pil_im, CUTS)
    # 1..7 のうち 2/4/6/7 を残す
    kept: Dict[int, Image.Image] = {}
    for idx in KEEP:
        block = parts[idx-1]
        l_pct, r_pct = TRIM_RULES[idx]
        kept[idx] = trim_lr_percent(block, l_pct, r_pct)

    # 7番に黒塗り
    kept[7] = draw_black_bars_for_7(kept[7])

    # 6の右隣に 2 を横並び
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
        # 最初の画像系添付を取る
        for a in ctx.message.attachments:
            if a.content_type and a.content_type.startswith("image/"):
                att = a
                break
        if att is None:
            await ctx.reply("画像の添付が見つかりませんでした。")
            return

        data = await att.read()
        pil = load_image_from_bytes(data)

        # パイプライン処理
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