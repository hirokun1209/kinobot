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
    7: (20.0, 50.0),   # 駐騎ナンバー＋免戦時間（左20% / 右50% を切り落として残す）
    6: (32.48, 51.50), # サーバー番号
    4: (44.0, 20.19),  # 停戦終了
    2: (75.98, 10.73), # 時計
}

# 行検出に使う正規表現（スペースや全角半角ゆらぎを吸収）
RE_YUEYI_LINE = re.compile(r"越\s*域\s*駐\s*[騎機]\s*場", re.I)
RE_IMMUNE_LINE = re.compile(r"免\s*戦\s*中", re.I)

# 文字にギリ触れないようにほんの少しだけ余白
TOP_PAD = 0       # 黒塗りの開始側（下方向へ何px下げるか）
BOTTOM_PAD = 0    # 黒塗りの終了側（上方向へ何px上げるか）

# ---------------------------
# Helpers
# ---------------------------

def load_image_from_bytes(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    im = ImageOps.exif_transpose(im)  # EXIFの向きを補正
    return im.convert("RGBA")

def slice_exact_7(im: Image.Image, cuts_pct: List[float]) -> List[Image.Image]:
    """境界％に従って7ブロックへ分割（1..7を返す）"""
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

# ---------- OCR (word -> line) ----------

def _google_ocr_words(pil_im: Image.Image):
    """Google Vision で word 単位の [(text, (x1,y1,x2,y2))] を返す"""
    buf = io.BytesIO()
    pil_im.convert("RGB").save(buf, format="JPEG", quality=95)
    image = vision.Image(content=buf.getvalue())
    resp = gcv_client.document_text_detection(image=image)
    if resp.error.message:
        raise RuntimeError(resp.error.message)

    words = []
    for page in resp.full_text_annotation.pages:
        for block in page.blocks:
            for para in block.paragraphs:
                for word in para.words:
                    txt = "".join(s.text for s in word.symbols)
                    xs = [v.x for v in word.bounding_box.vertices]
                    ys = [v.y for v in word.bounding_box.vertices]
                    x1, x2 = min(xs), max(xs)
                    y1, y2 = min(ys), max(ys)
                    words.append((txt, (x1, y1, x2, y2)))
    return words

def _cluster_lines_by_y(words: List[Tuple[str, Tuple[int,int,int,int]]], height: int):
    """
    y中心で近い単語を同一行としてクラスタリング。
    返り値: list[ {text, top, bottom, items=[(t,box),...]} ] を y順で。
    """
    if not words:
        return []
    # y中心でソート
    items = [ (t, box, (box[1]+box[3])//2) for t, box in words ]
    items.sort(key=lambda x: x[2])

    # 行しきい値（画像高さの約1.8% or 12px の大きい方）
    thr = max(12, int(round(height * 0.018)))

    lines = []
    cur = [items[0]]
    for t, box, cy in items[1:]:
        prev_cy = cur[-1][2]
        if abs(cy - prev_cy) <= thr:
            cur.append((t, box, cy))
        else:
            lines.append(cur)
            cur = [(t, box, cy)]
    lines.append(cur)

    out = []
    for line in lines:
        top = min(b[1] for _, b, _ in line)
        bot = max(b[3] for _, b, _ in line)
        # 行テキストはスペース無しと有りの両方作る
        joined_no_space = "".join(t for t, _, _ in line)
        joined_with_space = " ".join(t for t, _, _ in line)
        out.append({
            "top": top, "bottom": bot,
            "text_raw": joined_with_space,
            "text_compact": joined_no_space,
            "items": [(t,b) for t,b,_ in line],
        })
    # yで整列
    out.sort(key=lambda d: (d["top"], d["bottom"]))
    return out

# ---------- 7番ブロックの黒塗り（新ルール） ----------

def draw_black_bars_for_7(pil_im: Image.Image) -> Image.Image:
    """
    7番ブロック：
      ・各「越域駐騎(機)場◯」行の下端から、次の「越域駐騎(機)場◯」行の上端まで黒塗り
      ・途中に「免戦中…」行がある場合は、その行の下端から次の「越域駐騎(機)場◯」行の上端まで黒塗り
      ・最後は一番下まで
    """
    im = pil_im.copy()
    draw = ImageDraw.Draw(im)
    w, h = im.size

    words = _google_ocr_words(im)
    lines = _cluster_lines_by_y(words, h)

    # 対象行を抽出
    title_lines = []   # 「越域駐騎場」行
    immune_lines = []  # 「免戦中」行
    for ln in lines:
        t = ln["text_compact"]
        if RE_YUEYI_LINE.search(t):
            title_lines.append(ln)
        if RE_IMMUNE_LINE.search(t):
            immune_lines.append(ln)

    # タイトル行が無ければ何もしない
    if not title_lines:
        return im

    # 各タイトル行ごとに黒塗り区間を決定
    for i, cur_title in enumerate(title_lines):
        cur_bottom = cur_title["bottom"]
        next_top = title_lines[i+1]["top"] if i+1 < len(title_lines) else h

        # この区間に含まれる「免戦中」行を（タイトル直下に最も近いもの）探す
        chosen_start = cur_bottom
        for imm in immune_lines:
            if cur_bottom <= imm["top"] <= next_top:
                chosen_start = max(chosen_start, imm["bottom"])
                break  # 一番近いのが見つかったら採用

        top_y = min(next_top, chosen_start + TOP_PAD)
        bot_y = max(top_y, next_top - BOTTOM_PAD)

        if bot_y > top_y:
            # 横は全幅を塗りつぶし
            draw.rectangle([(0, top_y), (w, bot_y)], fill=(0, 0, 0, 255))

    return im

# ---------- 合成系 ----------

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

# ---------- OpenAI OCR ----------

def openai_ocr_png(pil_im: Image.Image) -> Tuple[str, bytes]:
    """OpenAI へ画像OCR依頼。返り値: (テキスト, 送ったPNGバイト列)"""
    buf = io.BytesIO()
    pil_im.convert("RGB").save(buf, format="PNG")
    png_bytes = buf.getvalue()

    b64 = base64.b64encode(png_bytes).decode("utf-8")
    resp = oai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role":"user",
            "content":[
                {"type":"text", "text":"以下の画像内の日本語テキストを改行・数字を保持してそのまま読み取ってください。"},
                {"type":"image_url", "image_url":{"url": f"data:image/png;base64,{b64}"}}
            ]
        }],
        temperature=0.0,
    )
    text = resp.choices[0].message.content.strip()
    return text, png_bytes

# ---------- パイプライン ----------

def process_image_pipeline(pil_im: Image.Image) -> Tuple[Image.Image, str, bytes]:
    """
    指定のスライス→トリム→黒塗り→合成→OpenAI OCR
    戻り値: (最終合成画像, OpenAI OCRテキスト, OpenAIへ送った画像bytes)
    """
    parts = slice_exact_7(pil_im, CUTS)

    kept: Dict[int, Image.Image] = {}
    for idx in KEEP:
        block = parts[idx-1]
        l_pct, r_pct = TRIM_RULES[idx]
        kept[idx] = trim_lr_percent(block, l_pct, r_pct)

    # 7番に新ルールで黒塗り
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

        sent_buf = io.BytesIO(sent_png); sent_buf.seek(0)

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