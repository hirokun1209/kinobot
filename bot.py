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
    7: (20.0, 50.0),   # 駐騎ナンバー＋免戦時間
    6: (32.48, 51.50), # サーバー番号
    4: (44.0, 20.19),  # 停戦終了
    2: (75.98, 10.73), # 時計
}

# 正規表現
RE_IMMUNE = re.compile(r"免\s*戦\s*中")                         # 「免戦中」
RE_TITLE  = re.compile(r"越\s*域\s*駐\s*[騎機]\s*場")             # 「越域駐騎場/越域駐機場」
RE_TIME   = re.compile(r"\d{1,2}[:：]\d{2}(?:[:：]\d{2})?")       # 05:53 / 01:02:13 など

# 余白（必要なら調整）
MARGIN_AFTER_INFO = 0               # 免戦中や時間の“下端から”詰め開始
MARGIN_BEFORE_NEXT_TITLE = 0        # 次タイトルの“上端まで”詰める

# ---------------------------
# Helpers
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

def _words_to_lines(words: List[Tuple[str, Tuple[int,int,int,int]]],
                    y_tol: int) -> List[Tuple[str, Tuple[int,int,int,int]]]:
    """words -> 行クラスタ (joined_text, bbox)"""
    if not words:
        return []
    words = sorted(words, key=lambda w: (((w[1][1] + w[1][3]) // 2), w[1][0]))
    lines = []
    cur = []

    def flush():
        if not cur:
            return
        xs = [b[0] for _, b in cur] + [b[2] for _, b in cur]
        ys = [b[1] for _, b in cur] + [b[3] for _, b in cur]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        text = "".join(t for t, _ in sorted(cur, key=lambda w: w[1][0]))
        lines.append((text, (x1, y1, x2, y2)))

    base_y = None
    for t, (x1, y1, x2, y2) in words:
        yc = (y1 + y2) // 2
        if base_y is None or abs(yc - base_y) > y_tol:
            flush()
            base_y = yc
            cur = [(t, (x1, y1, x2, y2))]
        else:
            cur.append((t, (x1, y1, x2, y2)))
    flush()
    return sorted(lines, key=lambda it: it[1][1])

def google_ocr_line_boxes(pil_im: Image.Image, y_tol: int = None) -> List[Tuple[str, Tuple[int,int,int,int]]]:
    """行ボックスへ変換"""
    words = google_ocr_word_boxes(pil_im)
    if not words:
        return []
    if y_tol is None:
        y_tol = max(8, int(round(pil_im.height * 0.01)))
    return _words_to_lines(words, y_tol)

def remove_vertical_segments(im: Image.Image, segments: List[Tuple[int, int]]) -> Image.Image:
    """
    画像から [top, bottom) の縦区間を削除して詰める。
    segments は画像内y座標。重複/隣接は自動マージ。
    """
    if not segments:
        return im.copy()

    w, h = im.size
    # 正規化＆マージ
    segs = []
    for t, b in segments:
        t = max(0, min(h, t))
        b = max(0, min(h, b))
        if b <= t:
            continue
        segs.append((t, b))
    if not segs:
        return im.copy()

    segs.sort()
    merged = []
    cur_t, cur_b = segs[0]
    for t, b in segs[1:]:
        if t <= cur_b:  # 重複/隣接
            cur_b = max(cur_b, b)
        else:
            merged.append((cur_t, cur_b))
            cur_t, cur_b = t, b
    merged.append((cur_t, cur_b))

    # 残す区間を上から貼り合わせ
    keep_ranges = []
    cursor = 0
    for t, b in merged:
        if t > cursor:
            keep_ranges.append((cursor, t))
        cursor = b
    if cursor < h:
        keep_ranges.append((cursor, h))

    # 新しい高さ
    new_h = sum(b - a for a, b in keep_ranges)
    canvas = Image.new("RGBA", (w, new_h), (0, 0, 0, 0))
    y = 0
    for a, b in keep_ranges:
        piece = im.crop((0, a, w, b))
        canvas.paste(piece, (0, y))
        y += (b - a)
    return canvas

def compress_block7_by_rules(pil_im: Image.Image) -> Image.Image:
    """
    7番ブロック内の「詰め」処理：
      ・タイトル行（越域駐騎/駐機場）を見出しとして抽出
      ・各タイトルiの下端〜タイトルi+1の上端を“1ブロック”
      ・ブロック内に「免戦中」 or 時間(05:53/01:02:13等)があれば
         その“最後に出る行”の下端+MARGIN_AFTER_INFO 〜 次タイトル上端-MARGIN_BEFORE_NEXT_TITLE を削除
      ・どちらも無ければ タイトル下端〜次タイトル上端を削除
      ・最後のブロックの終端は画像最下部
    """
    im = pil_im.copy()
    w, h = im.size

    lines = google_ocr_line_boxes(im)

    # 行の抽出
    titles: List[Tuple[int, int]] = []
    info_rows: List[Tuple[int, int]] = []
    for text, (x1, y1, x2, y2) in lines:
        t_no_space = text.replace(" ", "")
        if RE_TITLE.search(t_no_space):
            titles.append((y1, y2))
        if RE_IMMUNE.search(t_no_space) or RE_TIME.search(text):
            info_rows.append((y1, y2))
    titles.sort(key=lambda p: p[0])
    info_rows.sort(key=lambda p: p[0])

    if not titles:
        # タイトルが検出できない場合はそのまま返す（安全側）
        return im

    # 削除すべき縦区間を集める
    remove_segments: List[Tuple[int, int]] = []
    for i, (ty1, ty2) in enumerate(titles):
        start = ty2
        end = titles[i + 1][0] if i + 1 < len(titles) else h
        if end <= start:
            continue

        # ブロック内で一番下に現れる info 行の下端を使う
        last_info_bottom = None
        for iy1, iy2 in info_rows:
            if start <= iy1 < end:
                last_info_bottom = iy2

        top = (last_info_bottom + MARGIN_AFTER_INFO) if last_info_bottom is not None else start
        bottom = max(top, end - MARGIN_BEFORE_NEXT_TITLE)
        if bottom > top:
            remove_segments.append((top, bottom))

    # その区間を削除し、上に詰めた画像を返す
    return remove_vertical_segments(im, remove_segments)

def hstack(im_left: Image.Image, im_right: Image.Image, gap: int = 8, bg=(0,0,0,0)) -> Image.Image:
    """左右に結合（高さは大きい方に合わせ中央寄せ）"""
    h = max(im_left.height, im_right.height)
    w = im_left.width + gap + im_right.width
    canvas = Image.new("RGBA", (w, h), bg)
    y1 = (h - im_left.height) // 2
    y2 = (h - im_right.height) // 2
    canvas.paste(im_left, (0, y1))
    canvas.paste(im_right, (im_left.width + gap, y2))
    return canvas

def vstack(images: List[Image.Image], gap: int = 8, bg=(0,0,0,0)) -> Image.Image:
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
    指定のスライス→トリム→（7を詰め処理）→合成→OpenAI OCR
    戻り値: (最終合成画像, OpenAI OCRテキスト, OpenAIへ送った画像bytes)
    """
    parts = slice_exact_7(pil_im, CUTS)

    # 1..7 のうち 2/4/6/7 を残す
    kept: Dict[int, Image.Image] = {}
    for idx in KEEP:
        block = parts[idx - 1]
        l_pct, r_pct = TRIM_RULES[idx]
        kept[idx] = trim_lr_percent(block, l_pct, r_pct)

    # 7番を“黒塗り相当の区間を削除して詰める”
    kept[7] = compress_block7_by_rules(kept[7])

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

        # 受信直後に即レス
        await ctx.reply("解析中…")

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

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)