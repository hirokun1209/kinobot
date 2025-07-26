import os
import re
import discord
from discord.ext import commands
from datetime import datetime, timedelta
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from paddleocr import PaddleOCR
import tempfile

TOKEN = os.getenv("DISCORD_BOT_TOKEN")
PREFIX = "!"

# OCR初期化（日本語対応）
ocr = PaddleOCR(use_angle_cls=True, lang='japan')

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=PREFIX, intents=intents)

# ======================
# ✅ ヘルスチェックサーバー
# ======================
def run_healthcheck():
    class HealthHandler(SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")

    server = HTTPServer(("0.0.0.0", 8080), HealthHandler)
    server.serve_forever()

# ======================
# ✅ 時刻パース補助
# ======================
def parse_time_string(time_str):
    """12:34 → datetime.timedelta"""
    match = re.match(r"^(\d{1,2}):(\d{2})$", time_str)
    if not match:
        return None
    h, m = map(int, match.groups())
    return timedelta(hours=h, minutes=m)

def add_times(base, add):
    """timedelta同士を足して 24時間超えは繰り返し"""
    total_minutes = (base.total_seconds() + add.total_seconds()) / 60
    hours = int(total_minutes // 60) % 24
    minutes = int(total_minutes % 60)
    return f"{hours:02d}:{minutes:02d}"

# ======================
# ✅ 起動イベント
# ======================
@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")

# ======================
# ✅ OCRコマンド
# ======================
@bot.command()
async def ocr(ctx):
    """添付画像をOCR解析"""
    if not ctx.message.attachments:
        await ctx.send("❌ 画像を添付してね！")
        return

    img = ctx.message.attachments[0]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        await img.save(tmp.name)
        tmp_path = tmp.name

    result = ocr.ocr(tmp_path, cls=True)
    os.remove(tmp_path)

    text_list = []
    for line in result:
        for word_info in line:
            text_list.append(word_info[1][0])

    if text_list:
        await ctx.send("✅ OCR結果:\n```\n" + "\n".join(text_list) + "\n```")
    else:
        await ctx.send("❌ テキストが検出できなかったよ")

# ======================
# ✅ 時間計算コマンド
# ======================
@bot.command()
async def time(ctx, base_time: str, add_time: str):
    """
    時刻計算: !time 12:30 01:15 → 13:45
    """
    base = parse_time_string(base_time)
    add = parse_time_string(add_time)

    if not base or not add:
        await ctx.send("❌ 時刻は HH:MM 形式で入力してね（例: !time 12:30 01:15）")
        return

    result = add_times(base, add)
    await ctx.send(f"⏰ **{base_time} + {add_time} = {result}**")

# ======================
# ✅ OCR + 時刻抽出 → 計算
# ======================
@bot.command()
async def ocr_time(ctx, add_time: str):
    """
    添付画像からOCRした時間に追加時間を足す:
    !ocr_time 01:15 （画像に12:30 → 13:45）
    """
    if not ctx.message.attachments:
        await ctx.send("❌ 画像を添付してね！")
        return

    # 画像保存
    img = ctx.message.attachments[0]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        await img.save(tmp.name)
        tmp_path = tmp.name

    # OCR解析
    result = ocr.ocr(tmp_path, cls=True)
    os.remove(tmp_path)

    # OCR結果から時間パターンを検索
    detected_text = " ".join(word_info[1][0] for line in result for word_info in line)
    time_match = re.search(r"(\d{1,2}:\d{2})", detected_text)
    if not time_match:
        await ctx.send("❌ OCRで時間が見つからなかったよ")
        return

    base_time_str = time_match.group(1)
    base_time = parse_time_string(base_time_str)
    add = parse_time_string(add_time)

    if not add:
        await ctx.send("❌ 追加時間は HH:MM 形式で入力してね")
        return

    result_time = add_times(base_time, add)
    await ctx.send(f"🖼 OCRで検出した時間: `{base_time_str}`\n➕ 追加 `{add_time}`\n➡ **{result_time}**")

if __name__ == "__main__":
    # ✅ ヘルスチェックHTTPサーバーをバックグラウンド起動
    threading.Thread(target=run_healthcheck, daemon=True).start()
    # ✅ Discord BOT起動
    bot.run(TOKEN)