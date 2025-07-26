import discord
import asyncio
import re
from datetime import timedelta
from paddleocr import PaddleOCR

# 🔑 Discord BOT のトークンを設定
TOKEN = "YOUR_DISCORD_BOT_TOKEN"

# Discord の Intents 設定
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# PaddleOCR 初期化（英語、日本語両対応なら lang='japan' も可）
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# 時間抽出用正規表現（例: 02:34:56）
time_pattern = re.compile(r'(\d{1,2}:\d{2}:\d{2})')

# サーバー番号抽出用（例: s1281）
server_pattern = re.compile(r's\d{3,4}')

def parse_time_to_timedelta(time_str: str) -> timedelta:
    """OCR で取得した HH:MM:SS を timedelta に変換"""
    h, m, s = map(int, time_str.split(":"))
    return timedelta(hours=h, minutes=m, seconds=s)

@client.event
async def on_ready():
    print(f"✅ Logged in as {client.user}")

@client.event
async def on_message(message):
    # BOT のメッセージは無視
    if message.author.bot:
        return

    # メッセージに画像が含まれている場合のみ処理
    if message.attachments:
        for attachment in message.attachments:
            # 画像を一時保存
            img_path = f"/tmp/{attachment.filename}"
            await attachment.save(img_path)

            # PaddleOCR でテキスト抽出
            result = ocr.ocr(img_path, cls=True)
            extracted_text = " ".join([line[1][0] for block in result for line in block])
            print("📜 OCR結果:", extracted_text)

            # サーバー番号を取得
            server_match = server_pattern.search(extracted_text)
            server_num = server_match.group() if server_match else "???"

            # 時間を取得
            time_match = time_pattern.search(extracted_text)
            ocr_time = time_match.group() if time_match else None

            # スクリーンショットのアップロード時刻（UTC → JST）
            screenshot_timestamp = attachment.created_at.replace(tzinfo=None)
            screenshot_timestamp_jst = screenshot_timestamp + timedelta(hours=9)

            # デフォルトメッセージ
            final_message = "❌ 時間が読み取れませんでした"

            if ocr_time:
                # OCR 時刻を timedelta に変換
                delta = parse_time_to_timedelta(ocr_time)

                # スクショ時刻 + OCR時間 → 実際のイベント発生時刻
                real_event_time = screenshot_timestamp_jst + delta

                # ✅ 月日を削除して、HH:MM:SS のみ
                real_event_str = real_event_time.strftime("%H:%M:%S")

                # サーバー番号が s1281 の場合は「防衛」、それ以外は「奪取」
                mode = "防衛" if server_num == "s1281" else "奪取"

                # 形式: 奪取-s1281-14:30:39
                final_message = f"{mode}-{server_num}-{real_event_str}"

            await message.channel.send(final_message)

# BOT 実行
client.run(TOKEN)