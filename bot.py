import os
import discord
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv
from PIL import Image
import pytesseract

# ====== .env 読み込み ======
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
NOTIFY_CHANNEL_ID = int(os.getenv("NOTIFY_CHANNEL_ID"))

# ====== Discord 初期化 ======
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# ====== OCR関数 ======
def extract_text_from_image(image_path: str) -> str:
    """画像からOCRで文字を読み取る"""
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang="eng+jpn")  # 日本語・英語両方
    return text

def parse_server_number(ocr_text: str) -> str:
    """OCR結果から sxxxx 形式のサーバー番号を抽出"""
    import re
    match = re.search(r"s\d{3,4}", ocr_text)
    return match.group(0) if match else "unknown"

def parse_event_time(ocr_text: str) -> str:
    """OCR結果から 00:00:00 形式の時刻を抽出"""
    import re
    match = re.search(r"\d{2}:\d{2}:\d{2}", ocr_text)
    return match.group(0) if match else None

def parse_time_to_timedelta(time_str: str) -> timedelta:
    """OCRの時刻(02:38:18) → timedeltaに変換"""
    h, m, s = map(int, time_str.split(":"))
    return timedelta(hours=h, minutes=m, seconds=s)

# ====== 通知スケジュール ======
async def schedule_notification(mode, server_num, event_time):
    """イベント時刻の5分前と15秒前に通知"""
    notify_channel = client.get_channel(NOTIFY_CHANNEL_ID)
    if notify_channel is None:
        print("⚠ 通知チャンネルが見つからない！")
        return

    event_str = event_time.strftime("%H:%M:%S")
    now = datetime.now()
    diff = (event_time - now).total_seconds()

    if diff <= 0:
        return  # すでに過ぎていたら何もしない

    # 5分前通知
    if diff > 300:
        await asyncio.sleep(diff - 300)
        await notify_channel.send(f"⏳ {mode}-{server_num}-{event_str} の開始5分前！")

    # 15秒前通知
    now2 = datetime.now()
    diff2 = (event_time - now2).total_seconds()
    if diff2 > 15:
        await asyncio.sleep(diff2 - 15)
    elif diff2 <= 0:
        return
    await notify_channel.send(f"⚠️ {mode}-{server_num}-{event_str} の開始15秒前！")

# ====== OCR結果処理 ======
async def process_ocr_result(message, server_num, ocr_time, screenshot_timestamp_jst):
    """OCRで取得した時刻にスクショのタイムスタンプを加算 → 通知も設定"""
    if ocr_time:
        delta = parse_time_to_timedelta(ocr_time)
        real_event_time = screenshot_timestamp_jst + delta
        real_event_str = real_event_time.strftime("%H:%M:%S")

        # s1281なら防衛、それ以外は奪取
        mode = "防衛" if server_num == "s1281" else "奪取"
        final_message = f"{mode}-{server_num}-{real_event_str}"

        # スクショ投稿チャンネルにも送る
        await message.channel.send(final_message)

        # 通知専用チャンネルに 5分前＆15秒前通知をスケジュール
        asyncio.create_task(schedule_notification(
            mode,
            server_num,
            real_event_time
        ))

# ====== Discordイベント ======
@client.event
async def on_ready():
    print(f"✅ ログイン完了: {client.user}")

@client.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    # テキストだけならテスト動作
    if message.content.startswith("テスト"):
        server_num = "s1281"
        ocr_time = "02:38:18"
        screenshot_timestamp_jst = datetime.now()
        await process_ocr_result(message, server_num, ocr_time, screenshot_timestamp_jst)
        return

    # 画像が添付されてたらOCRする
    if message.attachments:
        for attachment in message.attachments:
            if attachment.filename.lower().endswith((".png", ".jpg", ".jpeg")):
                # 一時保存
                save_path = f"/tmp/{attachment.filename}"
                await attachment.save(save_path)

                # OCR読み取り
                ocr_text = extract_text_from_image(save_path)
                print("📸 OCR結果:\n", ocr_text)

                # サーバー番号・時間を抽出
                server_num = parse_server_number(ocr_text)
                ocr_time = parse_event_time(ocr_text)

                if not ocr_time:
                    await message.channel.send("⚠ 時刻が読み取れませんでした")
                    return

                # スクショが撮られた時間 (Discordのメッセージ投稿時刻を使用)
                screenshot_timestamp_jst = message.created_at.astimezone()

                # 処理 & 通知スケジュール
                await process_ocr_result(message, server_num, ocr_time, screenshot_timestamp_jst)

# ====== 実行 ======
client.run(TOKEN)