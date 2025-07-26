import os
import discord
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
NOTIFY_CHANNEL_ID = int(os.getenv("NOTIFY_CHANNEL_ID"))  # 通知専用チャンネル

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# ========== 時刻処理 ==========
def parse_time_to_timedelta(time_str: str) -> timedelta:
    """OCRの時刻(02:38:18) → timedeltaに変換"""
    h, m, s = map(int, time_str.split(":"))
    return timedelta(hours=h, minutes=m, seconds=s)

# ========== 通知スケジュール ==========
async def schedule_notification(mode, server_num, event_time):
    """
    イベント時刻の5分前と15秒前に通知専用チャンネルにメッセージ送信
    """
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

# ========== OCR結果処理 ==========
async def process_ocr_result(message, server_num, ocr_time, screenshot_timestamp_jst):
    """
    OCRで取得した時刻にスクショのタイムスタンプを加算 → 通知も設定
    """
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

# ========== Discord起動 ==========
@client.event
async def on_ready():
    print(f"✅ ログイン完了: {client.user}")

@client.event
async def on_message(message):
    if message.author.bot:
        return

    # ここでOCR処理やサーバー番号抽出して `server_num`, `ocr_time`, `screenshot_timestamp_jst` を決める
    # 仮の例
    if message.content.startswith("テスト"):
        server_num = "s1281"
        ocr_time = "02:38:18"
        screenshot_timestamp_jst = datetime.now()
        await process_ocr_result(message, server_num, ocr_time, screenshot_timestamp_jst)

client.run(TOKEN)