import os
import discord
import io
import cv2
import re
import asyncio
import numpy as np
from paddleocr import PaddleOCR
from datetime import datetime, timedelta, timezone
from PIL import Image

# =======================
#  タイムゾーン設定 (JST固定)
# =======================
JST = timezone(timedelta(hours=9))  # 日本標準時

# =======================
#  BOT設定
# =======================
TOKEN = os.getenv("DISCORD_TOKEN")
NOTIFY_CHANNEL_ID = int(os.getenv("NOTIFY_CHANNEL_ID", "0"))

if not TOKEN:
    raise ValueError("❌ DISCORD_TOKEN が設定されていません！")

# =======================
#  Discord Client定義
# =======================
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# =======================
#  OCR初期化
# =======================
ocr = PaddleOCR(use_angle_cls=True, lang='japan')

# =======================
#  通知ブロック管理
# =======================
pending_places = {}  # key: txt, value: (解除予定時刻, テキスト, サーバー番号, 登録時刻)
summary_blocks = []  # [{ "events": [(dt, txt)], "min": dt, "max": dt, "msg": discord.Message or None }]
SKIP_NOTIFY_START = 2
SKIP_NOTIFY_END = 14

# =======================
#  JSTユーティリティ
# =======================
def now_jst():
    return datetime.now(JST)

def cleanup_old_entries():
    now = now_jst()
    expired = [k for k,v in pending_places.items() if (now - v[3]) > timedelta(hours=6)]
    for k in expired:
        del pending_places[k]

def crop_top_right(img: np.ndarray) -> np.ndarray:
    h, w, _ = img.shape
    return img[0:int(h * 0.2), int(w * 0.7):w]

def crop_center_area(img: np.ndarray) -> np.ndarray:
    h, w, _ = img.shape
    return img[int(h * 0.35):int(h * 0.65), 0:w]

def extract_text_from_image(img: np.ndarray):
    result = ocr.ocr(img, cls=True)
    return [line[1][0] for line in result[0]] if result and result[0] else []

def extract_server_number(center_texts):
    for t in center_texts:
        match = re.search(r"[sS](\d{3,4})", t)
        if match:
            return match.group(1)
    return None

def add_time(base_time_str, duration_str):
    today = now_jst().date()
    try:
        base_time_only = datetime.strptime(base_time_str, "%H:%M:%S").time()
    except ValueError:
        return None, None
    base_time = datetime.combine(today, base_time_only, tzinfo=JST)

    parts = duration_str.split(":")
    if len(parts)==3: h,m,s = map(int,parts)
    elif len(parts)==2: h=0; m,s=map(int,parts)
    else: return None,None

    dt = base_time + timedelta(hours=h, minutes=m, seconds=s)
    return dt, dt.strftime("%H:%M:%S")

def parse_multiple_places(center_texts, top_time_texts):
    res=[]
    top_time = next((t for t in top_time_texts if re.match(r"\d{2}:\d{2}:\d{2}", t)), None)
    if not top_time: return []
    server = extract_server_number(center_texts)
    if not server: return []
    mode = "警備" if server=="1281" else "奪取"
    current=None
    for t in center_texts:
        place_match = re.search(r"越域駐騎場(\d+)",t)
        if place_match: current = place_match.group(1)
        duration_match = re.search(r"免戦中(\d{1,2}:\d{2}(?::\d{2})?)",t)
        if duration_match and current:
            dt, unlock = add_time(top_time,duration_match.group(1))
            if dt: res.append((dt,f"{mode} {server}-{current}-{unlock}"))
            current=None
    return res

# =======================
#  ブロック管理ロジック
# =======================
def find_or_create_block(new_dt):
    """45分以内なら既存ブロック、超えたら新規"""
    for block in summary_blocks:
        if new_dt <= block["max"] + timedelta(minutes=45):
            return block
    # 新規
    b={"events":[],"min":new_dt,"max":new_dt,"msg":None}
    summary_blocks.append(b)
    return b

def format_block_msg(block):
    lines=["⏰ スケジュールのお知らせ📢",""]
    events=sorted(block["events"],key=lambda x:x[0])
    lines += [txt+"  " for _,txt in events]
    lines.append("")
    diff=int((block["min"]-now_jst()).total_seconds()//60)
    if diff<30:
        lines.append(f"⚠️ {diff}分後に始まるよ⚠️")
    else:
        lines.append("⚠️ 30分後に始まるよ⚠️")
    return "\n".join(lines)

async def schedule_block_summary(block,channel):
    # wait until min_time -30min
    wait_sec=(block["min"]-timedelta(minutes=30)-now_jst()).total_seconds()
    if wait_sec<0: wait_sec=0
    await asyncio.sleep(wait_sec)
    # 送信または編集
    if not block["msg"]:
        msg_txt=format_block_msg(block)
        block["msg"]=await channel.send(msg_txt)
    else:
        await block["msg"].edit(content=format_block_msg(block))

async def handle_new_event(dt,txt,channel):
    block=find_or_create_block(dt)
    block["events"].append((dt,txt))
    block["min"]=min(block["min"],dt)
    block["max"]=max(block["max"],dt)

    # 通知前ならスケジュールセット
    if dt>=block["min"]:
        if block["msg"]:
            # すでに通知済み→編集で追加
            await block["msg"].edit(content=format_block_msg(block))
        else:
            # 最小時間30分前通知タスク
            asyncio.create_task(schedule_block_summary(block,channel))

# =======================
#  通知 (個別2分前/15秒前)
# =======================
async def schedule_notification(unlock_dt,text,channel):
    now=now_jst()
    if unlock_dt<=now: return
    if text.startswith("奪取") and not (SKIP_NOTIFY_START<=unlock_dt.hour<SKIP_NOTIFY_END):
        n2=unlock_dt - timedelta(minutes=2)
        n15=unlock_dt - timedelta(seconds=15)
        if n2>now:
            await asyncio.sleep((n2-now).total_seconds())
            await channel.send(f"⏰ {text} **2分前です！！**")
        if n15>now_jst():
            await asyncio.sleep((n15-now_jst()).total_seconds())
            await channel.send(f"⏰ {text} **15秒前です！！**")

# =======================
#  イベント処理
# =======================
@client.event
async def on_ready():
    print("✅ ログイン成功！")

@client.event
async def on_message(message):
    if message.author.bot: return
    cleanup_old_entries()
    channel=client.get_channel(NOTIFY_CHANNEL_ID) if NOTIFY_CHANNEL_ID else None

    # デバッグ "!1234-7-12:34:56"
    if message.content.startswith("!"):
        m=re.match(r"!([0-9]{3,4})-(\d+)-(\d{2}:\d{2}:\d{2})",message.content)
        if m:
            server,place,t=m.groups()
            if len(server)==3: server="1"+server
            mode="警備" if server=="1281" else "奪取"
            txt=f"{mode} {server}-{place}-{t}"
            dt=datetime.combine(now_jst().date(),datetime.strptime(t,"%H:%M:%S").time(),tzinfo=JST)
            pending_places[txt]=(dt,txt,server,now_jst())
            await message.channel.send(f"✅デバッグ登録:{txt}")
            if channel:
                asyncio.create_task(handle_new_event(dt,txt,channel))
                asyncio.create_task(schedule_notification(dt,txt,channel))
        return

    # 手動 "281-1-12:34:56"
    manual=re.findall(r"(\d{3,4})-(\d+)-(\d{2}:\d{2}:\d{2})",message.content)
    if manual:
        for server,place,t in manual:
            if len(server)==3: server="1"+server
            mode="警備" if server=="1281" else "奪取"
            txt=f"{mode} {server}-{place}-{t}"
            dt=datetime.combine(now_jst().date(),datetime.strptime(t,"%H:%M:%S").time(),tzinfo=JST)
            if txt not in pending_places:
                pending_places[txt]=(dt,txt,server,now_jst())
                await message.channel.send(f"✅手動登録:{txt}")
                if channel:
                    asyncio.create_task(handle_new_event(dt,txt,channel))
                    if txt.startswith("奪取"):
                        asyncio.create_task(schedule_notification(dt,txt,channel))
        return

    # 画像OCR
    if message.attachments:
        processing=await message.channel.send("🔄解析中…")
        for a in message.attachments:
            b=await a.read()
            img=Image.open(io.BytesIO(b)).convert("RGB")
            np_img=cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
            top=crop_top_right(np_img)
            center=crop_center_area(np_img)
            top_txts=extract_text_from_image(top)
            center_txts=extract_text_from_image(center)
            parsed=parse_multiple_places(center_txts,top_txts)
            for dt,txt in parsed:
                if txt not in pending_places:
                    pending_places[txt]=(dt,txt,"",now_jst())
                    if channel:
                        asyncio.create_task(handle_new_event(dt,txt,channel))
                        if txt.startswith("奪取"):
                            asyncio.create_task(schedule_notification(dt,txt,channel))
        cleanup_old_entries()
        await processing.edit(content="✅OCR処理完了")

if __name__=="__main__":
    client.run(TOKEN)