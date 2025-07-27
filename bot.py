import os
import discord
import io
import cv2
import numpy as np
from paddleocr import PaddleOCR
from datetime import datetime, timedelta
from PIL import Image

# ✅ Discord Botトークンを環境変数から取得
TOKEN = os.getenv("DISCORD_TOKEN")
if not TOKEN:
    raise ValueError("❌ DISCORD_TOKEN が設定されていません！Koyeb の Environment Variables を確認してください。")

# ✅ OCR初期化（日本語）
ocr = PaddleOCR(use_angle_cls=True, lang='japan')

# ✅ Discordクライアント
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

def add_time(base_time_str: str, duration_str: str) -> str:
    """右上の時間 + 免戦時間を計算して解除時刻を返す"""
    try:
        base_time = datetime.strptime(base_time_str, "%H:%M:%S")
    except ValueError:
        return base_time_str  # 読み取り失敗時はそのまま返す

    parts = duration_str.strip().split(":")
    if len(parts) == 3:  # HH:MM:SS
        h, m, s = map(int, parts)
    elif len(parts) == 2:  # MM:SS → 0時間扱い
        h = 0
        m, s = map(int, parts)
    else:
        return base_time_str  # 想定外 → 右上時間そのまま返す

    delta = timedelta(hours=h, minutes=m, seconds=s)
    new_time = (base_time + delta).time()
    return new_time.strftime("%H:%M:%S")

def crop_top_right(img: np.ndarray) -> np.ndarray:
    """右上20%の領域をトリミング"""
    h, w, _ = img.shape
    return img[0:int(h * 0.2), int(w * 0.8):w]  # 上20% & 右20%

def crop_center_area(img: np.ndarray) -> np.ndarray:
    """上下35%をカットして中央エリアをトリミング"""
    h, w, _ = img.shape
    return img[int(h * 0.35):int(h * 0.65), 0:w]

def extract_text_from_image(img: np.ndarray):
    """PaddleOCRで文字認識"""
    result = ocr.ocr(img, cls=True)
    return [line[1][0] for line in result[0]] if result and result[0] else []

def parse_info(center_texts, top_time_texts):
    """OCR結果から必要な情報を抽出 & 整形"""
    # ✅ 右上の時間（必ず HH:MM:SS）
    top_time = next((t for t in top_time_texts if ":" in t and len(t.split(":")) == 3), None)

    # ✅ サーバー番号 / 駐騎場番号 / 免戦時間を抽出
    server = None
    place_num = None
    duration = None

    for t in center_texts:
        # サーバー番号 (例: s1281)
        if t.startswith("s") and t[1:].isdigit():
            server = t
        # 駐騎場番号 (数字だけ)
        elif t.isdigit():
            place_num = t
        # 免戦時間 (HH:MM:SS or MM:SS)
        elif ":" in t:
            duration = t

    # ✅ 必要な情報が揃わない場合は None
    if not (server and place_num and duration and top_time):
        return None

    # ✅ モード判定（s1281は警備、それ以外は奪取）
    mode = "警備" if server == "s1281" else "奪取"

    # ✅ 解除時刻計算
    new_time = add_time(top_time, duration)

    # ✅ 出力フォーマット → `警備 s1281-3-20:14:54`
    return f"{mode} {server}-{place_num}-{new_time}"

@client.event
async def on_ready():
    print(f"✅ ログイン成功！Bot名: {client.user}")

@client.event
async def on_message(message):
    # ✅ Bot自身のメッセージは無視
    if message.author.bot:
        return

    # ✅ 画像が添付されたメッセージのみ処理
    if message.attachments:
        results = []

        for attachment in message.attachments:
            img_bytes = await attachment.read()

            # Pillowで画像を開き、OpenCV形式に変換
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # 右上 & 中央部分をトリミング
            top_img = crop_top_right(img_np)
            center_img = crop_center_area(img_np)

            # OCR結果
            top_texts = extract_text_from_image(top_img)
            center_texts = extract_text_from_image(center_img)

            # 必要情報をパース
            info = parse_info(center_texts, top_texts)

            if info:
                results.append(info)
            else:
                results.append("⚠️ 必要な情報が読み取れませんでした")

        # OCR結果をまとめて返信
        await message.channel.send("\n".join(results))

# ✅ Bot起動（エラー時はメッセージを表示）
if __name__ == "__main__":
    try:
        client.run(TOKEN)
    except discord.errors.LoginFailure:
        print("❌ Discord トークンが無効です！Koyeb の Environment Variables を確認してください。")
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")