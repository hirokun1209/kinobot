import cv2
import numpy as np
import easyocr
import re

# EasyOCRリーダー（数字専用モード）
reader = easyocr.Reader(['en'], gpu=False)

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # コントラスト強調
    img = cv2.convertScaleAbs(img, alpha=1.8, beta=10)
    # ノイズ除去
    img = cv2.GaussianBlur(img, (3,3), 0)
    # 二値化
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # シャープ化（OCRが文字の輪郭を認識しやすくなる）
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    img = cv2.filter2D(img, -1, kernel)
    return img

def ocr_digits_only(img_path):
    img = preprocess_image(img_path)
    # EasyOCRで数字だけ読む
    result = reader.readtext(img, detail=1, allowlist="0123456789")
    # 信頼度付き結果から文字列だけ抽出
    text_list = [text for (bbox, text, conf) in result if conf > 0.4]
    text = "".join(text_list)
    # 数字以外は除去
    return re.sub(r"[^0-9]", "", text)

# ==== 時間補正ロジック ====
def normalize_time(raw_digits):
    digits = re.sub(r"[^0-9]", "", raw_digits)
    if len(digits) < 4:  # 4桁未満なら無効
        return None

    # 桁数ごとに処理
    if len(digits) >= 6:
        # 6桁以上なら → 中央4桁を分秒扱い
        # 例: 5014220 → 00:42:20
        h = 0
        m = int(digits[-6:-4])
        s = int(digits[-4:-2])
    else:
        # 4桁なら 00:MM:SS として扱う
        h = 0
        m = int(digits[:2])
        s = int(digits[2:4])

    # 6時間以上は出ないので補正
    if h >= 6: h = 0
    if m >= 60: m = m % 60
    if s >= 60: s = s % 60

    return f"{h:02}:{m:02}:{s:02}"