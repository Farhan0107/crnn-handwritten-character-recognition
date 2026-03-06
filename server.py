"""
server.py — EasyOCR + CRNN Hybrid Server
==========================================
Primary engine  : EasyOCR  (pretrained, 95%+ accuracy on real handwriting)
Fallback engine : Your trained CRNN model

EasyOCR handles:
  - Words, sentences, paragraphs, full documents
  - Multiple lines, mixed case, digits, symbols
  - Real handwriting photos AND drawn canvas input

Install:
    pip install easyocr flask flask-cors opencv-python pillow numpy torch

Run:
    python server.py
    (first run downloads EasyOCR model ~100 MB, once only)
"""

import base64
import io
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, ImageEnhance
import torchvision.transforms as T
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── EasyOCR ───────────────────────────────────────────────────────────────────
try:
    import easyocr
    print('[Server] Loading EasyOCR (downloads ~100 MB on first run)...')
    OCR_READER = easyocr.Reader(
        ['en'],
        gpu=torch.cuda.is_available(),
        verbose=False,
    )
    print('[Server] EasyOCR ready ✅')
    EASYOCR_AVAILABLE = True
except ImportError:
    print('[Server] ⚠️  EasyOCR not found. Run: pip install easyocr')
    EASYOCR_AVAILABLE = False

# ── CRNN (fallback) ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model   import CRNN
from dataset import NUM_CLASSES, decode_ctc

CHECKPOINT = './checkpoints/best_crnn.pth'
IMG_H = 32
IMG_W = 32
PORT  = 8080

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

print(f'[Server] Device  : {DEVICE}')

if not os.path.exists(CHECKPOINT):
    print(f'❌  Checkpoint not found: {CHECKPOINT}')
    sys.exit(1)

crnn_model = CRNN(num_classes=NUM_CLASSES).to(DEVICE)
ckpt       = torch.load(CHECKPOINT, map_location=DEVICE)
crnn_model.load_state_dict(ckpt['model_state'])
crnn_model.eval()
print(f'[Server] CRNN loaded — epoch={ckpt.get("epoch","?")} acc={ckpt.get("accuracy",0)*100:.1f}%')

char_transform = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])


# ══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def enhance_for_easyocr(pil_img):
    """
    Prepare image for EasyOCR.
    - Upscales small images (needs ~32px tall text minimum)
    - Boosts contrast with CLAHE
    - Inverts dark-background images (canvas drawings)
    """
    img = pil_img.convert('RGB')
    w, h = img.size

    # Upscale tiny images
    if h < 100:
        scale = max(2, 100 // h)
        img   = img.resize((w * scale, h * scale), Image.LANCZOS)

    arr  = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Invert if dark background (canvas = white strokes on black bg)
    if gray.mean() < 100:
        arr = cv2.bitwise_not(arr)

    # CLAHE contrast enhancement
    lab       = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b   = cv2.split(lab)
    clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l         = clahe.apply(l)
    arr       = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

    return arr


def binarize(pil_img):
    """→ numpy uint8, BLACK text on WHITE background (CRNN fallback)."""
    img = pil_img.convert('L')
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = ImageOps.autocontrast(img, cutoff=2)
    arr = np.array(img, dtype=np.uint8)
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if binary.mean() < 128:
        binary = cv2.bitwise_not(binary)
    return binary


# ══════════════════════════════════════════════════════════════════════════════
#  EASYOCR PIPELINE  (primary)
# ══════════════════════════════════════════════════════════════════════════════

def process_easyocr(pil_img):
    """
    EasyOCR recognition — handles words, sentences, paragraphs, documents.
    """
    arr     = enhance_for_easyocr(pil_img)
    results = OCR_READER.readtext(arr, detail=1, paragraph=False, min_size=10, text_threshold=0.3, low_text=0.3, link_threshold=0.3)
    print(f'[EasyOCR] {len(results)} region(s) detected')

    if not results:
        print('[EasyOCR] Nothing found — falling back to CRNN')
        return process_crnn_multi(pil_img)

    # Sort top→bottom, then left→right
    results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))

    # Group detections into lines by y-position proximity
    line_texts  = []
    current_line = []
    prev_y       = None

    for bbox, text, conf in results:
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        if prev_y is None or abs(y_center - prev_y) > 30:
            if current_line:
                line_texts.append(' '.join(current_line))
            current_line = [text]
            prev_y = y_center
        else:
            current_line.append(text)

    if current_line:
        line_texts.append(' '.join(current_line))

    recognized = '\n'.join(line_texts).strip()
    avg_conf   = float(np.mean([r[2] for r in results]))
    char_count = len(recognized.replace('\n', '').replace(' ', ''))

    print(f'[EasyOCR] "{recognized}"  conf={avg_conf:.2f}')

    return {
        'recognized_text': recognized if recognized else '[unclear]',
        'confidence'     : round(avg_conf, 4),
        'line_count'     : len(line_texts),
        'char_count'     : char_count,
        'mode'           : 'EasyOCR',
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CRNN PIPELINE  (fallback)
# ══════════════════════════════════════════════════════════════════════════════

def find_char_boxes(binary_img):
    inv = cv2.bitwise_not(binary_img)
    H, W = binary_img.shape
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    dilated = cv2.dilate(inv, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    min_h = max(4, H * 0.03)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 2 or h < min_h:
            continue
        if w > h * 2.5 and h > 10:
            n_chars = max(2, round(w / h))
            cw = w // n_chars
            for i in range(n_chars):
                boxes.append((x + i * cw, y, cw, h))
        else:
            boxes.append((x, y, w, h))

    if not boxes:
        return []

    boxes.sort(key=lambda b: b[1])
    lines = []
    for box in boxes:
        x, y, w, h = box
        placed = False
        for line in lines:
            ly0 = min(b[1] for b in line)
            ly1 = max(b[1] + b[3] for b in line)
            if min(y + h, ly1) - max(y, ly0) > min(h, ly1 - ly0) * 0.3:
                line.append(box); placed = True; break
        if not placed:
            lines.append([box])

    return [sorted(line, key=lambda b: b[0]) for line in lines]


def box_to_tensor(binary_img, x, y, w, h, pad=4):
    H_img, W_img = binary_img.shape
    x0 = max(0, x-pad);  y0 = max(0, y-pad)
    x1 = min(W_img, x+w+pad);  y1 = min(H_img, y+h+pad)
    crop = binary_img[y0:y1, x0:x1]
    pil  = ImageOps.invert(Image.fromarray(crop))
    pil.thumbnail((IMG_W, IMG_H), Image.LANCZOS)
    canvas = Image.new('L', (IMG_W, IMG_H), 0)
    canvas.paste(pil, ((IMG_W-pil.width)//2, (IMG_H-pil.height)//2))
    return char_transform(canvas).unsqueeze(0)


@torch.no_grad()
def recognize_batch(tensors):
    if not tensors:
        return []
    batch  = torch.cat(tensors, dim=0).to(DEVICE)
    logits = crnn_model(batch)
    probs  = logits.softmax(2)
    return [(decode_ctc(probs[:, b, :].argmax(1).cpu()) or '?',
             float(probs[:, b, :].max(1).values.mean()))
            for b in range(batch.size(0))]


def process_crnn_multi(pil_img):
    binary      = binarize(pil_img)
    line_groups = find_char_boxes(binary)
    if not line_groups:
        return process_crnn_single(pil_img)

    all_lines = [];  all_confs = [];  total_chars = 0
    for line_boxes in line_groups:
        gaps    = [line_boxes[i+1][0]-(line_boxes[i][0]+line_boxes[i][2]) for i in range(len(line_boxes)-1)]
        avg_w   = np.mean([b[2] for b in line_boxes])
        spaces  = {i for i, g in enumerate(gaps) if g > avg_w * 1.1}
        tensors = [box_to_tensor(binary, x, y, w, h) for x, y, w, h in line_boxes]
        chars   = recognize_batch(tensors)
        line_text = ''
        for i, (ch, conf) in enumerate(chars):
            line_text += ch;  all_confs.append(conf);  total_chars += 1
            if i in spaces: line_text += ' '
        all_lines.append(line_text)

    recognized = '\n'.join(all_lines).strip()
    return {
        'recognized_text': recognized or '[unclear]',
        'confidence'     : round(float(np.mean(all_confs)) if all_confs else 0.0, 4),
        'line_count'     : len(line_groups),
        'char_count'     : total_chars,
        'mode'           : 'CRNN',
    }


def process_crnn_single(pil_img):
    binary = binarize(pil_img)
    inv    = cv2.bitwise_not(binary)
    rows   = np.any(inv > 0, axis=1);  cols = np.any(inv > 0, axis=0)
    if rows.any() and cols.any():
        r0, r1 = np.where(rows)[0][[0,-1]]
        c0, c1 = np.where(cols)[0][[0,-1]]
        binary = binary[max(0,r0-4):r1+4, max(0,c0-4):c1+4]
    pil = ImageOps.invert(Image.fromarray(binary))
    pil.thumbnail((IMG_W, IMG_H), Image.LANCZOS)
    canvas = Image.new('L', (IMG_W, IMG_H), 0)
    canvas.paste(pil, ((IMG_W-pil.width)//2, (IMG_H-pil.height)//2))
    tensor = char_transform(canvas).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs   = crnn_model(tensor).softmax(2)
        text    = decode_ctc(probs.argmax(2).squeeze(1).cpu())
        conf    = float(probs.max(2).values.mean())
    return {
        'recognized_text': text or '[unclear]',
        'confidence'     : round(conf, 4),
        'line_count'     : 1,
        'char_count'     : len(text) if text else 0,
        'mode'           : 'CRNN',
    }


# ══════════════════════════════════════════════════════════════════════════════
#  FLASK
# ══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        img_b64 = data['image']
        if ',' in img_b64:
            img_b64 = img_b64.split(',')[1]

        pil_img = Image.open(io.BytesIO(base64.b64decode(img_b64)))
        w, h    = pil_img.size
        print(f'[OCR] Image: {w}×{h}')

        if EASYOCR_AVAILABLE:
            result = process_easyocr(pil_img)
        elif w >= 64 or h >= 64:
            result = process_crnn_multi(pil_img)
        else:
            result = process_crnn_single(pil_img)

        result['epoch']          = ckpt.get('epoch', '?')
        result['model_accuracy'] = round(ckpt.get('accuracy', 0) * 100, 2)
        return jsonify(result)

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/status', methods=['GET'])
def status():
    engine = 'EasyOCR + CRNN' if EASYOCR_AVAILABLE else 'CRNN only'
    return jsonify({
        'status'   : 'running',
        'device'   : str(DEVICE),
        'accuracy' : round(ckpt.get('accuracy', 0) * 100, 2),
        'epoch'    : ckpt.get('epoch', '?'),
        'classes'  : NUM_CLASSES,
        'engine'   : engine,
    })


if __name__ == '__main__':
    engine = 'EasyOCR + CRNN fallback' if EASYOCR_AVAILABLE else 'CRNN only'
    print(f'\n✅  Server ready [{engine}] → http://localhost:{PORT}\n')
    app.run(host='0.0.0.0', port=PORT, debug=False)