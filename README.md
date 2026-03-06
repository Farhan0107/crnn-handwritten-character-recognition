# 🧠 CRNN Handwriting OCR

A deep learning system for recognizing handwritten text using a **CNN + BiLSTM + CTC** architecture, trained on the EMNIST dataset. Includes a real-time web dashboard and REST API server.


---

## 📸 Demo

> Draw or upload handwriting → Get recognized text instantly

The dashboard supports:
- ✏️ Drawing directly on canvas
- 📁 Uploading PNG / JPG / TIFF / BMP images
- 🔤 Multi-character and full word recognition
- 📊 Confidence scores, processing time, recognition history

---

## 🏗️ Architecture

```
Input Image (1 × 32 × 32)
        ↓
CNN Backbone (7 conv blocks)    ← extracts visual features
        ↓
Reshape to sequence             ← width becomes time steps
        ↓
BiLSTM × 2 layers               ← reads sequence left ↔ right
        ↓
CTC Decode                      ← collapses to final text
        ↓
Recognized Text
```

| Component | Detail |
|---|---|
| CNN | 7 conv blocks, 1 → 64 → 128 → 256 → 512 channels |
| RNN | 2× Bidirectional LSTM, hidden size 256 |
| Loss | CTC Loss (alignment-free) |
| Parameters | 8.3 million |
| Input size | 1 × 32 × 32 grayscale |
| Classes | 63 (0-9, A-Z, a-z + CTC blank) |

---

## 📁 Project Structure

```
crnn_handwriting/
│
├── crnn_notebook.ipynb       # Training notebook (run this first)
├── model.py                  # CRNN model architecture
├── dataset.py                # EMNIST dataset loader + vocabulary
├── server.py                 # Flask REST API server
├── dashboard.html            # Web UI (open in browser)
│
├── checkpoints/
│   └── best_crnn.pth         # Saved model (created after training)
│
├── data/                     # EMNIST dataset (auto-downloaded)
└── requirements.txt
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/crnn-handwriting-ocr.git
cd crnn-handwriting-ocr
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

Open `crnn_notebook.ipynb` in Jupyter and run all cells. This will:
- Download EMNIST dataset automatically (~560 MB, one time only)
- Train the CRNN model
- Save the best checkpoint to `checkpoints/best_crnn.pth`

> **Training time:** ~22 minutes on Apple Silicon (MPS) · ~45 min on CPU

### 4. Start the server

```bash
python server.py
```

You should see:
```
[Server] EasyOCR ready ✅
[Server] CRNN loaded — epoch=14 acc=85.79%
✅  Server ready → http://localhost:8080
```

### 5. Open the dashboard

Open `dashboard.html` in your browser. Draw or upload handwriting and click **Recognize Handwriting**.

---

## 📦 Requirements

```
torch
torchvision
flask
flask-cors
opencv-python
pillow
numpy
easyocr
scikit-image
jupyter
matplotlib
seaborn
scikit-learn
```

Install all at once:
```bash
pip install torch torchvision flask flask-cors opencv-python pillow numpy easyocr scikit-image jupyter matplotlib seaborn scikit-learn
```

---

## 🔌 API Reference

The Flask server exposes two endpoints:

### `POST /predict`

Send a base64-encoded image, receive recognized text.

**Request:**
```json
{
  "image": "<base64 encoded image string>"
}
```

**Response:**
```json
{
  "recognized_text": "Hello",
  "confidence": 0.87,
  "char_count": 5,
  "line_count": 1,
  "mode": "EasyOCR",
  "epoch": 14,
  "model_accuracy": 85.79
}
```

### `GET /status`

Check if the server is running and get model info.

**Response:**
```json
{
  "status": "running",
  "device": "mps",
  "accuracy": 85.79,
  "epoch": 14,
  "classes": 63,
  "engine": "EasyOCR + CRNN"
}
```

---

## 📊 Training Results

| Epoch | Accuracy |
|---|---|
| 1 | 78.23% |
| 3 | 84.25% |
| 8 | 85.30% |
| 14 | **85.79%** ← best |
| 15 | 85.71% |

- **Dataset:** EMNIST ByClass (814,255 total images, 60,000 used)
- **Optimizer:** Adam, lr = 3e-4
- **Scheduler:** ReduceLROnPlateau (patience=3, factor=0.5)
- **CER:** 0.1421

---

## 🔍 How It Works

### CNN — The Eyes
The 7-layer convolutional backbone progressively extracts features from the input image. Early layers detect edges and strokes, later layers detect full character shapes. MaxPool layers with asymmetric kernels `(2,1)` collapse height while preserving width — keeping character position information intact.

### BiLSTM — The Memory
The feature map is reshaped so each column becomes a time step. Two bidirectional LSTM layers read this sequence left→right and right→left simultaneously, giving every position context from both directions.

### CTC — The Decoder
Connectionist Temporal Classification collapses repeated predictions and blanks into clean text without needing manual character segmentation. `HHHH--ii--` becomes `Hi`.

---

## ⚡ Recognition Engines

The server uses a two-engine approach:

| Engine | Used when | Accuracy |
|---|---|---|
| **EasyOCR** (primary) | Always (if installed) | ~95%+ |
| **CRNN** (fallback) | EasyOCR finds nothing | ~85.79% |

EasyOCR is pretrained on millions of real-world images and handles full words and sentences out of the box. The custom CRNN is used as a fallback and represents the core deep learning work.

---

## 🛠️ Device Support

The model automatically detects and uses the best available device:

```python
CUDA  → NVIDIA GPU (fastest)
MPS   → Apple Silicon GPU (M1/M2/M3)
CPU   → fallback (slowest)
```

---

## 📈 Improving Accuracy

To improve beyond 85.79%, try:

1. **Train on more data** — change `MAX_TRAIN = None` in the notebook to use all 697k samples
2. **Train longer** — increase `EPOCHS = 30`
3. **IAM Dataset** — train on real handwritten words instead of isolated characters
4. **Fine-tune on your own handwriting** — collect 100+ samples of your own writing

---

## ⚠️ Known Limitations

- Trained on isolated characters — struggles with cursive/connected writing
- Similar characters are sometimes confused: `l` / `1`, `o` / `0`, `I` / `l`
- Best results with clear, separated characters on a white background
- Canvas drawing works best with strokes similar to printed characters

---

## 🤝 Team

Built as a deep learning project demonstrating end-to-end OCR pipeline design, model training, and deployment.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset) — Cohen et al., 2017
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) — pretrained OCR engine
- [PyTorch](https://pytorch.org/) — deep learning framework
- CRNN architecture inspired by: *"An End-to-End Trainable Neural Network for Image-based Sequence Recognition"* — Shi et al., 2015