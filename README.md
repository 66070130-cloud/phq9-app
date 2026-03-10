# 🧠 PHQ-9 Depression Screening App

แอปพลิเคชันประเมินสุขภาพจิตด้วย PHQ-9 โดยใช้ AI วิเคราะห์เสียงและข้อความ Deploy บน Google Cloud Run

---

## 📋 ภาพรวม

```
ผู้ใช้อัดเสียงตอบ PHQ-9
        ↓
┌──────────────────────────────────────────┐
│  Whisper (fine-tuned Thai PHQ-9)         │
│  → Transcribe เสียง → ข้อความไทย        │
│  → Keyword Match → คะแนน PHQ-9          │
└──────────────────────────────────────────┘
        +
┌──────────────────────────────────────────┐
│  AudioOnlyModel (best_model.pt)          │
│  → Wav2Vec2 สกัด features (T, 768)      │
│  → แบ่ง 16 segments → TransformerEncoder│
│  → 4-class: ไม่เลย/บางวัน/มากกว่าครึ่ง/แทบทุกวัน
│  → Weighted depression score (0-1)       │
└──────────────────────────────────────────┘
        ↓
  Combined Score = PHQ-9 (70%) + โทนเสียง (30%)
  ซึมเศร้า ถ้า combined_score > 0.37
        ↓
┌──────────────────────────────────────────┐
│  Gemini API (ใช้เฉพาะตรงนี้เท่านั้น)    │
│  → สร้างข้อความฮีลใจ personalized       │
│  → กิจกรรมแนะนำตามระดับความรุนแรง      │
│  → คำคมฮีลใจ                            │
└──────────────────────────────────────────┘
        ↓
  แสดงผล: ระดับ + คะแนน + คำแนะนำ + ฮีลใจ
```

---

## 🗂️ โครงสร้างโปรเจกต์

```
project_mlops/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── audio_model.py       # AudioOnlyModel (Wav2Vec2 + TransformerEncoder)
│   └── requirements.txt
├── frontend/
│   └── index.html           # Single-page application
├── models/                  # ← วาง model files ที่นี่
│   ├── best_model.pt        # AudioOnlyModel weights (4-class)
│   └── whisper/             # Whisper fine-tuned Thai PHQ-9
│       ├── model.safetensors
│       ├── config.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── vocab.json
│       ├── merges.txt
│       ├── normalizer.json
│       ├── added_tokens.json
│       ├── preprocessor_config.json
│       ├── processor_config.json
│       └── generation_config.json
├── Dockerfile
└── README.md
```

---

## 🚀 วิธี Deploy บน Google Cloud Run

```bash
# Build image
docker build --platform linux/amd64 -t gcr.io/<PROJECT_ID>/phq9-app .

# Push to GCR
docker push gcr.io/<PROJECT_ID>/phq9-app

# Deploy
gcloud run deploy phq9-app \
  --image gcr.io/<PROJECT_ID>/phq9-app \
  --platform managed \
  --region asia-southeast1 \
  --allow-unauthenticated \
  --memory 8Gi \
  --cpu 4 \
  --timeout 300 \
  --set-env-vars GEMINI_API_KEY=<YOUR_KEY>
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/questions` | ดึงคำถาม PHQ-9 ทั้ง 9 ข้อ |
| POST | `/api/transcribe` | audio → Whisper → keyword match → คะแนน |
| POST | `/api/classify-audio` | audio → Wav2Vec2 → AudioOnlyModel → 4-class |
| POST | `/api/analyze` | รวมคะแนน + Gemini healing message |
| GET | `/health` | Health check |

### POST /api/transcribe

```bash
curl -X POST https://<SERVICE_URL>/api/transcribe \
  -F "audio=@recording.webm"
```

Response:
```json
{
  "transcribed_text": "บางวันครับ",
  "mapped_answer": "บางวัน",
  "score": 1
}
```

### POST /api/classify-audio

```bash
curl -X POST https://<SERVICE_URL>/api/classify-audio \
  -F "audio=@recording.webm"
```

Response:
```json
{
  "predicted_label": "บางวัน",
  "probabilities": {
    "ไม่เลย": 0.12,
    "บางวัน": 0.55,
    "มากกว่าครึ่ง": 0.22,
    "แทบทุกวัน": 0.11
  },
  "depression_probability": 0.33
}
```

### POST /api/analyze

```bash
curl -X POST https://<SERVICE_URL>/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "phq_scores": [1,2,0,1,0,1,2,0,0],
    "audio_results": [{"depression_probability": 0.3, "predicted_label": "บางวัน"}]
  }'
```

---

## 🎯 การแปลผล PHQ-9

| คะแนน | ระดับ |
|-------|-------|
| 0–4 | ปกติ / อาการน้อยมาก |
| 5–9 | ซึมเศร้าระดับน้อย |
| 10–14 | ซึมเศร้าระดับปานกลาง |
| 15–19 | ซึมเศร้าระดับค่อนข้างรุนแรง |
| 20–27 | ซึมเศร้าระดับรุนแรง |

**Combined Score** = (PHQ-9 / 27) × 0.7 + Audio depression prob × 0.3
**Weighted Score** = p0×0 + p1×0.33 + p2×0.67 + p3×1.0

---

## 🛠️ Tech Stack

- **Frontend**: Vanilla HTML/CSS/JS
- **Backend**: FastAPI (Python 3.11)
- **Speech-to-Text**: Whisper fine-tuned Thai PHQ-9
- **Audio Classifier**: Wav2Vec2 + TransformerEncoder (4-class)
- **Healing Message**: Google Gemini 1.5 Flash Latest
- **Container**: Docker (2-stage build, linux/amd64)
- **Cloud**: Google Cloud Run (asia-southeast1, 8Gi RAM, 4 CPU)

---

## ☎️ สายด่วนสุขภาพจิต

**1323** — กรมสุขภาพจิต เปิด 24 ชั่วโมง ทุกวัน