# 🧠 PHQ-9 Depression Screening App

แอปพลิเคชันประเมินสุขภาพจิตด้วย PHQ-9 โดยใช้ AI วิเคราะห์เสียงและข้อความ Deploy บน Google Cloud Run

---

## 📋 ภาพรวม

```
ผู้ใช้อัดเสียงตอบ PHQ-9
        ↓
┌──────────────────────────────────────────┐
│  Whisper (model.safetensors)             │
│  → Transcribe เสียง → ข้อความไทย        │
│  → Map คำตอบ → คะแนน PHQ-9             │
└──────────────────────────────────────────┘
        +
┌──────────────────────────────────────────┐
│  AudioOnlyModel (best_model.pt)          │
│  → Wav2Vec2 classify โทนเสียง           │
│  → Depression probability (0-1)          │
└──────────────────────────────────────────┘
        ↓
  รวมคะแนน PHQ-9 (70%) + โทนเสียง (30%)
        ↓
┌──────────────────────────────────────────┐
│  Gemini API                              │
│  → สร้างข้อความฮีลใจ personalized       │
│  → คำแนะนำตามระดับความรุนแรง            │
└──────────────────────────────────────────┘
        ↓
  แสดงผล: ระดับ + คะแนน + คำแนะนำ + ฮีลใจ
```

---

## 🗂️ โครงสร้างโปรเจกต์

```
phq9-depression-app/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── audio_model.py       # AudioOnlyModel (Wav2Vec2) definition
│   └── requirements.txt
├── frontend/
│   └── index.html           # Single-page application
├── models/                  # ← วาง model files ที่นี่
│   ├── best_model.pt        # AudioOnlyModel weights
│   └── whisper/             # Whisper fine-tuned directory
│       ├── model.safetensors
│       ├── config.json
│       ├── tokenizer.json
│       └── ...
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## 🚀 วิธี Deploy

### Prerequisites

- Docker & Docker Compose
- Google Cloud SDK (`gcloud`)
- Gemini API Key จาก [Google AI Studio](https://aistudio.google.com)

### 1. วาง Model Files

```bash
# สร้างโฟลเดอร์ models
mkdir -p models/whisper

# วาง best_model.pt ใน models/
cp /path/to/best_model.pt models/

# วาง Whisper fine-tuned files ใน models/whisper/
cp /path/to/whisper/* models/whisper/
```

### 2. รันแบบ Local (Docker Compose)

```bash
# Copy .env
cp .env.example .env
# แก้ไข GEMINI_API_KEY ใน .env

# Build และรัน
docker-compose up --build

# เปิดเบราว์เซอร์
open http://localhost:8080
```

### 3. Deploy บน Google Cloud Run

```bash
# ตั้งค่า project
export PROJECT_ID="your-gcp-project-id"
export REGION="asia-southeast1"
export SERVICE_NAME="phq9-app"

# Login
gcloud auth login
gcloud config set project $PROJECT_ID

# Enable APIs
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com

# สร้าง Artifact Registry
gcloud artifacts repositories create phq9-repo \
  --repository-format=docker \
  --location=$REGION

# Build & Push image
gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT_ID/phq9-repo/$SERVICE_NAME

# Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
  --image $REGION-docker.pkg.dev/$PROJECT_ID/phq9-repo/$SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --set-env-vars GEMINI_API_KEY=your_key_here
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/questions` | ดึงคำถาม PHQ-9 ทั้ง 9 ข้อ |
| POST | `/api/transcribe` | อัปโหลด audio → transcribe → map คำตอบ |
| POST | `/api/classify-audio` | อัปโหลด audio → classify โทนเสียง |
| POST | `/api/analyze` | รวมคะแนน + วิเคราะห์ + สร้างข้อความ |
| GET | `/health` | Health check |

### POST /api/transcribe

```bash
curl -X POST http://localhost:8080/api/transcribe \
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
curl -X POST http://localhost:8080/api/classify-audio \
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
curl -X POST http://localhost:8080/api/analyze \
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

**Combined Score** = PHQ-9 normalized × 0.7 + Audio depression prob × 0.3

---

## 🛠️ Tech Stack

- **Frontend**: Vanilla HTML/CSS/JS (Single Page, ไม่ต้อง build)
- **Backend**: FastAPI (Python)
- **Speech-to-Text**: Whisper fine-tuned Thai (`model.safetensors`)
- **Audio Classifier**: Wav2Vec2-based (`best_model.pt`)
- **Healing Message**: Google Gemini 1.5 Flash
- **Container**: Docker
- **Cloud**: Google Cloud Run

---

## ☎️ สายด่วนสุขภาพจิต

**1323** — กรมสุขภาพจิต เปิด 24 ชั่วโมง ทุกวัน
