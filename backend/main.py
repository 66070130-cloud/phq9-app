import os
import tempfile
import logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import torchaudio
import numpy as np
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PHQ-9 Depression Screening API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Helper: always read key fresh from env ───────────────────────────────────
def get_gemini_key() -> str:
    return os.getenv("GEMINI_API_KEY", "").strip()


def get_gemini_model(model_name: str = "gemini-1.5-flash-latest"):
    key = get_gemini_key()
    if not key:
        raise ValueError("GEMINI_API_KEY is not set")
    genai.configure(api_key=key)
    return genai.GenerativeModel(model_name)


# ─── PHQ-9 ────────────────────────────────────────────────────────────────────
PHQ9_QUESTIONS = [
    "1. เบื่อ ทำอะไร ๆ ก็ไม่เพลิดเพลิน",
    "2. ไม่สบายใจ ซึมเศร้า หรือท้อแท้",
    "3. หลับยาก หรือหลับ ๆ ตื่น ๆ หรือหลับมากไป",
    "4. เหนื่อยง่าย หรือไม่ค่อยมีแรง",
    "5. เบื่ออาหาร หรือกินมากเกินไป",
    "6. รู้สึกไม่ดีกับตัวเอง คิดว่าตัวเองล้มเหลว หรือเป็นคนทำให้ตัวเอง หรือครอบครัวผิดหวัง",
    "7. สมาธิไม่ดีเวลาทำอะไร เช่น ดูโทรทัศน์ ฟังวิทยุ หรือทำงานที่ต้องใช้ความตั้งใจ",
    "8. พูดหรือทำอะไรช้าจนคนอื่นมองเห็น หรือกระสับกระส่ายจนท่านอยู่ไม่นิ่งเหมือนเคย",
    "9. คิดทำร้ายตนเอง หรือคิดว่าถ้าตาย ๆ ไปเสียคงจะดี",
]

SCORE_MAP = {"ไม่เลย": 0, "บางวัน": 1, "มากกว่าครึ่ง": 2, "แทบทุกวัน": 3}

KEYWORD_MAP = {
    "ไม่เลย": [
        "ไม่เลย", "ไม่มี", "เปล่าเลย", "เปล่า", "ปกติดี", "สบายดี",
        "ไม่รู้สึก", "ไม่มีอาการ", "ไม่เคย", "ไม่ได้เป็น", "ไม่ได้มี",
        "ไม่ค่อย", "ไม่ได้", "หายแล้ว", "โอเค", "โอเคนะ", "ดีนะ",
        "ไม่ได้รู้สึก", "ไม่มีเลย", "ปฏิเสธ", "ไม่", "เปล่านะ",
    ],
    "บางวัน": [
        "บางวัน", "บางครั้ง", "บางที", "นิดหน่อย", "นาน ๆ ครั้ง",
        "เล็กน้อย", "ไม่บ่อย", "นิดเดียว", "บางทีก็มี", "มีบ้าง",
        "นานๆครั้ง", "นานๆที", "บางโอกาส", "ไม่ถึงครึ่ง", "แค่นิดหน่อย",
        "หน่อยหนึ่ง", "เป็นบางครั้ง", "เป็นบางวัน", "ประมาณหนึ่ง",
        "พอมีบ้าง", "มีบางวัน",
    ],
    "มากกว่าครึ่ง": [
        "มากกว่าครึ่ง", "บ่อย", "บ่อยครั้ง", "หลายวัน", "ค่อนข้าง",
        "เกินครึ่ง", "พอสมควร", "บ่อยๆ", "ค่อนข้างบ่อย", "หลายครั้ง",
        "บ่อยมาก", "เยอะ", "มากพอสมควร", "เกินกว่าครึ่ง", "บ่อยพอสมควร",
        "มากกว่าปกติ", "เป็นส่วนใหญ่",
    ],
    "แทบทุกวัน": [
        "แทบทุกวัน", "ทุกวัน", "ตลอด", "ตลอดเวลา", "เกือบทุกวัน",
        "ทั้งวัน", "มากที่สุด", "ทุกๆวัน", "แทบตลอด", "ตลอดเลย",
        "ทุกวันเลย", "ไม่หยุด", "ตลอดทุกวัน", "แทบจะทุกวัน",
        "เกือบตลอด", "แทบทุกที", "ทุกที",
    ],
}

# ─── Model Loading ─────────────────────────────────────────────────────────────
whisper_model = None
audio_model = None
whisper_processor = None


def load_models():
    global whisper_model, audio_model, whisper_processor
    try:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        whisper_path = os.path.join(MODEL_DIR, "whisper")
        logger.info(f"Looking for Whisper at: {whisper_path}")
        logger.info(f"Files in MODEL_DIR: {os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else 'DIR NOT FOUND'}")
        if os.path.exists(whisper_path):
            logger.info(f"Whisper dir contents: {os.listdir(whisper_path)}")
            logger.info("Loading Whisper model...")
            from transformers import WhisperFeatureExtractor, WhisperTokenizer
            feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_path)
            tokenizer = WhisperTokenizer.from_pretrained(whisper_path, language="th", task="transcribe")
            whisper_processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_path).to(DEVICE)
            whisper_model.eval()
            logger.info("✅ Whisper loaded")
        else:
            logger.warning(f"Whisper model not found at {whisper_path}, using fallback")
    except Exception as e:
        logger.error(f"Failed to load Whisper: {e}")

    try:
        audio_model_path = os.path.join(MODEL_DIR, "best_model.pt")
        if os.path.exists(audio_model_path):
            logger.info("Loading AudioOnlyModel...")
            from audio_model import AudioOnlyModel
            audio_model = AudioOnlyModel()
            audio_model.load_state_dict(torch.load(audio_model_path, map_location=DEVICE))
            audio_model.to(DEVICE)
            audio_model.eval()
            logger.info("✅ AudioOnlyModel loaded")
        else:
            logger.warning(f"AudioOnlyModel not found at {audio_model_path}, using fallback")
    except Exception as e:
        logger.error(f"Failed to load AudioOnlyModel: {e}")

    key = get_gemini_key()
    if key:
        logger.info(f"✅ GEMINI_API_KEY found (ends: ...{key[-6:]})")
    else:
        logger.error("❌ GEMINI_API_KEY is EMPTY — healing message will use fallback")


@app.on_event("startup")
async def startup_event():
    load_models()


# ─── Helper Functions ──────────────────────────────────────────────────────────

def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe Thai audio — ใช้ Whisper เป็น primary, mock เป็น fallback
    (ไม่ใช้ Gemini สำหรับ transcribe)
    """
    if whisper_model is not None and whisper_processor is not None:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            f.write(audio_bytes)
            webm_path = f.name
        wav_path = webm_path.replace(".webm", ".wav")
        try:
            import subprocess
            subprocess.run(
                ["ffmpeg", "-y", "-i", webm_path, "-ar", "16000", "-ac", "1", wav_path],
                capture_output=True, check=True
            )
            tmp_path = wav_path
        except Exception as conv_e:
            logger.error(f"ffmpeg convert error in transcribe: {conv_e}")
            os.unlink(webm_path)
            return _mock_transcribe()
        try:
            waveform, sample_rate = torchaudio.load(tmp_path)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            waveform = waveform.mean(dim=0).numpy()
            inputs = whisper_processor(
                waveform, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(DEVICE)
            with torch.no_grad():
                predicted_ids = whisper_model.generate(inputs, language="th", task="transcribe")
            text = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            logger.info(f"Whisper transcribe: {text.strip()}")
            return text.strip()
        except Exception as e:
            logger.error(f"Whisper error: {e}")
        finally:
            if os.path.exists(tmp_path): os.unlink(tmp_path)
            if os.path.exists(webm_path): os.unlink(webm_path)

    return _mock_transcribe()


def _mock_transcribe():
    import random
    return random.choice(list(KEYWORD_MAP.keys()))


def classify_audio_tone(audio_bytes: bytes) -> dict:
    """Classify audio using AudioOnlyModel (4-class weighted score)."""
    if audio_model is None:
        return _mock_audio_classify()

    # บันทึก webm ก่อน แล้ว convert เป็น wav ด้วย ffmpeg
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(audio_bytes)
        webm_path = f.name
    tmp_path = webm_path.replace(".webm", ".wav")
    try:
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-i", webm_path, "-ar", "16000", "-ac", "1", tmp_path],
            capture_output=True, check=True
        )
    except Exception as e:
        logger.error(f"ffmpeg convert error: {e}")
        os.unlink(webm_path)
        return _mock_audio_classify()

    try:
        from transformers import Wav2Vec2Model, Wav2Vec2Processor

        w2v_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(DEVICE)
        w2v_model.eval()

        import soundfile as sf
        import librosa
        waveform_np, sample_rate = sf.read(tmp_path)
        if waveform_np.ndim > 1:
            waveform_np = waveform_np.mean(axis=1)
        if sample_rate != 16000:
            waveform_np = librosa.resample(waveform_np.astype(float), orig_sr=sample_rate, target_sr=16000)

        inputs = w2v_processor(waveform_np, sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)
        with torch.no_grad():
            features = w2v_model(inputs).last_hidden_state.squeeze(0).cpu()

        N_SEG = 16
        T = features.shape[0]
        if T >= N_SEG:
            seg_len = T // N_SEG
            segs = [features[i * seg_len:(i + 1) * seg_len].mean(dim=0) for i in range(N_SEG)]
        else:
            segs = list(features)
            while len(segs) < N_SEG:
                segs.append(torch.zeros(768))
        x = torch.stack(segs).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = audio_model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]  # shape (4,)

        # 4-class: 0=ไม่เลย, 1=บางวัน, 2=มากกว่าครึ่ง, 3=แทบทุกวัน
        # Weighted score: p0×0 + p1×0.33 + p2×0.67 + p3×1.0
        WEIGHTS = [0.0, 0.33, 0.67, 1.0]
        CLASS_LABELS = ["ไม่เลย", "บางวัน", "มากกว่าครึ่ง", "แทบทุกวัน"]
        depression_prob = float(sum(probs[i] * WEIGHTS[i] for i in range(4)))
        predicted_idx = int(np.argmax(probs))

        return {
            "predicted_label": CLASS_LABELS[predicted_idx],
            "probabilities": {CLASS_LABELS[i]: float(probs[i]) for i in range(4)},
            "depression_probability": depression_prob,
        }
    except Exception as e:
        logger.error(f"Audio classify error: {e}")
        return _mock_audio_classify()
    finally:
        if os.path.exists(tmp_path): os.unlink(tmp_path)
        if os.path.exists(webm_path): os.unlink(webm_path)


def _mock_audio_classify():
    probs = np.random.dirichlet(np.ones(4))
    WEIGHTS = [0.0, 0.33, 0.67, 1.0]
    CLASS_LABELS = ["ไม่เลย", "บางวัน", "มากกว่าครึ่ง", "แทบทุกวัน"]
    depression_prob = float(sum(probs[i] * WEIGHTS[i] for i in range(4)))
    predicted_idx = int(np.argmax(probs))
    return {
        "predicted_label": CLASS_LABELS[predicted_idx],
        "probabilities": {CLASS_LABELS[i]: float(probs[i]) for i in range(4)},
        "depression_probability": depression_prob,
    }


def map_text_to_answer(text: str) -> Optional[str]:
    """Map transcribed text to PHQ-9 answer — keyword match only (ไม่ใช้ Gemini)."""
    text_lower = text.strip().lower()
    for answer, keywords in KEYWORD_MAP.items():
        for kw in keywords:
            if kw in text_lower:
                return answer
    return None


def get_depression_level(phq_score: int) -> dict:
    if phq_score <= 4:
        return {"level": "minimal", "label": "ปกติ / อาการน้อยมาก", "color": "#4CAF50", "emoji": "🌱"}
    elif phq_score <= 9:
        return {"level": "mild", "label": "ซึมเศร้าระดับน้อย", "color": "#FFC107", "emoji": "🌤️"}
    elif phq_score <= 14:
        return {"level": "moderate", "label": "ซึมเศร้าระดับปานกลาง", "color": "#FF9800", "emoji": "⛅"}
    elif phq_score <= 19:
        return {"level": "moderately_severe", "label": "ซึมเศร้าระดับค่อนข้างรุนแรง", "color": "#FF5722", "emoji": "🌧️"}
    else:
        return {"level": "severe", "label": "ซึมเศร้าระดับรุนแรง", "color": "#F44336", "emoji": "⛈️"}


def get_recommendation(level: str) -> str:
    recs = {
        "minimal": (
            "✅ สุขภาพจิตของคุณอยู่ในเกณฑ์ดี\n\n"
            "วิธีดูแลตัวเองต่อไป:\n"
            "• นอนหลับให้ได้ 7-8 ชั่วโมงต่อคืน\n"
            "• ออกกำลังกายสม่ำเสมอ อย่างน้อย 30 นาที/วัน\n"
            "• ทำกิจกรรมที่ทำให้มีความสุข เช่น งานอดิเรก พบเพื่อน\n"
            "• ฝึก Mindfulness หรือการหายใจเพื่อผ่อนคลาย"
        ),
        "mild": (
            "🌤️ คุณมีสัญญาณเตือนเล็กน้อย ควรดูแลตัวเองมากขึ้น\n\n"
            "วิธีรักษาและดูแลตัวเอง:\n"
            "• พูดคุยกับคนที่ไว้วางใจเกี่ยวกับความรู้สึก\n"
            "• ออกกำลังกายสม่ำเสมอ ช่วยเพิ่ม Serotonin ในสมอง\n"
            "• ลดคาเฟอีนและแอลกอฮอล์\n"
            "• ลองทำ CBT (Cognitive Behavioral Therapy) ด้วยตัวเองผ่านแอป เช่น Woebot\n"
            "• ถ้าอาการไม่ดีขึ้นใน 2 สัปดาห์ ควรปรึกษานักจิตวิทยา"
        ),
        "moderate": (
            "⛅ ควรพบผู้เชี่ยวชาญโดยเร็ว\n\n"
            "วิธีรักษาที่แนะนำ:\n"
            "• 🏥 พบจิตแพทย์หรือนักจิตวิทยา เพื่อรับการประเมินอย่างละเอียด\n"
            "• 💊 อาจได้รับยาต้านซึมเศร้า (Antidepressants) เช่น SSRIs ซึ่งได้ผลดีมาก\n"
            "• 🧠 จิตบำบัด CBT หรือ Interpersonal Therapy (IPT)\n"
            "• 👥 เข้ากลุ่มสนับสนุน (Support Group)\n"
            "• 📞 กรมสุขภาพจิต 1323 ให้คำปรึกษาฟรี 24 ชั่วโมง"
        ),
        "moderately_severe": (
            "🌧️ อาการระดับนี้ต้องการการรักษาจากผู้เชี่ยวชาญ\n\n"
            "วิธีรักษาที่จำเป็น:\n"
            "• 🏥 พบจิตแพทย์โดยเร็วที่สุด ไม่ควรรอ\n"
            "• 💊 การรักษาด้วยยาร่วมกับจิตบำบัด มีประสิทธิภาพสูงสุด\n"
            "• 🧠 Psychotherapy เช่น CBT, DBT (Dialectical Behavior Therapy)\n"
            "• 👨‍👩‍👧 แจ้งครอบครัวหรือคนใกล้ชิดให้รับรู้และช่วยดูแล\n"
            "• 📞 สายด่วนสุขภาพจิต 1323 ตลอด 24 ชั่วโมง\n"
            "• 🏨 โรงพยาบาลรัฐที่มีแผนกจิตเวช เช่น รพ.ศรีธัญญา, รพ.สมเด็จเจ้าพระยา"
        ),
        "severe": (
            "⛈️ ต้องการความช่วยเหลือด่วน กรุณาติดต่อผู้เชี่ยวชาญทันที\n\n"
            "ขั้นตอนเร่งด่วน:\n"
            "• 📞 โทร 1323 สายด่วนสุขภาพจิต ทันที (24 ชั่วโมง)\n"
            "• 🚨 ไปห้องฉุกเฉินโรงพยาบาลใกล้บ้านได้เลย\n"
            "• 🏥 รพ.ศรีธัญญา: 02-528-7800 | รพ.สมเด็จเจ้าพระยา: 02-442-2200\n"
            "• 👤 บอกคนที่ไว้วางใจให้อยู่เป็นเพื่อนคุณในตอนนี้\n\n"
            "การรักษา:\n"
            "• การรักษาในโรงพยาบาล (Inpatient) เพื่อความปลอดภัย\n"
            "• ยาและจิตบำบัดอย่างเข้มข้น\n"
            "• คุณสำคัญมาก ชีวิตของคุณมีคุณค่า 💜"
        ),
    }
    return recs.get(level, "")


async def generate_healing_message(phq_score: int, depression_prob: float, level_info: dict) -> str:
    """ใช้ Gemini สร้างข้อความฮีลใจ + กิจกรรม + คำคม (Gemini ใช้เฉพาะตรงนี้เท่านั้น)"""
    key = get_gemini_key()
    if not key:
        return _fallback_healing_message(level_info["level"])

    try:
        model = get_gemini_model()
        level_context = {
            "minimal":           "สุขภาพจิตดี ยังไม่มีอาการ",
            "mild":              "มีความเครียดหรืออารมณ์ขึ้นลงเล็กน้อย",
            "moderate":          "รู้สึกหนักใจบ่อยครั้ง ต้องการการดูแลเพิ่มขึ้น",
            "moderately_severe": "รู้สึกหนักใจมาก ต้องการความช่วยเหลือจากผู้เชี่ยวชาญ",
            "severe":            "ต้องการความช่วยเหลือด่วน",
        }.get(level_info["level"], "")

        prompt = f"""คุณคือเพื่อนที่อบอุ่น เข้าใจ และเป็นนักจิตวิทยาที่เชี่ยวชาญด้านการฮีลใจ

ผู้ใช้เพิ่งทำแบบประเมินความรู้สึก PHQ-9:
- คะแนน: {phq_score}/27
- สถานะ: {level_info['label']} — {level_context}

กรุณาเขียนเนื้อหา 3 ส่วน โดยใช้ภาษาอ่อนโยน อบอุ่น เป็นกันเอง เชิงบวก 100% ห้ามพูดถึงแง่ลบ:

💜 ส่วนที่ 1 — ข้อความฮีลใจ (3-4 ประโยค)
- เริ่มด้วยการชื่นชมที่เขากล้าดูแลตัวเอง
- รับรู้ความรู้สึกอย่างอ่อนโยน ไม่ตัดสิน
- เน้นว่าเขาไม่ได้อยู่คนเดียว และมีความหวังเสมอ
- ให้กำลังใจที่จริงใจ อุ่นใจ

🌱 ส่วนที่ 2 — กิจกรรมฮีลใจ 4-5 ข้อ (เหมาะกับระดับ {level_info['label']})
- เป็นกิจกรรมที่ทำได้จริง ง่าย สนุก ไม่กดดัน
- เน้นกิจกรรมที่ช่วยให้รู้สึกดีขึ้น เช่น ศิลปะ ดนตรี ธรรมชาติ การเคลื่อนไหว การเชื่อมต่อกับผู้คน
- ถ้าระดับปานกลางขึ้นไป ให้รวมการพูดคุยกับผู้เชี่ยวชาญเป็นหนึ่งในกิจกรรม แต่เขียนในแง่บวกว่าเป็นการดูแลตัวเอง ไม่ใช่การรักษาโรค
- ใส่ emoji หน้าแต่ละข้อ

✨ ส่วนที่ 3 — คำคมฮีลใจ 1 ประโยค
- สั้น กินใจ อบอุ่น ให้ความรู้สึกโอบกอด
- เหมาะกับสถานะของเขา

ห้ามใช้คำเหล่านี้: โรคซึมเศร้า, รักษาโรค, ผู้ป่วย, อาการ, วินิจฉัย
ใช้แทนด้วย: ความรู้สึก, การดูแลตัวเอง, การเติมพลัง, การฟื้นฟูจิตใจ"""

        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return _fallback_healing_message(level_info["level"])


def _fallback_healing_message(level: str) -> str:
    messages = {
        "minimal": "คุณดูแลตัวเองได้ดีมากเลยนะ 💚 การที่คุณใส่ใจสุขภาพจิตตัวเองแสดงว่าคุณรักตัวเองอย่างแท้จริง ขอให้คุณมีความสุขและพลังงานดีๆ ทุกวันนะ\n\n✨ คำคม: \"ความสุขไม่ได้อยู่ที่จุดหมาย แต่อยู่ที่การเดินทางในทุกๆ วัน\"",
        "mild": "ขอบคุณที่กล้าดูแลสุขภาพจิตของตัวเองนะ 🌤️ ทุกคนต่างมีวันที่รู้สึกหนักใจบ้าง และการรับรู้ความรู้สึกตัวเองคือก้าวแรกที่สำคัญที่สุดแล้ว อย่าลืมดูแลตัวเองด้วยความอ่อนโยนเหมือนที่คุณดูแลคนที่รักนะ\n\n✨ คำคม: \"แม้เมฆจะมืดครึ้ม แสงแดดยังคงอยู่เหนือมันเสมอ\"",
        "moderate": "ขอบคุณที่ไว้ใจบอกความรู้สึกเหล่านี้ 🌧️ มันคงหนักมากที่แบกรับมาคนเดียว แต่คุณไม่จำเป็นต้องสู้คนเดียวเลย การขอความช่วยเหลือคือความกล้าหาญ ไม่ใช่ความอ่อนแอ มีคนพร้อมเดินเคียงข้างคุณเสมอ\n\n✨ คำคม: \"ความกล้าไม่ใช่การไม่กลัว แต่คือการลุกขึ้นอีกครั้งแม้จะกลัว\"",
        "moderately_severe": "ใจแข็งมากที่ผ่านมาได้ถึงทุกวันนี้ 💙 ความรู้สึกที่แบกอยู่นั้นหนักมาก แต่คุณไม่ต้องเดินคนเดียว มีผู้เชี่ยวชาญที่พร้อมช่วยเหลือ และการก้าวออกมาขอความช่วยเหลือวันนี้คือสิ่งที่กล้าหาญที่สุด\n\n✨ คำคม: \"หลังคืนที่มืดที่สุด รุ่งอรุณย่อมมาเสมอ — Victor Hugo\"",
        "severe": "คุณสำคัญมาก และชีวิตของคุณมีคุณค่าอย่างที่สุด 💜 ขอบคุณที่ยังอยู่และยังสู้ต่อ ขอให้ติดต่อผู้เชี่ยวชาญโดยเร็วที่สุด โทร 1323 ได้เลยตอนนี้ คุณไม่ต้องเผชิญสิ่งนี้คนเดียว\n\n✨ คำคม: \"แม้เพียงหนึ่งวันที่ยังมีลมหายใจ ก็คือความหวังที่ไม่มีวันหมด\"",
    }
    return messages.get(level, messages["mild"])


# ─── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/questions")
async def get_questions():
    return {"questions": PHQ9_QUESTIONS, "answers": list(SCORE_MAP.keys())}


@app.post("/api/transcribe")
async def transcribe_endpoint(audio: UploadFile = File(...)):
    try:
        audio_bytes = await audio.read()
        transcribed_text = transcribe_audio(audio_bytes)
        mapped_answer = map_text_to_answer(transcribed_text)
        return {
            "transcribed_text": transcribed_text,
            "mapped_answer": mapped_answer,
            "score": SCORE_MAP.get(mapped_answer, 0) if mapped_answer else None,
        }
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/classify-audio")
async def classify_audio_endpoint(audio: UploadFile = File(...)):
    try:
        audio_bytes = await audio.read()
        result = classify_audio_tone(audio_bytes)
        return result
    except Exception as e:
        logger.error(f"Audio classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class AnalyzeRequest(BaseModel):
    phq_scores: list[int]
    audio_results: list[dict]


@app.post("/api/analyze")
async def analyze_endpoint(req: AnalyzeRequest):
    try:
        phq_total = sum(req.phq_scores)

        if req.audio_results:
            avg_depression_prob = np.mean([r.get("depression_probability", 0) for r in req.audio_results])
        else:
            avg_depression_prob = 0.0

        phq_normalized = phq_total / 27.0
        combined_score = (phq_normalized * 0.7) + (avg_depression_prob * 0.3)

        level_info = get_depression_level(phq_total)
        recommendation = get_recommendation(level_info["level"])
        healing_message = await generate_healing_message(phq_total, avg_depression_prob, level_info)

        return {
            "phq_total": phq_total,
            "phq_normalized": phq_normalized,
            "avg_depression_prob_from_voice": float(avg_depression_prob),
            "combined_score": float(combined_score),
            "level_info": level_info,
            "recommendation": recommendation,
            "healing_message": healing_message,
            "is_depressed": bool(combined_score > 0.37),
        }
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    key = get_gemini_key()
    return {
        "status": "ok",
        "whisper_loaded": whisper_model is not None,
        "audio_model_loaded": audio_model is not None,
        "gemini_configured": bool(key),
    }


# Serve frontend static files
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(os.path.join(frontend_path, "dist")):
    app.mount("/", StaticFiles(directory=os.path.join(frontend_path, "dist"), html=True), name="static")
elif os.path.exists(os.path.join(frontend_path, "index.html")):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
