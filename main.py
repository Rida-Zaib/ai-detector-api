from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os
import base64
import sqlite3
import re
import math
import io
from datetime import datetime
from docx import Document
import pdfplumber
 
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
load_dotenv()
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
 
DB_PATH = "detector_history.db"
 
 
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS checks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text_preview TEXT,
            full_text TEXT,
            ai_score REAL,
            verdict TEXT,
            paste_percentage REAL,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()
 
 
init_db()
 
 
def save_check(text, score, verdict, paste_percentage=0.0):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO checks (text_preview, full_text, ai_score, verdict, paste_percentage, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (text[:80], text, score, verdict, paste_percentage, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()
 
 
# --------------------------------------------------------------------------
# Plagiarism check -- compares against previously submitted work stored in
# the database. This is intentionally scoped: it catches students copying
# each other or resubmitting past work, and grows more useful as more
# submissions accumulate. It does NOT check against the open web -- that
# would require a paid plagiarism API (Copyleaks, Turnitin) or a web-search
# integration, which is a reasonable "next step" to mention in a portfolio.
# --------------------------------------------------------------------------
def check_plagiarism(text: str, exclude_recent_seconds: int = 2):
    words = set(re.findall(r"\b\w+\b", text.lower()))
    if not words:
        return 0.0, []
 
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT id, full_text, created_at FROM checks ORDER BY id DESC LIMIT 200").fetchall()
    conn.close()
 
    best_score = 0.0
    matches = []
    for row_id, other_text, created_at in rows:
        if not other_text:
            continue
        other_words = set(re.findall(r"\b\w+\b", other_text.lower()))
        if not other_words:
            continue
        overlap = len(words & other_words) / len(words | other_words)
        if overlap > 0.35 and overlap < 0.999:  # 0.999+ likely the same exact resubmission check, still flag but separately
            matches.append({"submission_id": row_id, "similarity_percent": round(overlap * 100, 1)})
        best_score = max(best_score, overlap)
 
    return round(best_score, 3), sorted(matches, key=lambda m: -m["similarity_percent"])[:5]
 
 
def get_plagiarism_label(score: float):
    if score < 0.20:
        return "Original"
    elif score < 0.40:
        return "Minor Overlap"
    elif score < 0.65:
        return "Partial Overlap Detected"
    else:
        return "Substantial Overlap Detected"
 
 
# --------------------------------------------------------------------------
# Professional verdict labels (5-tier, replacing the casual "likely_ai" style)
# --------------------------------------------------------------------------
def get_verdict_label(score: float):
    if score < 0.20:
        return "Human-Authored"
    elif score < 0.40:
        return "Likely Human-Authored"
    elif score < 0.60:
        return "Indeterminate"
    elif score < 0.80:
        return "Likely AI-Generated"
    else:
        return "AI-Generated"
 
 
# --------------------------------------------------------------------------
# File extraction functions
# --------------------------------------------------------------------------
def extract_text_from_image(image_bytes: bytes) -> str:
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    response = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract and return ONLY the text visible in this image, nothing else."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        temperature=0,
        max_completion_tokens=1024
    )
    return response.choices[0].message.content.strip()
 
 
def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)
 
 
def extract_text_from_pdf(file_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)
 
 
# --------------------------------------------------------------------------
# Core detection logic (enhanced with extra signals for higher sensitivity)
# --------------------------------------------------------------------------
def detect_ai_text(text: str):
    words = re.findall(r"\b\w+\b", text.lower())
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s]
 
    reasons = []
 
    # Signal 1: Sentence length uniformity
    if len(sentences) >= 2:
        lengths = [len(re.findall(r"\b\w+\b", s)) for s in sentences]
        mean_len = sum(lengths) / len(lengths)
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        std_dev = math.sqrt(variance)
        uniformity_score = 1 / (1 + std_dev)
    else:
        uniformity_score = 0.5
 
    if uniformity_score > 0.55:
        reasons.append("Sentence length shows low variance, consistent with algorithmically generated text.")
 
    # Signal 2: Vocabulary richness
    if words:
        richness = len(set(words)) / len(words)
        richness_score = 1 - richness
    else:
        richness_score = 0.5
 
    if richness_score > 0.45:
        reasons.append("Vocabulary diversity is below the expected range for human-authored text of this length.")
 
    # Signal 3: Formal AI-style transition markers
    ai_markers = [
        "furthermore", "moreover", "in conclusion", "it is important to note",
        "overall", "in summary", "additionally", "on the other hand",
        "it is worth noting", "in essence", "notably", "consequently"
    ]
    marker_hits = sum(text.lower().count(m) for m in ai_markers)
    marker_score = min(marker_hits / 3, 1.0)
 
    if marker_hits > 0:
        reasons.append(f"Contains {marker_hits} formal transitional phrase(s) associated with AI-generated writing.")
 
    # Signal 4 (NEW): Contraction usage -- humans use contractions far more often
    contractions = re.findall(r"\b\w+'(?:t|re|ve|ll|d|s|m)\b", text.lower())
    contraction_ratio = len(contractions) / max(len(sentences), 1)
    contraction_score = max(0.0, 1 - (contraction_ratio * 2))  # fewer contractions -> higher AI signal
 
    if contraction_ratio < 0.1 and len(sentences) >= 3:
        reasons.append("Near-total absence of contractions, atypical for informal human writing.")
 
    # Signal 5 (NEW): Repetitive sentence openers (AI often reuses structural openers)
    openers = [s.strip().split()[0].lower() for s in sentences if s.strip()]
    if openers:
        opener_repetition = 1 - (len(set(openers)) / len(openers))
    else:
        opener_repetition = 0.0
 
    if opener_repetition > 0.3 and len(sentences) >= 4:
        reasons.append("Multiple sentences share the same opening word, a mild structural AI indicator.")
 
    # Weighted combination -- more signals now contribute for finer-grained sensitivity
    pattern_score = (
        0.28 * uniformity_score +
        0.28 * richness_score +
        0.20 * marker_score +
        0.14 * contraction_score +
        0.10 * opener_repetition
    )
    pattern_score = round(min(max(pattern_score, 0.0), 1.0), 3)
 
    if not reasons:
        reasons.append("No significant surface-level AI-writing indicators detected.")
 
    return pattern_score, reasons
 
 
def ai_judge_check(text: str):
    """
    Asks the LLM to reason about depth/specificity rather than just surface
    style -- more robust against text told to 'sound more human', though
    still not a foolproof guarantee (no detector fully is).
    """
    judge_prompt = f"""You are a meticulous document examiner evaluating whether a piece of writing was authored by a human or generated by an AI system, including AI text that has been deliberately revised to sound more natural or "humanized."
 
Evaluate carefully and look past surface style (sentence length variety, casual tone, contractions) since those are trivially imitated. Instead, examine:
- Specific, verifiable, idiosyncratic personal details versus generic, plausible-sounding statements
- Genuine logical progression and argumentation versus smooth-but-shallow reasoning
- Natural imperfections: tangents, mild inconsistencies, strong opinions, versus balanced "on one hand / on the other hand" hedging
- Whether claims carry the oddly specific texture of a real recollection versus vague-but-fluent generality
- Idea density and originality versus safe, expected, consensus-style phrasing
 
Be rigorous and err toward flagging subtle signals rather than dismissing them, since sophisticated AI text is often very fluent.
 
Text to evaluate:
\"\"\"{text}\"\"\"
 
Respond in this exact format, nothing else:
SCORE: <a number 0-100, where 100 = very likely AI-generated even if humanized>
REASON: <one or two sentences explaining the strongest signal found>
"""
 
    response = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0,
        max_completion_tokens=200
    )
 
    output = response.choices[0].message.content.strip()
 
    score = 50.0
    reason = "Semantic analysis could not be parsed from model response."
    try:
        for line in output.split("\n"):
            if line.startswith("SCORE:"):
                score = float(line.replace("SCORE:", "").strip())
            if line.startswith("REASON:"):
                reason = line.replace("REASON:", "").strip()
    except Exception:
        pass
 
    return round(score / 100, 3), reason
 
 
def calculate_grade(ai_score: float):
    if ai_score < 0.2:
        return "A"
    elif ai_score < 0.4:
        return "B"
    elif ai_score < 0.6:
        return "C"
    elif ai_score < 0.8:
        return "D"
    else:
        return "F"
 
 
def check_input_behavior(paste_percentage: float):
    if paste_percentage >= 80:
        return "High Paste Usage", f"{paste_percentage}% of the submission was pasted rather than typed. Recommend manual review."
    elif paste_percentage >= 30:
        return "Mixed Input", f"{paste_percentage}% of the submission was pasted. Partial paste activity detected."
    else:
        return "Primarily Typed", f"Only {paste_percentage}% of the submission was pasted. Input pattern is consistent with manual typing."
 
 
class TextDetectionRequest(BaseModel):
    text: str
 
 
@app.post("/detect/text")
def detect_text(payload: TextDetectionRequest):
    pattern_score, reasons = detect_ai_text(payload.text)
    verdict = get_verdict_label(pattern_score)
    save_check(payload.text, pattern_score, verdict)
    return {"ai_likelihood_score": pattern_score, "verdict": verdict, "reasons": reasons}
 
 
@app.post("/detect/report")
def detect_report(
    text: str = Form(None),
    file: UploadFile = File(None),
    paste_percentage: float = Form(0.0)
):
    if file is not None:
        file_bytes = file.file.read()
        filename = (file.filename or "").lower()
 
        if filename.endswith(".pdf"):
            extracted_text = extract_text_from_pdf(file_bytes)
            source = "pdf_document"
        elif filename.endswith(".docx"):
            extracted_text = extract_text_from_docx(file_bytes)
            source = "word_document"
        elif filename.endswith((".png", ".jpg", ".jpeg", ".webp")):
            extracted_text = extract_text_from_image(file_bytes)
            source = "image"
        else:
            return {"error": "Unsupported file type. Please upload a .docx, .pdf, or image file (.png/.jpg)."}
 
    elif text is not None:
        extracted_text = text
        source = "text"
    else:
        return {"error": "Please provide either 'text' or upload a supported file."}
 
    if len(extracted_text.strip()) < 20:
        return {"error": "Not enough text found/extracted to analyze (minimum 20 characters)."}
 
    pattern_score, reasons = detect_ai_text(extracted_text)
    judge_score, judge_reason = ai_judge_check(extracted_text)
 
    # Semantic judge weighted higher -- more robust against paraphrasing/humanizing
    final_score = round((0.3 * pattern_score) + (0.7 * judge_score), 3)
    verdict = get_verdict_label(final_score)
 
    reasons.append(f"Semantic analysis: {judge_reason}")
    grade = calculate_grade(final_score)
 
    # Plagiarism check -- runs BEFORE saving this submission, so it only
    # compares against work submitted prior to this one
    plag_score, plag_matches = check_plagiarism(extracted_text)
    plag_label = get_plagiarism_label(plag_score)
 
    save_check(extracted_text, final_score, verdict, paste_percentage)
 
    behavior_flag, behavior_message = check_input_behavior(paste_percentage)
 
    return {
        "source": source,
        "extracted_text_preview": extracted_text[:200],
        "ai_likelihood_percent": round(final_score * 100, 1),
        "human_likelihood_percent": round((1 - final_score) * 100, 1),
        "verdict": verdict,
        "reasons": reasons,
        "grade": grade,
        "plagiarism": {
            "similarity_percent": round(plag_score * 100, 1),
            "label": plag_label,
            "matches": plag_matches
        },
        "input_behavior": {
            "paste_percentage": paste_percentage,
            "flag": behavior_flag,
            "message": behavior_message
        }
    }
 
 
@app.get("/")
def root():
    return {"message": "Server chal raha hai!"}
