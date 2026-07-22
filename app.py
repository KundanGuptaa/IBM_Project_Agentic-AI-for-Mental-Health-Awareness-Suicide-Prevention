# =============================================================================
#  MindGuard AI – Mental Health Awareness & Suicide Prevention Agent
#  Powered by IBM watsonx.ai Studio | IBM Granite Models | Agentic AI
# =============================================================================
#
#  Architecture:
#    • 5 Specialized AI Agents
#    • Master Orchestrator Agent
#    • Lightweight RAG System
#    • IBM Granite LLM Integration
#    • Flask Single-Page Application (Bootstrap 5)
#
#  Suitable for:
#    IBM SkillsBuild | watsonx.ai Studio Demos | Hackathons | Academic Projects
#
#  Author  : MindGuard AI Team
#  Platform: IBM watsonx.ai Studio
#  Model   : IBM Granite (ibm/granite-3-3-8b-instruct)
# =============================================================================

# --------------------------------------------------------------------------- #
#  Standard Library Imports
# --------------------------------------------------------------------------- #
import os
import re
import math
import json
import datetime
import textwrap

# --------------------------------------------------------------------------- #
#  Third-Party Imports
# --------------------------------------------------------------------------- #
# Load .env file automatically so credentials don't need to be exported manually
# Graceful import – app works even if python-dotenv is not installed
try:
    from dotenv import load_dotenv
    load_dotenv()                    # reads .env from the current working directory
    print("[INFO] .env file loaded via python-dotenv.")
except ImportError:
    # python-dotenv not installed – fall back to manual .env parsing
    import re as _re
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(_env_path):
        with open(_env_path, encoding="utf-8") as _ef:
            for _line in _ef:
                _line = _line.strip()
                if not _line or _line.startswith("#") or "=" not in _line:
                    continue
                _k, _, _v = _line.partition("=")
                _k = _k.strip()
                _v = _v.strip().strip('"').strip("'")
                if _k and _k not in os.environ:   # don't overwrite existing env vars
                    os.environ[_k] = _v
        print("[INFO] .env file loaded via built-in fallback parser.")
    else:
        print("[INFO] No .env file found – using system environment variables.")

from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename

# IBM watsonx.ai SDK
try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    WATSONX_AVAILABLE = True
except ImportError:
    WATSONX_AVAILABLE = False
    print("[WARN] ibm-watsonx-ai SDK not installed. Running in demo mode.")

# PDF text extraction (optional dependency)
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("[WARN] PyPDF2 not installed. PDF uploads will be skipped.")

# =============================================================================
#  Flask Application Setup
# =============================================================================
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB upload limit
app.config["UPLOAD_FOLDER"]      = "/tmp/mindguard_uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# =============================================================================
#  IBM watsonx.ai Credentials  ← READ FROM ENVIRONMENT VARIABLES
# =============================================================================
WATSONX_API_KEY    = os.getenv("WATSONX_API_KEY",    "your-api-key-here")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "your-project-id-here")
WATSONX_URL        = os.getenv("WATSONX_URL",        "https://us-south.ml.cloud.ibm.com")

# IBM Granite Model identifier
# Preferred order – first one available in the project will be used
GRANITE_MODEL_CANDIDATES = [
    "ibm/granite-4-h-small",           # newest Granite (2025)
    "ibm/granite-3-1-8b-base",         # Granite 3.1 8B base
    "ibm/granite-8b-code-instruct",     # code-capable Granite
    "ibm/granite-3-3-8b-instruct",      # original target (may not exist in all regions)
]
GRANITE_MODEL_ID = GRANITE_MODEL_CANDIDATES[0]  # overridden at init if needed

# =============================================================================
#  IBM Granite Model Initialization  ← watsonx.ai Integration Point
# =============================================================================
granite_model = None   # populated on first call (lazy init)

def _init_granite_model():
    """
    Lazily initialize the IBM Granite model via watsonx.ai SDK.
    Tries each model in GRANITE_MODEL_CANDIDATES until one succeeds.
    Falls back to demo mode if credentials are missing or SDK is unavailable.
    """
    global granite_model, GRANITE_MODEL_ID
    if granite_model is not None:
        return True                     # already initialized

    if not WATSONX_AVAILABLE:
        return False

    if WATSONX_API_KEY == "your-api-key-here":
        print("[WARN] IBM watsonx.ai credentials not configured. Demo mode active.")
        return False

    # ── IBM watsonx.ai Credentials ──────────────────────────────────────────
    credentials = Credentials(
        url    = WATSONX_URL,
        api_key= WATSONX_API_KEY,
    )

    # ── Try each Granite candidate until one initialises successfully ────────
    last_error = None
    for candidate in GRANITE_MODEL_CANDIDATES:
        try:
            model = ModelInference(
                model_id   = candidate,
                credentials= credentials,
                project_id = WATSONX_PROJECT_ID,
                params     = {
                    GenParams.MAX_NEW_TOKENS    : 800,
                    GenParams.TEMPERATURE       : 0.7,
                    GenParams.TOP_P             : 0.9,
                    GenParams.REPETITION_PENALTY: 1.1,
                },
            )
            # Quick smoke-test to confirm the model actually responds
            model.generate_text(prompt="Hi")
            granite_model    = model
            GRANITE_MODEL_ID = candidate
            print(f"[INFO] IBM Granite model '{GRANITE_MODEL_ID}' initialized successfully.")
            return True
        except Exception as exc:
            last_error = exc
            print(f"[INFO] Model '{candidate}' unavailable, trying next... ({str(exc)[:80]})")

    print(f"[ERROR] All Granite model candidates failed. Last error: {last_error}")
    return False

# =============================================================================
#  Core watsonx.ai Helper Functions
# =============================================================================

def generate_response(prompt: str, max_tokens: int = 600) -> str:
    """
    Send a prompt to IBM Granite and return the generated text.
    Falls back to a rule-based demo response if watsonx.ai is unavailable.

    IBM watsonx.ai Integration Point ← ModelInference.chat() (v1/text/chat)
    Uses the newer chat API to avoid deprecation warnings.
    """
    if _init_granite_model() and granite_model is not None:
        try:
            # Use chat() API (preferred over deprecated generate_text)
            response = granite_model.chat(
                messages=[{"role": "user", "content": prompt}],
                params={GenParams.MAX_NEW_TOKENS: max_tokens},
            )
            result = response["choices"][0]["message"]["content"]
            return result.strip() if result else _demo_fallback(prompt)
        except Exception as exc:
            print(f"[ERROR] generate_response: {exc}")
            return _demo_fallback(prompt)
    return _demo_fallback(prompt)


def detect_risk(text: str) -> dict:
    """
    Use IBM Granite to analyse text for mental-health risk.
    Returns a structured dict: { risk_level, risk_score, explanation, next_steps }

    IBM watsonx.ai Integration Point ← distress reasoning via Granite
    """
    prompt = f"""You are a mental health risk assessment AI.
Analyse the following message for distress indicators such as hopelessness,
isolation, persistent sadness, severe stress, or emotional exhaustion.

Message: "{text}"

Respond ONLY in valid JSON with these exact keys:
{{
  "risk_level": "Low Risk" | "Moderate Risk" | "High Risk",
  "risk_score": <integer 0-100>,
  "explanation": "<one or two sentences>",
  "next_steps": "<brief recommendation>"
}}"""

    raw = generate_response(prompt, max_tokens=300)
    try:
        # Extract JSON from the model response
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass

    # Fallback: simple keyword scoring
    return _keyword_risk_score(text)


def generate_wellness_plan(mood: str, stress_level: str, emotional_state: str) -> str:
    """
    Use IBM Granite to create a personalised wellness plan.

    IBM watsonx.ai Integration Point ← personalised plan generation
    """
    prompt = f"""You are a professional wellness coach AI powered by IBM Granite.
Create a personalised daily wellness plan for someone with the following profile:
- Current Mood: {mood}
- Stress Level: {stress_level}
- Emotional State: {emotional_state}

Include:
1. Morning routine (5 min breathing or meditation)
2. Midday check-in activity
3. Evening journaling prompt
4. Sleep hygiene tip
5. One positive affirmation

Keep each section concise and actionable."""

    return generate_response(prompt, max_tokens=600)


def retrieve_context(query: str, top_k: int = 3) -> str:
    """
    Lightweight RAG retrieval: search the in-memory document store
    and return the top_k most relevant text chunks.
    """
    if not RAG_STORE:
        return ""

    query_words = set(query.lower().split())
    scored = []
    for doc in RAG_STORE:
        chunk_words = set(doc["chunk"].lower().split())
        overlap     = len(query_words & chunk_words)
        if overlap > 0:
            scored.append((overlap, doc["chunk"]))

    scored.sort(reverse=True, key=lambda x: x[0])
    top_chunks = [chunk for _, chunk in scored[:top_k]]
    return "\n---\n".join(top_chunks) if top_chunks else ""


def orchestrate_agents(user_input: str) -> dict:
    """
    Master Orchestrator Agent.
    Decides which specialized agent(s) to invoke, aggregates results,
    and returns a unified response for the UI.

    Orchestration Logic:
      • Crisis/suicide keywords  → Distress + Support Connector
      • Emotion/feeling keywords → Emotional Support + Wellness
      • Information questions    → Awareness Agent
      • Wellness/advice request  → Wellness Agent
      • Default                  → Emotional Support
    """
    lower = user_input.lower()

    # ── Step 1: Routing Decision ─────────────────────────────────────────────
    crisis_keywords   = ["suicide", "kill myself", "end my life", "want to die",
                         "no reason to live", "self harm", "hurt myself"]
    distress_keywords = ["hopeless", "worthless", "depressed", "overwhelmed",
                         "can't cope", "breaking down", "falling apart", "exhausted",
                         "give up", "lonely", "alone", "sad", "anxious", "panic"]
    wellness_keywords = ["relax", "meditation", "breathing", "exercise", "sleep",
                         "tips", "advice", "help me", "coping", "strategy", "plan",
                         "recommend", "wellness", "mindfulness", "journal"]
    info_keywords     = ["what is", "what are", "how does", "explain", "define",
                         "tell me about", "meaning of", "signs of", "symptoms"]

    is_crisis   = any(kw in lower for kw in crisis_keywords)
    is_distress = any(kw in lower for kw in distress_keywords)
    is_wellness = any(kw in lower for kw in wellness_keywords)
    is_info     = any(kw in lower for kw in info_keywords)

    # ── Step 2: Agent Selection & Invocation ─────────────────────────────────
    agents_activated = []
    primary_response = ""
    risk_data        = {}
    wellness_plan    = ""
    resources        = {}

    if is_crisis:
        agents_activated = ["Distress Detection Agent", "Human Support Connector Agent"]
        risk_data        = distress_detection_agent(user_input)
        resources        = support_connector_agent(risk_level="High Risk")
        primary_response = emotional_support_agent(user_input)
        reason           = "Crisis indicators detected >> activating Distress Detection + Support Connector"

    elif is_distress:
        agents_activated = ["Distress Detection Agent", "Emotional Support Agent", "Prevention & Wellness Agent"]
        risk_data        = distress_detection_agent(user_input)
        primary_response = emotional_support_agent(user_input)
        mood_guess       = "low"
        wellness_plan    = generate_wellness_plan(mood_guess, "high", "distressed")
        resources        = support_connector_agent(risk_level=risk_data.get("risk_level", "Moderate Risk"))
        reason           = "Distress signals detected >> activating Emotional Support + Distress Detection + Wellness"

    elif is_info:
        agents_activated = ["Mental Health Awareness Agent"]
        primary_response = awareness_agent(user_input)
        reason           = "Informational query detected >> activating Awareness Agent"

    elif is_wellness:
        agents_activated = ["Prevention & Wellness Agent", "Emotional Support Agent"]
        wellness_plan    = wellness_agent(user_input)
        primary_response = emotional_support_agent(user_input)
        reason           = "Wellness/coping request detected >> activating Wellness + Emotional Support"

    else:
        # Default: general emotional support
        agents_activated = ["Emotional Support Agent"]
        primary_response = emotional_support_agent(user_input)
        reason           = "General message >> activating Emotional Support Agent"

    # ── Step 3: Assemble Orchestrated Output ─────────────────────────────────
    return {
        "agents_activated" : agents_activated,
        "orchestrator_reason": reason,
        "primary_response" : primary_response,
        "risk_data"        : risk_data,
        "wellness_plan"    : wellness_plan,
        "resources"        : resources,
        "timestamp"        : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

# =============================================================================
#  Specialized Agent Functions
# =============================================================================

def awareness_agent(query: str) -> str:
    """
    Agent 1 – Mental Health Awareness Agent
    Educates users about mental health topics using IBM Granite.
    """
    rag_ctx = retrieve_context(query)
    context_block = f"\n\nRelevant Knowledge Base Context:\n{rag_ctx}" if rag_ctx else ""

    prompt = f"""You are MindGuard AI's Mental Health Awareness Agent, powered by IBM Granite.
Your role is to educate users about mental health topics in a clear, compassionate, and evidence-based way.
Cover topics such as anxiety, depression, stress, burnout, emotional wellness, mindfulness, and self-care.{context_block}

User Question: {query}

Provide a helpful, accurate, and empathetic educational response (3-5 paragraphs):"""

    return generate_response(prompt, max_tokens=600)


def emotional_support_agent(message: str) -> str:
    """
    Agent 2 – Emotional Support Agent
    Provides empathetic conversational support without judgment.
    """
    prompt = f"""You are MindGuard AI's Emotional Support Agent, powered by IBM Granite.
Your role is to provide warm, empathetic, non-judgmental emotional support.
Listen actively, acknowledge feelings, and encourage healthy coping.
Never diagnose. Always be compassionate and supportive.

User says: "{message}"

Respond with genuine empathy and support (2-4 paragraphs).
End your response with:
"— MindGuard AI provides educational and emotional support. It is not a substitute for professional medical or psychological care." """

    return generate_response(prompt, max_tokens=500)


def distress_detection_agent(text: str) -> dict:
    """
    Agent 3 – Distress Detection Agent
    Analyses text for mental-health risk using IBM Granite reasoning.
    """
    return detect_risk(text)


def wellness_agent(message: str) -> str:
    """
    Agent 4 – Prevention & Wellness Agent
    Generates proactive, personalised wellness recommendations.
    """
    prompt = f"""You are MindGuard AI's Prevention & Wellness Agent, powered by IBM Granite.
Based on what the user shared, create practical wellness recommendations.
Include breathing exercises, mindfulness tips, journaling prompts, sleep advice, or movement suggestions.

User message: "{message}"

Provide a warm, actionable wellness recommendation (2-4 paragraphs):"""

    return generate_response(prompt, max_tokens=500)


def support_connector_agent(risk_level: str = "Low Risk") -> dict:
    """
    Agent 5 – Human Support Connector Agent
    Returns crisis resources and professional support recommendations.
    Always reminds users that professional care is available.
    """
    crisis_resources = {
        "crisis_hotlines": [
            {"name": "National Suicide Prevention Lifeline (USA)", "contact": "988 or 1-800-273-8255", "available": "24/7"},
            {"name": "Crisis Text Line (USA)",                     "contact": "Text HOME to 741741",   "available": "24/7"},
            {"name": "Samaritans (UK)",                            "contact": "116 123",               "available": "24/7"},
            {"name": "iCall (India)",                              "contact": "9152987821",             "available": "Mon–Sat 8am–10pm"},
            {"name": "Vandrevala Foundation (India)",              "contact": "1860-2662-345",          "available": "24/7"},
            {"name": "International Association for Suicide Prevention", "contact": "https://www.iasp.info/resources/Crisis_Centres/", "available": "Global directory"},
        ],
        "online_resources": [
            {"name": "WHO Mental Health",      "url": "https://www.who.int/health-topics/mental-health"},
            {"name": "NAMI (USA)",             "url": "https://www.nami.org"},
            {"name": "Mind (UK)",              "url": "https://www.mind.org.uk"},
            {"name": "Mental Health Foundation","url": "https://www.mentalhealth.org.uk"},
            {"name": "Headspace (Meditation)", "url": "https://www.headspace.com"},
        ],
        "professional_support": [
            "Speak with a licensed therapist or counsellor",
            "Contact your primary care physician",
            "Visit a community mental health centre",
            "Ask your employer about Employee Assistance Programs (EAP)",
            "Explore tele-therapy platforms (BetterHelp, Talkspace)",
        ],
        "risk_message": _get_risk_message(risk_level),
    }
    return crisis_resources


# =============================================================================
#  Utility / Demo Fallback Functions
# =============================================================================

def _demo_fallback(prompt: str) -> str:
    """
    Rule-based responses used when IBM watsonx.ai credentials are not configured.
    This keeps the UI fully functional for demos without a live API key.
    """
    lower = prompt.lower()

    if "awareness" in lower or "what is" in lower or "anxiety" in lower:
        return ("Mental health encompasses our emotional, psychological, and social well-being. "
                "Anxiety is a natural stress response, but when persistent, it can interfere with daily life. "
                "Signs include excessive worry, restlessness, and difficulty concentrating. "
                "Evidence-based approaches like Cognitive Behavioural Therapy (CBT), mindfulness, "
                "regular exercise, and healthy sleep habits are highly effective. "
                "Remember: seeking help is a sign of strength, not weakness.\n\n"
                "[Demo Mode – Configure WATSONX_API_KEY for IBM Granite responses]")

    if "support" in lower or "feel" in lower or "overwhelm" in lower or "lonely" in lower:
        return ("I hear you, and I want you to know that your feelings are completely valid. "
                "It takes courage to reach out, and I'm genuinely glad you did. "
                "Whatever you're going through, you don't have to face it alone. "
                "Taking things one small step at a time – even just one deep breath – can make a difference. "
                "You matter, and support is available.\n\n"
                "— MindGuard AI provides educational and emotional support. "
                "It is not a substitute for professional medical or psychological care.\n\n"
                "[Demo Mode – Configure WATSONX_API_KEY for IBM Granite responses]")

    if "wellness" in lower or "relax" in lower or "meditation" in lower:
        return ("Here is a simple wellness practice you can try right now:\n\n"
                "🌬️ **Box Breathing (4-4-4-4):** Inhale for 4 counts, hold for 4, exhale for 4, hold for 4. "
                "Repeat 4 times. This activates your parasympathetic nervous system and reduces cortisol.\n\n"
                "📓 **Evening Journal Prompt:** Write 3 things you are grateful for and 1 small win from today.\n\n"
                "💤 **Sleep Tip:** Dim screens 1 hour before bed and keep a consistent sleep schedule.\n\n"
                "[Demo Mode – Configure WATSONX_API_KEY for IBM Granite responses]")

    if "risk" in lower or "assess" in lower:
        return json.dumps({
            "risk_level" : "Low Risk",
            "risk_score" : 15,
            "explanation": "No significant distress indicators detected in this message.",
            "next_steps" : "Continue practising self-care and reach out if feelings intensify.",
        })

    return ("Thank you for reaching out to MindGuard AI. I'm here to support you. "
            "Feel free to share what's on your mind – whether it's a question about mental health, "
            "an emotion you're experiencing, or a need for coping strategies. "
            "You are not alone.\n\n"
            "[Demo Mode – Configure WATSONX_API_KEY for IBM Granite responses]")


def _keyword_risk_score(text: str) -> dict:
    """Fallback keyword-based risk scoring when the LLM fails to return valid JSON."""
    lower = text.lower()
    score = 0

    high_risk_words = ["suicide", "kill myself", "end my life", "want to die",
                       "no reason to live", "self harm", "hurt myself"]
    moderate_words  = ["hopeless", "worthless", "can't go on", "give up",
                       "breaking down", "exhausted", "can't cope"]
    low_risk_words  = ["sad", "lonely", "stressed", "anxious", "worried",
                       "overwhelmed", "tired", "frustrated"]

    for w in high_risk_words:
        if w in lower: score += 35
    for w in moderate_words:
        if w in lower: score += 15
    for w in low_risk_words:
        if w in lower: score += 5

    score = min(score, 100)

    if score >= 60:
        level      = "High Risk"
        next_steps = "Please contact a crisis helpline immediately. You are not alone."
    elif score >= 25:
        level      = "Moderate Risk"
        next_steps = "Consider speaking with a mental health professional. Reach out to a trusted person."
    else:
        level      = "Low Risk"
        next_steps = "Maintain self-care practices and monitor your emotional well-being."

    return {
        "risk_level" : level,
        "risk_score" : score,
        "explanation": f"Keyword-based assessment detected distress indicators (score: {score}/100).",
        "next_steps" : next_steps,
    }


def _get_risk_message(risk_level: str) -> str:
    """Return an appropriate message based on detected risk level."""
    if risk_level == "High Risk":
        return ("⚠️ High distress detected. Please reach out to a crisis helpline immediately. "
                "You are not alone, and help is available right now.")
    if risk_level == "Moderate Risk":
        return ("Your well-being matters. Consider speaking with a mental health professional "
                "or a trusted person in your life.")
    return ("Keep prioritising your mental health. Regular self-care and social connection "
            "are powerful protective factors.")

# =============================================================================
#  Lightweight RAG System
# =============================================================================
# In-memory document store: list of { source, chunk } dicts
RAG_STORE: list = []
CHUNK_SIZE = 500   # characters per chunk


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list:
    """Split text into overlapping chunks for retrieval."""
    words    = text.split()
    chunks   = []
    step     = max(1, chunk_size // 6)          # ~16% overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i: i + chunk_size // 6 * 6])
        if chunk:
            chunks.append(chunk)
    return chunks


def ingest_document(filename: str, text: str):
    """Add document chunks to the RAG in-memory store."""
    global RAG_STORE
    chunks = _chunk_text(text)
    for chunk in chunks:
        RAG_STORE.append({"source": filename, "chunk": chunk})
    print(f"[RAG] Ingested '{filename}': {len(chunks)} chunks. Total chunks: {len(RAG_STORE)}")

# =============================================================================
#  Flask Routes
# =============================================================================

@app.route("/")
def index():
    """Serve the main single-page application."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Main chat endpoint.
    Receives user message → orchestrates agents → returns structured JSON.
    """
    data         = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    result = orchestrate_agents(user_message)
    return jsonify(result)


@app.route("/api/distress", methods=["POST"])
def api_distress():
    """Standalone distress detection endpoint."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    return jsonify(distress_detection_agent(text))


@app.route("/api/wellness", methods=["POST"])
def api_wellness():
    """Standalone wellness plan endpoint."""
    data           = request.get_json(silent=True) or {}
    mood           = data.get("mood", "neutral")
    stress_level   = data.get("stress_level", "moderate")
    emotional_state= data.get("emotional_state", "stable")
    plan = generate_wellness_plan(mood, stress_level, emotional_state)
    return jsonify({"wellness_plan": plan})


@app.route("/api/resources", methods=["GET"])
def api_resources():
    """Return support resources (optionally filtered by risk level)."""
    risk_level = request.args.get("risk_level", "Low Risk")
    return jsonify(support_connector_agent(risk_level))


@app.route("/api/rag/upload", methods=["POST"])
def api_rag_upload():
    """
    RAG document upload endpoint.
    Accepts PDF or TXT files and adds them to the in-memory RAG store.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    ext      = filename.rsplit(".", 1)[-1].lower()

    if ext == "txt":
        text = file.read().decode("utf-8", errors="ignore")
        ingest_document(filename, text)
        return jsonify({"status": "success", "message": f"Ingested '{filename}' into RAG store.",
                        "total_chunks": len(RAG_STORE)})

    if ext == "pdf":
        if not PDF_SUPPORT:
            return jsonify({"error": "PyPDF2 not installed. Only .txt files are supported."}), 400
        import io
        reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        text   = " ".join(page.extract_text() or "" for page in reader.pages)
        ingest_document(filename, text)
        return jsonify({"status": "success", "message": f"Ingested '{filename}' into RAG store.",
                        "total_chunks": len(RAG_STORE)})

    return jsonify({"error": "Unsupported file type. Upload .txt or .pdf only."}), 400


@app.route("/api/rag/status", methods=["GET"])
def api_rag_status():
    """Return RAG store statistics."""
    sources = list({doc["source"] for doc in RAG_STORE})
    return jsonify({"total_chunks": len(RAG_STORE), "documents": sources})


# =============================================================================
#  HTML Template – Modern Bootstrap 5 Single-Page Application
# =============================================================================
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>MindGuard AI – Mental Health Awareness Agent</title>

  <!-- Bootstrap 5 CSS -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet" />
  <!-- Bootstrap Icons -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css" rel="stylesheet" />

  <style>
    /* ── Global ─────────────────────────────────────────────────────────── */
    :root {
      --mg-teal   : #0f7b7b;
      --mg-teal-lt: #e6f4f4;
      --mg-purple : #6b48c8;
      --mg-blue   : #1a56db;
      --mg-amber  : #d97706;
      --mg-red    : #dc2626;
      --mg-green  : #16a34a;
      --mg-surface: #f8fafc;
      --mg-border : #e2e8f0;
      --mg-text   : #1e293b;
      --mg-muted  : #64748b;
    }
    body {
      background: #f0f4f8;
      font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
      color: var(--mg-text);
      font-size: 14.5px;
      line-height: 1.65;
    }

    /* ── Navbar ─────────────────────────────────────────────────────────── */
    .mg-navbar {
      background: linear-gradient(135deg, #0f4c75 0%, #0f7b7b 60%, #6b48c8 100%);
      padding: 0.75rem 1.5rem;
      box-shadow: 0 2px 12px rgba(0,0,0,0.18);
    }
    .mg-navbar .brand-title { font-size: 1.35rem; font-weight: 700; color: #fff; letter-spacing: 0.5px; }
    .mg-navbar .brand-sub   { font-size: 0.75rem; color: rgba(255,255,255,0.75); margin-top: -2px; }
    .mg-badge {
      background: rgba(255,255,255,0.18);
      color: #fff;
      border-radius: 20px;
      padding: 3px 10px;
      font-size: 0.72rem;
      font-weight: 600;
      border: 1px solid rgba(255,255,255,0.3);
    }

    /* ── Cards ──────────────────────────────────────────────────────────── */
    .mg-card {
      background: #fff;
      border: 1px solid var(--mg-border);
      border-radius: 14px;
      box-shadow: 0 1px 6px rgba(0,0,0,0.06);
      overflow: hidden;
    }
    .mg-card-header {
      padding: 0.75rem 1.1rem;
      font-weight: 600;
      font-size: 0.85rem;
      letter-spacing: 0.3px;
      display: flex;
      align-items: center;
      gap: 8px;
      border-bottom: 1px solid var(--mg-border);
    }
    .header-teal   { background: var(--mg-teal-lt); color: var(--mg-teal); }
    .header-purple { background: #ede9fb; color: var(--mg-purple); }
    .header-amber  { background: #fef3c7; color: var(--mg-amber); }
    .header-blue   { background: #dbeafe; color: var(--mg-blue); }
    .header-green  { background: #dcfce7; color: var(--mg-green); }

    /* ── Chat ───────────────────────────────────────────────────────────── */
    #chat-messages {
      height: 420px;
      overflow-y: auto;
      padding: 1rem;
      background: #f8fafc;
      scroll-behavior: smooth;
    }
    .msg-wrap { display: flex; margin-bottom: 12px; }
    .msg-wrap.user  { justify-content: flex-end; }
    .msg-wrap.bot   { justify-content: flex-start; }
    .msg-bubble {
      max-width: 78%;
      padding: 10px 14px;
      border-radius: 16px;
      font-size: 0.88rem;
      line-height: 1.6;
      white-space: pre-wrap;
    }
    .msg-bubble.user { background: var(--mg-teal); color: #fff; border-bottom-right-radius: 4px; }
    .msg-bubble.bot  { background: #fff; border: 1px solid var(--mg-border); border-bottom-left-radius: 4px; color: var(--mg-text); }
    .msg-avatar {
      width: 32px; height: 32px; border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 1rem; flex-shrink: 0;
    }
    .msg-avatar.user { background: var(--mg-teal); color: #fff; margin-left: 8px; }
    .msg-avatar.bot  { background: var(--mg-purple); color: #fff; margin-right: 8px; }

    /* ── Input bar ──────────────────────────────────────────────────────── */
    .chat-input-row { padding: 0.75rem 1rem; border-top: 1px solid var(--mg-border); background: #fff; }
    #user-input {
      border-radius: 24px;
      border: 1.5px solid var(--mg-border);
      padding: 0.55rem 1rem;
      font-size: 0.88rem;
      resize: none;
      transition: border 0.2s;
    }
    #user-input:focus { border-color: var(--mg-teal); box-shadow: none; outline: none; }
    .btn-send {
      background: var(--mg-teal);
      color: #fff;
      border: none;
      border-radius: 50%;
      width: 42px; height: 42px;
      display: flex; align-items: center; justify-content: center;
      font-size: 1.1rem;
      flex-shrink: 0;
      transition: background 0.2s;
    }
    .btn-send:hover { background: #0a5c5c; }

    /* ── Agent Cards ────────────────────────────────────────────────────── */
    .agent-card {
      border-radius: 10px;
      border: 1.5px solid var(--mg-border);
      padding: 10px 12px;
      margin-bottom: 8px;
      font-size: 0.82rem;
      transition: all 0.25s;
      background: #fff;
    }
    .agent-card.active {
      border-color: var(--mg-teal);
      background: var(--mg-teal-lt);
      box-shadow: 0 0 0 3px rgba(15,123,123,0.12);
    }
    .agent-name { font-weight: 700; font-size: 0.83rem; }
    .agent-icon { font-size: 1.15rem; margin-right: 6px; }

    /* ── Risk Gauge ─────────────────────────────────────────────────────── */
    .risk-bar-bg {
      height: 10px;
      background: #e2e8f0;
      border-radius: 8px;
      overflow: hidden;
      margin: 6px 0;
    }
    .risk-bar-fill { height: 100%; border-radius: 8px; transition: width 0.6s ease; }
    .risk-low      { background: var(--mg-green); }
    .risk-moderate { background: var(--mg-amber); }
    .risk-high     { background: var(--mg-red); }
    .risk-badge    { font-size: 0.78rem; font-weight: 700; padding: 3px 10px; border-radius: 20px; }
    .risk-badge.low      { background: #dcfce7; color: var(--mg-green); }
    .risk-badge.moderate { background: #fef3c7; color: var(--mg-amber); }
    .risk-badge.high     { background: #fee2e2; color: var(--mg-red); }

    /* ── Wellness ───────────────────────────────────────────────────────── */
    .wellness-item {
      background: #f0fdf4;
      border-left: 3px solid var(--mg-green);
      border-radius: 6px;
      padding: 8px 12px;
      margin-bottom: 8px;
      font-size: 0.84rem;
    }

    /* ── Resources ──────────────────────────────────────────────────────── */
    .resource-row {
      display: flex; align-items: flex-start; gap: 8px;
      border-bottom: 1px solid var(--mg-border);
      padding: 7px 0;
      font-size: 0.83rem;
    }
    .resource-row:last-child { border-bottom: none; }
    .hotline-badge {
      background: #fee2e2;
      color: var(--mg-red);
      border-radius: 6px;
      padding: 2px 7px;
      font-size: 0.75rem;
      font-weight: 700;
      white-space: nowrap;
    }

    /* ── RAG Upload ─────────────────────────────────────────────────────── */
    .rag-drop-zone {
      border: 2px dashed var(--mg-border);
      border-radius: 10px;
      padding: 20px;
      text-align: center;
      color: var(--mg-muted);
      font-size: 0.84rem;
      cursor: pointer;
      transition: border 0.2s;
    }
    .rag-drop-zone:hover { border-color: var(--mg-teal); color: var(--mg-teal); }

    /* ── Spinner ────────────────────────────────────────────────────────── */
    .typing-indicator { display: flex; gap: 4px; padding: 6px 0 2px; }
    .typing-dot {
      width: 7px; height: 7px; border-radius: 50%;
      background: var(--mg-muted);
      animation: bounce 1.2s infinite;
    }
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    @keyframes bounce {
      0%, 60%, 100% { transform: translateY(0); }
      30%            { transform: translateY(-6px); }
    }

    /* ── Workflow Stepper ───────────────────────────────────────────────── */
    .step-flow {
      display: flex; flex-wrap: wrap; gap: 6px;
      align-items: center; padding: 10px 0;
    }
    .step-pill {
      background: var(--mg-surface);
      border: 1px solid var(--mg-border);
      border-radius: 20px;
      padding: 4px 12px;
      font-size: 0.78rem;
      color: var(--mg-muted);
    }
    .step-pill.active { background: var(--mg-teal); color: #fff; border-color: var(--mg-teal); }
    .step-arrow { color: var(--mg-muted); font-size: 0.9rem; }

    /* ── Utility ────────────────────────────────────────────────────────── */
    .section-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase; color: var(--mg-muted); margin-bottom: 6px; }
    .disclaimer    { font-size: 0.75rem; color: var(--mg-muted); font-style: italic; border-top: 1px solid var(--mg-border); padding-top: 6px; margin-top: 8px; }
    .powered-by    { font-size: 0.73rem; color: var(--mg-muted); display: flex; align-items: center; gap: 5px; }
    .granite-chip  { background: #1a56db; color: #fff; border-radius: 4px; padding: 1px 7px; font-size: 0.7rem; font-weight: 700; }
    .watsonx-chip  { background: #0f4c75; color: #fff; border-radius: 4px; padding: 1px 7px; font-size: 0.7rem; font-weight: 700; }

    /* ── Tabs ───────────────────────────────────────────────────────────── */
    .nav-tabs .nav-link { font-size: 0.82rem; color: var(--mg-muted); border: none; padding: 0.4rem 0.85rem; }
    .nav-tabs .nav-link.active { color: var(--mg-teal); font-weight: 700; border-bottom: 2px solid var(--mg-teal); background: none; }

    /* ── Scrollbar ──────────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 10px; }

    /* ── Welcome Banner ─────────────────────────────────────────────────── */
    .welcome-banner {
      background: linear-gradient(135deg, #0f4c75, #0f7b7b);
      color: #fff;
      border-radius: 12px;
      padding: 18px 22px;
      margin-bottom: 16px;
    }
    .welcome-banner h5 { font-weight: 700; margin-bottom: 4px; }
    .welcome-banner p  { font-size: 0.84rem; opacity: 0.88; margin: 0; }
  </style>
</head>

<body>

<!-- ═══════════════════════════════ NAVBAR ═══════════════════════════════ -->
<nav class="mg-navbar d-flex align-items-center justify-content-between flex-wrap gap-2">
  <div>
    <div class="brand-title"><i class="bi bi-shield-heart-fill me-2"></i>MindGuard AI</div>
    <div class="brand-sub">Mental Health Awareness &amp; Suicide Prevention Agent</div>
  </div>
  <div class="d-flex align-items-center gap-2 flex-wrap">
    <span class="mg-badge"><i class="bi bi-cpu-fill me-1"></i>IBM Granite</span>
    <span class="mg-badge"><i class="bi bi-cloud-fill me-1"></i>watsonx.ai</span>
    <span class="mg-badge"><i class="bi bi-diagram-3-fill me-1"></i>Agentic AI</span>
    <span class="mg-badge"><i class="bi bi-heart-pulse-fill me-1"></i>5 Agents</span>
  </div>
</nav>

<!-- ═══════════════════════════════ MAIN ════════════════════════════════ -->
<div class="container-fluid px-3 py-3">
  <div class="row g-3">

    <!-- ─────────────────────── LEFT COLUMN ─────────────────────────── -->
    <div class="col-lg-4 col-xl-3">

      <!-- Welcome Banner -->
      <div class="welcome-banner">
        <h5><i class="bi bi-shield-heart-fill me-2"></i>Your Safe Space</h5>
        <p>MindGuard AI provides compassionate mental health support powered by IBM Granite models and Agentic AI architecture.</p>
      </div>

      <!-- Agent Status Panel -->
      <div class="mg-card mb-3">
        <div class="mg-card-header header-purple">
          <i class="bi bi-diagram-3-fill"></i> Agent Orchestration Panel
        </div>
        <div class="p-2">
          <div class="section-label">Active Agents</div>

          <div class="agent-card" id="agent-awareness">
            <span class="agent-icon">🧠</span><span class="agent-name">Awareness Agent</span>
            <div class="text-muted" style="font-size:0.77rem;margin-top:3px;">Educates on mental health topics</div>
          </div>
          <div class="agent-card" id="agent-support">
            <span class="agent-icon">💙</span><span class="agent-name">Emotional Support Agent</span>
            <div class="text-muted" style="font-size:0.77rem;margin-top:3px;">Provides empathetic conversation</div>
          </div>
          <div class="agent-card" id="agent-distress">
            <span class="agent-icon">🔍</span><span class="agent-name">Distress Detection Agent</span>
            <div class="text-muted" style="font-size:0.77rem;margin-top:3px;">Analyses risk indicators</div>
          </div>
          <div class="agent-card" id="agent-wellness">
            <span class="agent-icon">🌿</span><span class="agent-name">Prevention &amp; Wellness Agent</span>
            <div class="text-muted" style="font-size:0.77rem;margin-top:3px;">Generates wellness plans</div>
          </div>
          <div class="agent-card" id="agent-connector">
            <span class="agent-icon">🤝</span><span class="agent-name">Support Connector Agent</span>
            <div class="text-muted" style="font-size:0.77rem;margin-top:3px;">Connects to professional help</div>
          </div>

          <div class="mt-2 p-2 rounded" style="background:#f8fafc;font-size:0.8rem;" id="orchestrator-reason">
            <i class="bi bi-cpu text-muted me-1"></i>
            <span class="text-muted">Awaiting input…</span>
          </div>
        </div>
      </div>

      <!-- Workflow Stepper -->
      <div class="mg-card mb-3">
        <div class="mg-card-header header-blue">
          <i class="bi bi-arrows-angle-contract"></i> Orchestrator Workflow
        </div>
        <div class="p-2">
          <div class="step-flow" id="workflow-steps">
            <span class="step-pill">User Input</span>
            <span class="step-arrow">›</span>
            <span class="step-pill">Orchestrator</span>
            <span class="step-arrow">›</span>
            <span class="step-pill">Agent(s)</span>
            <span class="step-arrow">›</span>
            <span class="step-pill">IBM Granite</span>
            <span class="step-arrow">›</span>
            <span class="step-pill">Response</span>
          </div>
          <div class="powered-by mt-1">
            <span>Powered by</span>
            <span class="granite-chip">IBM Granite</span>
            <span class="watsonx-chip">watsonx.ai</span>
          </div>
        </div>
      </div>

      <!-- RAG Upload -->
      <div class="mg-card">
        <div class="mg-card-header header-amber">
          <i class="bi bi-database-fill-up"></i> RAG Knowledge Base
        </div>
        <div class="p-2">
          <div class="section-label">Upload Documents</div>
          <div class="rag-drop-zone" onclick="document.getElementById('rag-file-input').click()">
            <i class="bi bi-cloud-arrow-up fs-4 d-block mb-1"></i>
            Click to upload PDF or TXT<br>
            <small>WHO Guidelines · Coping Strategies · Resources</small>
          </div>
          <input type="file" id="rag-file-input" accept=".txt,.pdf" style="display:none" onchange="uploadRAG(this)" />
          <div id="rag-status" class="mt-2" style="font-size:0.8rem;color:var(--mg-muted);">
            No documents loaded.
          </div>
        </div>
      </div>

    </div><!-- /left -->

    <!-- ─────────────────────── CENTER COLUMN ────────────────────────── -->
    <div class="col-lg-5 col-xl-5">

      <!-- Chat Interface -->
      <div class="mg-card mb-3">
        <div class="mg-card-header header-teal">
          <i class="bi bi-chat-heart-fill"></i> MindGuard AI Chat
          <span class="ms-auto" style="font-size:0.75rem;color:var(--mg-muted);font-weight:400;">
            <i class="bi bi-circle-fill text-success me-1" style="font-size:0.5rem;"></i>Online
          </span>
        </div>
        <div id="chat-messages">
          <!-- Welcome message -->
          <div class="msg-wrap bot">
            <div class="msg-avatar bot"><i class="bi bi-shield-heart"></i></div>
            <div class="msg-bubble bot">Hello 👋 I'm MindGuard AI, your compassionate mental health companion.

I'm here to listen, support, and guide you. You can:
• Ask about mental health topics (anxiety, depression, stress)
• Share how you're feeling right now
• Request coping strategies or wellness tips
• Get connected to professional support

Remember, this is a safe space. How are you feeling today?</div>
          </div>
        </div>

        <div class="chat-input-row">
          <div class="d-flex gap-2 align-items-end">
            <textarea id="user-input" class="form-control" rows="2"
              placeholder="Share what's on your mind…" onkeydown="handleKey(event)"></textarea>
            <button class="btn-send" onclick="sendMessage()" title="Send">
              <i class="bi bi-send-fill"></i>
            </button>
          </div>
          <div class="d-flex flex-wrap gap-2 mt-2">
            <button class="btn btn-sm btn-outline-secondary rounded-pill" style="font-size:0.75rem;"
              onclick="quickSend('What is anxiety?')">What is anxiety?</button>
            <button class="btn btn-sm btn-outline-secondary rounded-pill" style="font-size:0.75rem;"
              onclick="quickSend('I feel overwhelmed and stressed')">I feel overwhelmed</button>
            <button class="btn btn-sm btn-outline-secondary rounded-pill" style="font-size:0.75rem;"
              onclick="quickSend('Give me breathing exercises to calm down')">Breathing exercises</button>
            <button class="btn btn-sm btn-outline-secondary rounded-pill" style="font-size:0.75rem;"
              onclick="quickSend('Signs of depression')">Signs of depression</button>
          </div>
        </div>
      </div>

      <!-- Risk Detection Panel -->
      <div class="mg-card">
        <div class="mg-card-header header-amber">
          <i class="bi bi-activity"></i> Risk Detection Panel
        </div>
        <div class="p-3" id="risk-panel">
          <div class="text-center text-muted py-3" style="font-size:0.85rem;">
            <i class="bi bi-shield-check fs-3 d-block mb-2 text-success"></i>
            Risk assessment will appear after your first message.
          </div>
        </div>
      </div>

    </div><!-- /center -->

    <!-- ─────────────────────── RIGHT COLUMN ─────────────────────────── -->
    <div class="col-lg-3 col-xl-4">

      <!-- Tabs: Wellness | Resources -->
      <div class="mg-card mb-3">
        <div class="mg-card-header header-green">
          <i class="bi bi-heart-pulse"></i> Wellness &amp; Support
        </div>
        <div class="p-0">
          <ul class="nav nav-tabs px-2 pt-1">
            <li class="nav-item">
              <a class="nav-link active" data-bs-toggle="tab" href="#tab-wellness">
                <i class="bi bi-leaf me-1"></i>Wellness Plan
              </a>
            </li>
            <li class="nav-item">
              <a class="nav-link" data-bs-toggle="tab" href="#tab-resources">
                <i class="bi bi-telephone-fill me-1"></i>Resources
              </a>
            </li>
          </ul>
          <div class="tab-content p-3">

            <!-- Wellness Tab -->
            <div class="tab-pane fade show active" id="tab-wellness">
              <div id="wellness-panel">
                <div class="text-center text-muted py-3" style="font-size:0.84rem;">
                  <i class="bi bi-leaf fs-3 d-block mb-2 text-success"></i>
                  Your personalised wellness plan will appear here after chatting.
                </div>
              </div>

              <div class="mt-3">
                <div class="section-label">Quick Wellness Check</div>
                <div class="d-flex gap-2 flex-wrap mb-2">
                  <select id="mood-select" class="form-select form-select-sm" style="font-size:0.8rem;flex:1;">
                    <option value="happy">😊 Happy</option>
                    <option value="neutral" selected>😐 Neutral</option>
                    <option value="sad">😔 Sad</option>
                    <option value="anxious">😰 Anxious</option>
                    <option value="exhausted">😩 Exhausted</option>
                  </select>
                  <select id="stress-select" class="form-select form-select-sm" style="font-size:0.8rem;flex:1;">
                    <option value="low">🟢 Low Stress</option>
                    <option value="moderate" selected>🟡 Moderate</option>
                    <option value="high">🔴 High Stress</option>
                  </select>
                </div>
                <button class="btn btn-sm w-100" style="background:var(--mg-green);color:#fff;font-size:0.82rem;"
                  onclick="generateWellnessPlan()">
                  <i class="bi bi-stars me-1"></i>Generate My Wellness Plan
                </button>
              </div>
            </div>

            <!-- Resources Tab -->
            <div class="tab-pane fade" id="tab-resources">
              <div id="resources-panel">
                <div class="section-label">Crisis Hotlines</div>
                <div class="resource-row">
                  <span class="hotline-badge">USA</span>
                  <div><strong>988 Suicide &amp; Crisis Lifeline</strong><br>
                    <span class="text-muted">Call or text 988 · 24/7</span></div>
                </div>
                <div class="resource-row">
                  <span class="hotline-badge">USA</span>
                  <div><strong>Crisis Text Line</strong><br>
                    <span class="text-muted">Text HOME to 741741 · 24/7</span></div>
                </div>
                <div class="resource-row">
                  <span class="hotline-badge">UK</span>
                  <div><strong>Samaritans</strong><br>
                    <span class="text-muted">116 123 · 24/7, free</span></div>
                </div>
                <div class="resource-row">
                  <span class="hotline-badge">INDIA</span>
                  <div><strong>iCall</strong><br>
                    <span class="text-muted">9152987821 · Mon–Sat</span></div>
                </div>
                <div class="resource-row">
                  <span class="hotline-badge">GLOBAL</span>
                  <div><strong>IASP Crisis Centres</strong><br>
                    <a href="https://www.iasp.info/resources/Crisis_Centres/" target="_blank"
                      style="font-size:0.8rem;color:var(--mg-blue);">iasp.info →</a></div>
                </div>

                <div class="section-label mt-3">Online Resources</div>
                <div class="resource-row">
                  <i class="bi bi-globe text-muted"></i>
                  <a href="https://www.who.int/health-topics/mental-health" target="_blank"
                    style="color:var(--mg-blue);font-size:0.83rem;">WHO Mental Health</a>
                </div>
                <div class="resource-row">
                  <i class="bi bi-globe text-muted"></i>
                  <a href="https://www.nami.org" target="_blank"
                    style="color:var(--mg-blue);font-size:0.83rem;">NAMI (USA)</a>
                </div>
                <div class="resource-row">
                  <i class="bi bi-globe text-muted"></i>
                  <a href="https://www.mind.org.uk" target="_blank"
                    style="color:var(--mg-blue);font-size:0.83rem;">Mind (UK)</a>
                </div>

                <div class="disclaimer mt-2">
                  MindGuard AI is not a substitute for professional medical or psychological care.
                  If you are in crisis, please contact emergency services or a crisis line immediately.
                </div>
              </div>
            </div>

          </div>
        </div>
      </div>

      <!-- IBM watsonx.ai Info Card -->
      <div class="mg-card">
        <div class="mg-card-header header-blue">
          <i class="bi bi-cloud-fill"></i> IBM watsonx.ai Studio
        </div>
        <div class="p-2" style="font-size:0.82rem;">
          <div class="resource-row">
            <i class="bi bi-cpu text-primary"></i>
            <div><strong>Model</strong><br><span class="text-muted">ibm/granite-3-3-8b-instruct</span></div>
          </div>
          <div class="resource-row">
            <i class="bi bi-diagram-3 text-primary"></i>
            <div><strong>Architecture</strong><br><span class="text-muted">5-Agent Orchestration</span></div>
          </div>
          <div class="resource-row">
            <i class="bi bi-database text-primary"></i>
            <div><strong>RAG</strong><br><span class="text-muted">Lightweight In-Memory Retrieval</span></div>
          </div>
          <div class="resource-row">
            <i class="bi bi-shield-check text-success"></i>
            <div><strong>Platform</strong><br><span class="text-muted">IBM watsonx.ai Studio</span></div>
          </div>

          <div class="mt-2 p-2 rounded" style="background:#eff6ff;font-size:0.78rem;color:#1e40af;">
            <i class="bi bi-info-circle me-1"></i>
            Set <code>WATSONX_API_KEY</code>, <code>WATSONX_PROJECT_ID</code>,
            and <code>WATSONX_URL</code> environment variables to enable live IBM Granite responses.
          </div>
        </div>
      </div>

    </div><!-- /right -->

  </div><!-- /row -->
</div><!-- /container -->

<!-- ═══════════════════════════ FOOTER ═════════════════════════════════ -->
<footer style="text-align:center;padding:14px 0 10px;border-top:1px solid #e2e8f0;
  font-size:0.73rem;color:#94a3b8;margin-top:10px;">
  Made with <span style="color:#e05252;">♥</span> using IBM watsonx.ai Studio &amp; IBM Granite &nbsp;·&nbsp;
  MindGuard AI &nbsp;·&nbsp; Agentic AI for Mental Health Awareness
</footer>

<!-- Bootstrap 5 JS -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>

<script>
/* ═══════════════════════════ CLIENT JS ══════════════════════════════ */

const AGENT_IDS = {
  "Mental Health Awareness Agent"   : "agent-awareness",
  "Emotional Support Agent"         : "agent-support",
  "Distress Detection Agent"        : "agent-distress",
  "Prevention & Wellness Agent"     : "agent-wellness",
  "Human Support Connector Agent"   : "agent-connector",
};

// ── Keyboard shortcut ──────────────────────────────────────────────────────
function handleKey(e) {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
}

function quickSend(text) {
  document.getElementById("user-input").value = text;
  sendMessage();
}

// ── Main send function ─────────────────────────────────────────────────────
async function sendMessage() {
  const input   = document.getElementById("user-input");
  const message = input.value.trim();
  if (!message) return;

  appendMessage("user", message);
  input.value = "";

  const typingId = showTyping();

  try {
    const res  = await fetch("/api/chat", {
      method : "POST",
      headers: { "Content-Type": "application/json" },
      body   : JSON.stringify({ message }),
    });
    const data = await res.json();
    removeTyping(typingId);

    // Main chat response
    if (data.primary_response) appendMessage("bot", data.primary_response);

    // Update UI panels
    updateAgentPanel(data.agents_activated || [], data.orchestrator_reason || "");
    updateWorkflowSteps(data.agents_activated || []);
    if (data.risk_data && data.risk_data.risk_level) updateRiskPanel(data.risk_data);
    if (data.wellness_plan) updateWellnessPanel(data.wellness_plan);

  } catch (err) {
    removeTyping(typingId);
    appendMessage("bot", "I'm sorry, I encountered an issue. Please try again.");
    console.error(err);
  }
}

// ── DOM helpers ────────────────────────────────────────────────────────────
function appendMessage(role, text) {
  const wrap = document.createElement("div");
  wrap.className = `msg-wrap ${role}`;

  const avatar = document.createElement("div");
  avatar.className = `msg-avatar ${role}`;
  avatar.innerHTML = role === "user"
    ? '<i class="bi bi-person-fill"></i>'
    : '<i class="bi bi-shield-heart"></i>';

  const bubble = document.createElement("div");
  bubble.className = `msg-bubble ${role}`;
  bubble.textContent = text;

  if (role === "user") {
    wrap.appendChild(bubble);
    wrap.appendChild(avatar);
  } else {
    wrap.appendChild(avatar);
    wrap.appendChild(bubble);
  }

  const msgs = document.getElementById("chat-messages");
  msgs.appendChild(wrap);
  msgs.scrollTop = msgs.scrollHeight;
}

function showTyping() {
  const id   = "typing-" + Date.now();
  const wrap = document.createElement("div");
  wrap.className = "msg-wrap bot";
  wrap.id = id;
  wrap.innerHTML = `
    <div class="msg-avatar bot"><i class="bi bi-shield-heart"></i></div>
    <div class="msg-bubble bot" style="padding:10px 16px;">
      <div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    </div>`;
  document.getElementById("chat-messages").appendChild(wrap);
  document.getElementById("chat-messages").scrollTop = 99999;
  return id;
}

function removeTyping(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

// ── Agent Panel ────────────────────────────────────────────────────────────
function updateAgentPanel(agents, reason) {
  // Reset all
  Object.values(AGENT_IDS).forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.remove("active");
  });
  // Activate selected agents
  agents.forEach(name => {
    const id = AGENT_IDS[name];
    if (id) {
      const el = document.getElementById(id);
      if (el) el.classList.add("active");
    }
  });
  // Update reason
  const reasonEl = document.getElementById("orchestrator-reason");
  if (reasonEl && reason) {
    reasonEl.innerHTML = `<i class="bi bi-cpu-fill text-primary me-1"></i><span>${reason}</span>`;
  }
}

// ── Workflow Stepper ───────────────────────────────────────────────────────
function updateWorkflowSteps(agents) {
  const container = document.getElementById("workflow-steps");
  const pills     = container.querySelectorAll(".step-pill");
  pills.forEach((p, i) => {
    p.classList.toggle("active", i < 5); // light up all when active
  });
}

// ── Risk Panel ─────────────────────────────────────────────────────────────
function updateRiskPanel(risk) {
  const level = risk.risk_level || "Low Risk";
  const score = risk.risk_score || 0;
  const cls   = level === "High Risk" ? "high" : level === "Moderate Risk" ? "moderate" : "low";
  const fillCls = `risk-${cls}`;

  document.getElementById("risk-panel").innerHTML = `
    <div class="d-flex justify-content-between align-items-center mb-2">
      <span class="section-label mb-0">Assessment Result</span>
      <span class="risk-badge ${cls}">${level}</span>
    </div>
    <div class="d-flex align-items-center gap-2 mb-1">
      <span style="font-size:0.8rem;color:var(--mg-muted);white-space:nowrap;">Risk Score</span>
      <div class="risk-bar-bg flex-grow-1">
        <div class="risk-bar-fill ${fillCls}" style="width:${score}%"></div>
      </div>
      <span style="font-size:0.8rem;font-weight:700;">${score}/100</span>
    </div>
    <div style="font-size:0.82rem;margin-top:6px;color:var(--mg-text);">${risk.explanation || ""}</div>
    ${risk.next_steps ? `<div class="wellness-item mt-2"><i class="bi bi-arrow-right-circle me-1"></i>${risk.next_steps}</div>` : ""}
    ${level === "High Risk" ? `
      <div class="mt-2 p-2 rounded" style="background:#fee2e2;border-left:3px solid #dc2626;font-size:0.82rem;color:#7f1d1d;">
        <i class="bi bi-exclamation-triangle-fill me-1"></i>
        <strong>Immediate support is available.</strong> Please call <strong>988</strong> (USA) or text HOME to <strong>741741</strong>.
        You are not alone.
      </div>` : ""}
  `;
}

// ── Wellness Panel ─────────────────────────────────────────────────────────
function updateWellnessPanel(plan) {
  const lines   = plan.split("\\n").filter(l => l.trim());
  const content = lines.map(line => `<div class="wellness-item">${escHtml(line)}</div>`).join("");
  document.getElementById("wellness-panel").innerHTML = `
    <div class="section-label">Your Wellness Plan</div>
    ${content}
  `;
  // Switch to wellness tab
  const tab = document.querySelector('[href="#tab-wellness"]');
  if (tab) bootstrap.Tab.getOrCreateInstance(tab).show();
}

// ── Standalone Wellness Plan ───────────────────────────────────────────────
async function generateWellnessPlan() {
  const mood   = document.getElementById("mood-select").value;
  const stress = document.getElementById("stress-select").value;

  const btn = event.target;
  btn.disabled   = true;
  btn.innerHTML  = '<i class="bi bi-hourglass-split me-1"></i>Generating…';

  try {
    const res  = await fetch("/api/wellness", {
      method : "POST",
      headers: { "Content-Type": "application/json" },
      body   : JSON.stringify({ mood, stress_level: stress, emotional_state: mood }),
    });
    const data = await res.json();
    if (data.wellness_plan) updateWellnessPanel(data.wellness_plan);
  } catch (err) {
    console.error(err);
  } finally {
    btn.disabled  = false;
    btn.innerHTML = '<i class="bi bi-stars me-1"></i>Generate My Wellness Plan';
  }
}

// ── RAG Upload ─────────────────────────────────────────────────────────────
async function uploadRAG(input) {
  const file = input.files[0];
  if (!file) return;

  const statusEl = document.getElementById("rag-status");
  statusEl.innerHTML = `<i class="bi bi-hourglass-split me-1"></i>Uploading ${file.name}…`;

  const form = new FormData();
  form.append("file", file);

  try {
    const res  = await fetch("/api/rag/upload", { method: "POST", body: form });
    const data = await res.json();
    if (data.status === "success") {
      statusEl.innerHTML = `<i class="bi bi-check-circle-fill text-success me-1"></i>${data.message}`;
    } else {
      statusEl.innerHTML = `<i class="bi bi-x-circle-fill text-danger me-1"></i>${data.error}`;
    }
  } catch (err) {
    statusEl.innerHTML = `<i class="bi bi-x-circle-fill text-danger me-1"></i>Upload failed.`;
    console.error(err);
  }

  input.value = ""; // reset input
}

// ── Utilities ──────────────────────────────────────────────────────────────
function escHtml(str) {
  return str.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

// Load RAG status on page load
(async () => {
  try {
    const res  = await fetch("/api/rag/status");
    const data = await res.json();
    if (data.total_chunks > 0) {
      document.getElementById("rag-status").innerHTML =
        `<i class="bi bi-database-fill-check text-success me-1"></i>${data.total_chunks} chunks from: ${data.documents.join(", ")}`;
    }
  } catch (_) {}
})();
</script>
</body>
</html>"""

# =============================================================================
#  Application Entry Point
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  MindGuard AI – Mental Health Awareness & Suicide Prevention Agent")
    print("  Powered by IBM watsonx.ai Studio | IBM Granite Models")
    print("  Agentic AI Architecture | Flask Single-Page Application")
    print("=" * 70)
    print(f"  Granite Model  : {GRANITE_MODEL_ID}")
    print(f"  watsonx.ai URL : {WATSONX_URL}")
    print(f"  API Key Status : {'Configured [OK]' if WATSONX_API_KEY != 'your-api-key-here' else 'Not set (demo mode)'}")
    print(f"  PDF Support    : {'Yes (PyPDF2)' if PDF_SUPPORT else 'No (install PyPDF2)'}")
    print(f"  watsonx.ai SDK : {'Yes' if WATSONX_AVAILABLE else 'No (install ibm-watsonx-ai)'}")
    print("=" * 70)
    print("  Set environment variables before running:")
    print("    $env:WATSONX_API_KEY    = 'your-api-key'")
    print("    $env:WATSONX_PROJECT_ID = 'your-project-id'")
    print("    $env:WATSONX_URL        = 'https://us-south.ml.cloud.ibm.com'")
    print("=" * 70)
    print("  Starting Flask server at http://127.0.0.1:5000")
    print("=" * 70)
    app.run(host="0.0.0.0", port=5000, debug=True)
