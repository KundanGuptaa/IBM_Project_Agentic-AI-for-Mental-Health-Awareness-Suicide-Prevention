<div align="center">

# 🛡️ MindGuard AI
### Mental Health Awareness & Suicide Prevention Agent

**Powered by IBM watsonx.ai Studio · IBM Granite Models · Agentic AI Architecture**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)](https://flask.palletsprojects.com)
[![IBM watsonx.ai](https://img.shields.io/badge/IBM-watsonx.ai-0f62fe?logo=ibm&logoColor=white)](https://dataplatform.cloud.ibm.com)
[![IBM Granite](https://img.shields.io/badge/Model-IBM%20Granite-0043ce?logo=ibm&logoColor=white)](https://www.ibm.com/granite)
[![Bootstrap](https://img.shields.io/badge/UI-Bootstrap%205-7952b3?logo=bootstrap&logoColor=white)](https://getbootstrap.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> A complete Agentic AI application for mental health awareness and suicide prevention,  
> demonstrating IBM watsonx.ai Studio, IBM Granite Models, multi-agent orchestration,  
> and a lightweight RAG system — all in a **single `app.py` file**.

---

*Suitable for IBM SkillsBuild · watsonx.ai Studio Demos · Hackathons · Academic Projects*

</div>

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [Architecture](#-architecture)
4. [The Five Specialized AI Agents](#-the-five-specialized-ai-agents)
5. [Master Orchestrator Agent](#-master-orchestrator-agent)
6. [Lightweight RAG System](#-lightweight-rag-system)
7. [IBM watsonx.ai Integration](#-ibm-watsonxai-integration)
8. [Flask API Reference](#-flask-api-reference)
9. [User Interface](#-user-interface)
10. [Project Structure](#-project-structure)
11. [Prerequisites](#-prerequisites)
12. [Installation & Setup](#-installation--setup)
13. [Configuration](#-configuration)
14. [Running the Application](#-running-the-application)
15. [Demo Mode](#-demo-mode)
16. [Troubleshooting](#-troubleshooting)
17. [Crisis Resources](#-crisis-resources)
18. [Disclaimer](#-disclaimer)

---

## 🎯 Project Overview

**MindGuard AI** is an intelligent mental health assistant built as a demonstration of **Agentic AI** principles using IBM's enterprise AI platform. It showcases how multiple specialized AI agents can collaborate under a master orchestrator to deliver a sophisticated, empathetic, and safety-aware mental health support experience.

The entire application — backend logic, all five agents, the RAG system, and the full Bootstrap 5 frontend — is contained in a **single `app.py` file** with no database, no external APIs beyond IBM watsonx.ai, and no separate HTML files.

### What It Does

| Capability | Description |
|---|---|
| 🧠 **Mental Health Education** | Explains anxiety, depression, stress, burnout, mindfulness, and self-care |
| 💙 **Empathetic Support** | Provides non-judgmental, warm conversational emotional support |
| 🔍 **Risk Detection** | Analyses text for distress signals and classifies Low / Moderate / High Risk |
| 🌿 **Wellness Planning** | Generates personalised wellness plans with breathing, journaling, and sleep tips |
| 🤝 **Crisis Connection** | Surfaces crisis hotlines, professional resources, and support organisations |
| 📚 **RAG Knowledge Base** | Augments responses with user-uploaded PDF/TXT documents |

---

## ✨ Key Features

- **Single-file application** — entire project lives in `app.py`
- **IBM Granite Models** — all AI reasoning powered by IBM's enterprise LLMs
- **5-Agent Agentic Architecture** — specialized agents with a master orchestrator
- **Auto model selection** — automatically tries all available Granite models in your instance
- **Lightweight RAG** — in-memory keyword-overlap retrieval from uploaded documents
- **Demo mode** — fully functional rule-based fallback when credentials are not set
- **Bootstrap 5 SPA** — modern, responsive, calm-palette single-page UI
- **Live agent visualization** — shows which agents are active and why
- **Risk dashboard** — animated risk score bar with color-coded levels
- **Zero dependencies beyond core packages** — no database, no vector store

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MindGuard AI                             │
│                   Flask Single-Page App                         │
└─────────────────────────┬───────────────────────────────────────┘
                          │  HTTP (JSON)
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Master Orchestrator Agent                       │
│   Receives query → Analyses keywords → Routes to agent(s)       │
│   Aggregates outputs → Returns unified response                 │
└──┬──────────┬───────────┬──────────────┬────────────────────────┘
   │          │           │              │              │
   ▼          ▼           ▼              ▼              ▼
┌──────┐ ┌──────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│Agent │ │Agent │  │ Agent 3  │  │ Agent 4  │  │ Agent 5  │
│  1   │ │  2   │  │ Distress │  │ Wellness │  │ Support  │
│Aware │ │Emot. │  │Detection │  │  Agent   │  │Connector │
│-ness │ │Supp. │  │          │  │          │  │  Agent   │
└──┬───┘ └──┬───┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
   │        │           │              │              │
   └────────┴───────────┴──────────────┴──────────────┘
                          │
                          ▼
           ┌──────────────────────────┐
           │   IBM watsonx.ai Studio  │
           │   IBM Granite Models     │
           │   (auto-selected)        │
           └──────────────────────────┘
                          │
                          ▼
           ┌──────────────────────────┐
           │   Lightweight RAG System │
           │   (in-memory, optional)  │
           └──────────────────────────┘
```

---

## 🤖 The Five Specialized AI Agents

### Agent 1 — Mental Health Awareness Agent
**Function:** [`awareness_agent()`](app.py)

Educates users about mental health topics using IBM Granite. Covers anxiety, depression, stress, burnout, emotional wellness, mindfulness, and self-care. Integrates with the RAG system to augment responses with knowledge-base documents.

**Triggered by:** Questions containing *"what is", "how does", "explain", "signs of", "symptoms"*

---

### Agent 2 — Emotional Support Agent
**Function:** [`emotional_support_agent()`](app.py)

Provides warm, empathetic, non-judgmental conversational support. Acknowledges emotions, encourages healthy coping, and never diagnoses. Every response ends with the standard care disclaimer.

**Triggered by:** Distress keywords, wellness requests, crisis situations, and as default fallback

---

### Agent 3 — Distress Detection Agent
**Function:** [`distress_detection_agent()`](app.py) → [`detect_risk()`](app.py)

Analyses message text for distress indicators using IBM Granite's reasoning capabilities. Returns structured JSON with:

| Field | Description |
|---|---|
| `risk_level` | `Low Risk` / `Moderate Risk` / `High Risk` |
| `risk_score` | Integer 0–100 |
| `explanation` | One or two sentence summary |
| `next_steps` | Recommended action |

Falls back to a keyword-scoring algorithm if the model doesn't return valid JSON.

---

### Agent 4 — Prevention & Wellness Agent
**Function:** [`wellness_agent()`](app.py) → [`generate_wellness_plan()`](app.py)

Generates personalised daily wellness plans using IBM Granite. Plans include:
- Morning breathing or meditation routine
- Midday check-in activity
- Evening journaling prompt
- Sleep hygiene tip
- Positive affirmation

**Input parameters:** `mood`, `stress_level`, `emotional_state`

---

### Agent 5 — Human Support Connector Agent
**Function:** [`support_connector_agent()`](app.py)

Returns structured crisis resources and professional support recommendations. Provides:
- Crisis hotlines (USA, UK, India, Global)
- Trusted online mental health resources
- Professional support pathways
- Risk-level-appropriate messaging

Always reminds users that professional care is available and never claims to replace it.

---

## 🎛️ Master Orchestrator Agent

**Function:** [`orchestrate_agents()`](app.py)

The brain of the system. Receives every user message, analyses it for intent signals, decides which agents to activate, calls them in sequence, and assembles a unified response.

### Routing Logic

| Signal Detected | Agents Activated |
|---|---|
| Crisis / suicide keywords | Distress Detection + Support Connector + Emotional Support |
| Distress / hopelessness | Distress Detection + Emotional Support + Wellness |
| Informational question | Mental Health Awareness |
| Wellness / coping request | Wellness + Emotional Support |
| General message (default) | Emotional Support |

### Response Schema

```json
{
  "agents_activated":     ["Agent Name", ...],
  "orchestrator_reason":  "Why these agents were selected",
  "primary_response":     "Main AI-generated response text",
  "risk_data": {
    "risk_level":  "Low Risk | Moderate Risk | High Risk",
    "risk_score":  0,
    "explanation": "...",
    "next_steps":  "..."
  },
  "wellness_plan":  "Personalised wellness plan text",
  "resources":      { "crisis_hotlines": [...], "online_resources": [...] },
  "timestamp":      "2025-01-01 12:00:00"
}
```

---

## 📚 Lightweight RAG System

MindGuard AI includes a built-in Retrieval-Augmented Generation system that requires no vector database.

### How It Works

```
Upload Document (PDF/TXT)
        │
        ▼
  Text Extraction
        │
        ▼
  Chunking (word-window with ~16% overlap)
        │
        ▼
  In-Memory Store  [{ source, chunk }, ...]
        │
        ▼
  User Query → Keyword-Overlap Scoring → Top-K Chunks
        │
        ▼
  Context prepended to Granite prompt
```

### Supported Document Types

| Type | Support | Notes |
|---|---|---|
| `.txt` | ✅ Always | Built-in, no extra packages |
| `.pdf` | ✅ Optional | Requires `PyPDF2` |

### Suggested Documents to Upload

- WHO Mental Health Guidelines
- DSM-5 Summary documents
- Coping Strategy guides
- Crisis Intervention resources
- Mental Health Awareness literature

### Key Functions

| Function | Description |
|---|---|
| `ingest_document(filename, text)` | Chunks and stores document in memory |
| `retrieve_context(query, top_k=3)` | Returns most relevant chunks for a query |
| `_chunk_text(text, chunk_size=500)` | Splits text into overlapping word windows |

---

## ☁️ IBM watsonx.ai Integration

### Credentials

The app reads credentials from environment variables (or a `.env` file):

| Variable | Description | Required |
|---|---|---|
| `WATSONX_API_KEY` | IBM Cloud API Key | ✅ Yes |
| `WATSONX_PROJECT_ID` | watsonx.ai Project ID | ✅ Yes |
| `WATSONX_URL` | Service endpoint URL | ✅ Yes |

### Model Auto-Selection

The app tries Granite models in this priority order until one succeeds:

```python
GRANITE_MODEL_CANDIDATES = [
    "ibm/granite-4-h-small",       # newest Granite (2025)
    "ibm/granite-3-1-8b-base",     # Granite 3.1 8B base
    "ibm/granite-8b-code-instruct", # code-capable Granite
    "ibm/granite-3-3-8b-instruct",  # original target
]
```

### Key Integration Functions

| Function | API Used | Purpose |
|---|---|---|
| `_init_granite_model()` | `Credentials` + `ModelInference` | Lazy init with auto-selection |
| `generate_response(prompt)` | `ModelInference.chat()` | General text generation |
| `detect_risk(text)` | `ModelInference.chat()` | Structured JSON risk assessment |
| `generate_wellness_plan(...)` | `ModelInference.chat()` | Personalised wellness plan |

> **Note:** The app uses the modern `/ml/v1/text/chat` API endpoint (`model.chat()`) to avoid deprecation warnings from the older `generate_text()` endpoint.

---

## 🌐 Flask API Reference

| Method | Endpoint | Description | Request Body |
|---|---|---|---|
| `GET` | `/` | Serves the Bootstrap 5 SPA | — |
| `POST` | `/api/chat` | Main orchestrator endpoint | `{ "message": "string" }` |
| `POST` | `/api/distress` | Standalone risk detection | `{ "text": "string" }` |
| `POST` | `/api/wellness` | Generate wellness plan | `{ "mood": "...", "stress_level": "...", "emotional_state": "..." }` |
| `GET` | `/api/resources` | Get support resources | `?risk_level=Low Risk` |
| `POST` | `/api/rag/upload` | Upload document to RAG | `multipart/form-data` file |
| `GET` | `/api/rag/status` | RAG store statistics | — |

---

## 🖥️ User Interface

The Bootstrap 5 SPA is served entirely via `render_template_string()` — no separate HTML files.

### UI Panels

| Panel | Location | Contents |
|---|---|---|
| **Chat Interface** | Centre | Multi-turn conversation, quick-prompt buttons, typing indicator |
| **Agent Orchestration Panel** | Left | Live agent highlighting, orchestrator reasoning text |
| **Orchestrator Workflow** | Left | Step-by-step flow visualization (animated on response) |
| **RAG Knowledge Base** | Left | Drag-and-drop document upload, chunk count display |
| **Risk Detection Panel** | Centre-bottom | Risk badge, animated progress bar, explanation, next steps |
| **Wellness Plan Tab** | Right | IBM Granite wellness plan + quick mood/stress generator |
| **Support Resources Tab** | Right | Crisis hotlines, online resources, professional referrals |
| **watsonx.ai Info Card** | Right | Model ID, architecture summary, credential setup hint |

### Quick-Prompt Buttons

The chat interface includes one-click example prompts:
- *"What is anxiety?"*
- *"I feel overwhelmed"*
- *"Breathing exercises"*
- *"Signs of depression"*

---

## 📁 Project Structure

```
IBM_BOB_Agent_DEMO_Project01/
│
├── app.py          ← Entire application (single file)
├── .env            ← Credentials (gitignored, never commit)
├── README.md       ← This file
│
└── .bob/           ← Bob AI agent workspace (auto-generated)
```

> The entire project is intentionally a **single-file application** to keep it portable, easy to share, and straightforward to run for demos and academic presentations.

---

## ✅ Prerequisites

| Requirement | Minimum Version | Notes |
|---|---|---|
| Python | 3.9+ | 3.11+ recommended |
| IBM Cloud Account | — | Free tier available |
| IBM watsonx.ai Project | — | Must be linked to a WML instance |
| IBM Cloud API Key | — | From IAM → API Keys |

---

## 📦 Installation & Setup

### Step 1 — Clone or download the project

```bash
git clone https://github.com/your-org/mindguard-ai.git
cd mindguard-ai
```

### Step 2 — Install Python dependencies

```bash
pip install flask ibm-watsonx-ai python-dotenv PyPDF2
```

Or install all at once using the recommended command:

```bash
python -m pip install flask ibm-watsonx-ai python-dotenv PyPDF2
```

> Always use `python -m pip` to ensure packages install into the same Python that runs `app.py`.

### Step 3 — Configure credentials

Create a `.env` file in the project root (see [Configuration](#-configuration) below).

### Step 4 — Run

```bash
python app.py
```

Open your browser at **http://127.0.0.1:5000**

---

## ⚙️ Configuration

Create a `.env` file in the project root directory:

```env
WATSONX_API_KEY=your-ibm-cloud-api-key
WATSONX_PROJECT_ID=your-watsonx-project-id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

### How to get these values

#### WATSONX_API_KEY
1. Log in to [cloud.ibm.com](https://cloud.ibm.com)
2. Go to **Manage → Access (IAM) → API Keys**
3. Click **Create an IBM Cloud API key**
4. Copy the key immediately (it is shown only once)

#### WATSONX_PROJECT_ID
1. Open [dataplatform.cloud.ibm.com](https://dataplatform.cloud.ibm.com)
2. Click **Projects** and open your project (or use **Sandbox**)
3. Go to **Manage → General**
4. Copy the **Project ID** (UUID format)

#### WATSONX_URL
Use the endpoint for your IBM Cloud region:

| Region | URL |
|---|---|
| US South (Dallas) | `https://us-south.ml.cloud.ibm.com` |
| EU (Frankfurt) | `https://eu-de.ml.cloud.ibm.com` |
| UK (London) | `https://eu-gb.ml.cloud.ibm.com` |
| Japan (Tokyo) | `https://jp-tok.ml.cloud.ibm.com` |

### Linking a WML Instance (required)

Your watsonx.ai project must be associated with a Watson Machine Learning service instance:

1. Open your project in [dataplatform.cloud.ibm.com](https://dataplatform.cloud.ibm.com)
2. Click **Manage** tab → **Services & integrations**
3. Click **Associate service** → select **Watson Machine Learning**
4. Choose an existing WML instance or create a free **Lite** tier instance
5. Click **Associate**

---

## 🚀 Running the Application

```bash
python app.py
```

**Expected console output:**

```
======================================================================
  MindGuard AI – Mental Health Awareness & Suicide Prevention Agent
  Powered by IBM watsonx.ai Studio | IBM Granite Models
======================================================================
  Granite Model  : ibm/granite-4-h-small
  watsonx.ai URL : https://us-south.ml.cloud.ibm.com
  API Key Status : Configured [OK]
  PDF Support    : Yes (PyPDF2)
  watsonx.ai SDK : Yes
======================================================================
[INFO] .env file loaded via python-dotenv.
[INFO] IBM Granite model 'ibm/granite-4-h-small' initialized successfully.
 * Running on http://127.0.0.1:5000
```

---

## 🎮 Demo Mode

MindGuard AI runs in **Demo Mode** automatically when:
- `WATSONX_API_KEY` is not set, or
- `ibm-watsonx-ai` SDK is not installed

In Demo Mode:
- All 5 agents remain active with intelligent rule-based responses
- The full UI, all panels, agent visualization, and RAG uploader work normally
- Responses are clearly marked with `[Demo Mode – Configure WATSONX_API_KEY for IBM Granite responses]`
- No API calls are made — fully offline capable

This makes the app safe to share and demonstrate without exposing credentials.

---

## 🔧 Troubleshooting

### `ModuleNotFoundError: No module named 'dotenv'`
```bash
python -m pip install python-dotenv
```
> Use `python -m pip` (not just `pip`) to ensure it installs into the correct Python.

### `ModuleNotFoundError: No module named 'ibm_watsonx_ai'`
```bash
python -m pip install ibm-watsonx-ai
```

### `403 – project_id is not associated with a WML instance`
Your watsonx.ai project needs a Watson Machine Learning service linked to it.  
Follow the [Linking a WML Instance](#linking-a-wml-instance-required) steps above.

### `Model 'ibm/granite-X' is not supported for this environment`
The app automatically tries all Granite model candidates and selects the first available one. Check the console output for `[INFO] IBM Granite model '...' initialized successfully.` to see which model was selected.

### `401 Unauthorized / API key not found`
- Verify your API key is active: [cloud.ibm.com/iam/apikeys](https://cloud.ibm.com/iam/apikeys)
- Ensure there are no extra spaces or quotes in your `.env` file
- Confirm the key has at least **Viewer** IAM role on the WML instance

### App starts but UI is blank
- Make sure Flask is running (`python app.py`)
- Open **http://127.0.0.1:5000** (not https)
- Check the terminal for any Python errors

---

## 📞 Crisis Resources

> **If you or someone you know is in crisis, please reach out immediately.**

| Organisation | Contact | Availability |
|---|---|---|
| **988 Suicide & Crisis Lifeline (USA)** | Call or text **988** | 24/7 |
| **Crisis Text Line (USA)** | Text **HOME** to **741741** | 24/7 |
| **Samaritans (UK)** | **116 123** | 24/7, free |
| **iCall (India)** | **9152987821** | Mon–Sat 8am–10pm |
| **Vandrevala Foundation (India)** | **1860-2662-345** | 24/7 |
| **IASP Global Directory** | [iasp.info/resources/Crisis_Centres](https://www.iasp.info/resources/Crisis_Centres/) | Global |

**Online Resources:**
- [WHO Mental Health](https://www.who.int/health-topics/mental-health)
- [NAMI – National Alliance on Mental Illness](https://www.nami.org)
- [Mind (UK)](https://www.mind.org.uk)
- [Mental Health Foundation](https://www.mentalhealth.org.uk)

---

## ⚠️ Disclaimer

**MindGuard AI is an educational and awareness tool only.**

- It is **not** a substitute for professional medical, psychological, or psychiatric care.
- It is **not** a crisis intervention service.
- It should **not** be used as the sole resource in a mental health emergency.
- All AI-generated content is for informational and educational purposes only.
- If you are experiencing a mental health crisis, please contact emergency services or a crisis helpline immediately.

This project is intended for educational demonstrations, academic research, IBM SkillsBuild learning, and AI hackathon showcases.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

Built with ❤️ using **IBM watsonx.ai Studio** and **IBM Granite Models**  
*MindGuard AI · Agentic AI for Mental Health Awareness*

</div>
