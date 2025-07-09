# 🧠 LexiSmart Backend

LexiSmart is an AI-powered educational platform designed to support dyslexic learners by simplifying academic content and generating real-time mind maps, summaries, and speech synthesis using state-of-the-art language models.

This is the Flask backend that powers LexiSmart.

---

## 🚀 Features

- ✅ GPT-4o–powered summarization optimized for readability (Flesch ≥ 85)
- 🧠 Named Entity Recognition + Concept Expansion (ConceptNet, Wikidata, DBpedia)
- 🔊 Text-to-Speech using ElevenLabs API
- 🗺️ Dynamic Mind Map generation using `networkx` and `spaCy`
- 📊 Composite scoring pipeline using ROUGE-L, BERTScore, and Flesch Reading Ease (research mode)
- 🌐 CORS enabled for full-stack integration

---

## 🧩 Tech Stack

- Python, Flask, spaCy, NetworkX
- OpenAI GPT-4o
- ElevenLabs (Text-to-Speech)
- ConceptNet / Wikidata / DBpedia APIs
- Render (Deployment)
- `textstat`, `bert_score`, `rouge_score`, `pandas`, `matplotlib`

---

## 🛠 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/lexismart-backend.git
cd lexismart-backend
