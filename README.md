# Railway Info System

**An AI-powered Indian Railway assistant — voice, text, and multilingual.**

Live deployment → [railinfo-stcb.onrender.com/ui](https://railinfo-stcb.onrender.com/ui)

![Python](https://img.shields.io/badge/Python-3.9%2B-0f0c29?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0f0c29?style=flat-square&logo=fastapi&logoColor=white)
![Gemini AI](https://img.shields.io/badge/Gemini_AI-0f0c29?style=flat-square&logo=google&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-0f0c29?style=flat-square&logo=tailwind-css&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-0f0c29?style=flat-square)

---

## Overview

Railway Info System bridges the gap between complex railway data and everyday travelers through a natural language interface. Ask a question in English or Hindi — by voice or text — and get a direct, spoken or written answer.

Built as a capstone project to explore real-world integration of AI, speech processing, and REST APIs at scale.

---

## Features

**AI Chatbot (Voice + Text)**
- Conversational queries powered by Google Gemini
- Speech-to-Text input and Text-to-Speech response output
- Auto language detection — responds in English or Hindi

**Train Services**
- Search trains between any two stations
- Real-time seat availability by class
- Live train tracking
- PNR status lookup

**Safety & Reporting**
- Emergency reporting with automated email alerts to administrators
- Contact form with spam protection and email notifications

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5, JavaScript (ES6+), TailwindCSS, FontAwesome |
| Backend | Python, FastAPI, Uvicorn |
| AI & NLP | Google Gemini API, Google Speech-to-Text, LangDetect |
| Audio | Pydub, FFmpeg, Google TTS |
| Utilities | Geopy, python-dotenv, SMTP |

---

## Project Structure

```
railway-info-system/
├── main.py                  # FastAPI backend, AI logic, and route handlers
├── index.html               # Main UI
├── requirements.txt         # Python dependencies
├── build.sh                 # Deployment build script
├── runtime.txt              # Python runtime pin
├── static/                  # CSS, JS, and image assets
├── data/                    # Data storage
├── emergency_reports.log    # Auto-generated emergency log
└── contacts.log             # Auto-generated contact log
```

---

## Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/BevanBenjamin/Capstone-Project.git
cd Capstone-Project
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

> Note: FFmpeg is required for audio processing. Install it separately for your OS.

**3. Configure environment variables**

Create a `.env` file in the root directory and add your API keys and SMTP credentials.

**4. Run the server**
```bash
uvicorn main:app --reload
```

**5. Open the UI**

Navigate to `http://localhost:8000/ui` in your browser.

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `/chatbot` | Handles voice and text queries via Gemini |
| `/seat-availability` | Fetches live seat data by class |
| `/live-status` | Returns real-time train location |
| `/contact` | Processes contact form submissions |
| `/report-emergency` | Logs emergencies and triggers email alerts |

---

## Screenshots

**Main Interface**

![Main Page](https://github.com/user-attachments/assets/3fb12811-9210-49e0-be3a-8b9728c25fe1)

**Feature Overview**

![Features](https://github.com/user-attachments/assets/c3430e13-6a25-408f-b802-68575a19749d)

**Chatbot in Action**

![ChatBot](https://github.com/user-attachments/assets/db0cf213-a132-4ccd-9b33-9c62f5d77941)

---

## Contributors

| Name | Role |
|---|---|
| Krishna Kumar Jha | AI/ML Developer, Web Developer |
| Bevan Benjamin | Developer |
| Harsh D Salian | Developer |

---

## Disclaimer

This project was built as a learning exercise to explore AI integration, speech APIs, and full-stack development with FastAPI. It is not intended for production or critical use. The live deployment runs on a free-tier server, so API endpoints may occasionally experience cold starts or response delays. Do not rely on it for real-time travel decisions.

---

*Have questions or want to collaborate? Open an issue or reach out directly.*
