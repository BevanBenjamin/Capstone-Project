# 🚄 Railway Info System – AI Powered Travel Assistant

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Gemini AI](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=google&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

A smart, conversational **Indian Railway assistant** built using **FastAPI**, **Gemini AI**, **Speech-to-Text**, and a fully responsive modern UI. This system delivers on-demand updates in text and speech, designed to work efficiently even in noisy environments.

---
Visit the website :[Railway Info System](https://railinfo-stcb.onrender.com/ui)


---

## 📑 Table of Contents
- [🌟 Overview](#-overview)
- [🎯 Key Features](#-key-features)
- [🛠️ Tech Stack](#️-tech-stack)
- [📂 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
- [🔌 API Endpoints](#-api-endpoints)
- [📸 Screenshots](#-screenshots)
- [🤝 Contributors](#-contributors)
- [📬 Contact](#-contact)

---

## 🌟 Overview

The **Railway Info System** is an intelligent, voice-enabled chatbot designed to democratize access to live railway data. By bridging the gap between complex railway databases and everyday users, it offers a natural language interface for checking train status, seat availability, and emergency reporting.

**Why this project?**
- **Multilingual Support:** Breaks language barriers for diverse travelers.
- **Voice-First Design:** Accessible for users who prefer speaking over typing.
- **Real-Time Accuracy:** Fetches live data directly via APIs.

---

## 🎯 Key Features

### 🤖 AI Chatbot (Voice & Text)
- **Natural Conversations:** Powered by Google Gemini to understand context and intent.
- **Speech-to-Text:** Speak your query comfortably.
- **Text-to-Speech (TTS):** Hear the response, perfect for on-the-go usage.

### 🚉 Comprehensive Train Services
- **🔍 Search Trains:** Find trains between any two stations.
- **🪑 Seat Availability:** Check real-time seat counts for your preferred class.
- **🚦 Live Status:** Track where your train is instantly.
- **📅 PNR Status:** Quick lookup for ticket confirmation.

### 🛡️ Safety & Support
- **⚠️ Emergency Reporting:** Instantly report issues; the system logs them and sends an **automated email alert** to administrators.
- **📩 Contact Form:** Direct line of communication with email notifications and spam protection.

### 🌐 User Experience
- **Multilingual:** Auto-detects and responds in English or Hindi.
- **Modern UI:** Built with TailwindCSS for a smooth, mobile-responsive experience.

---

## 🛠️ Tech Stack

| Component | Technologies Used |
|-----------|-------------------|
| **Frontend** | HTML5, JavaScript (ES6+), TailwindCSS, FontAwesome |
| **Backend** | Python, FastAPI, Uvicorn |
| **AI & NLP** | Google Gemini API, Google Speech-to-Text, LangDetect |
| **Audio Processing** | Pydub, FFmpeg, Google TTS |
| **Utilities** | Geopy, Dotenv, SMTP (Email) |

---

## 📂 Project Structure

```bash
Capstone-Project/
│
├── main.py                 # 🚀 Entry point: FastAPI backend & AI logic
├── index.html              # 🎨 Main User Interface
├── requirements.txt        # 📦 Python dependencies
├── build.sh                # 🛠️ Build script (for deployment)
├── runtime.txt             # ⚙️ Python runtime version
│
├── static/                 # 🖼️ Static assets (CSS, Images, JS)
│
├── data/                   # 💾 Data storage
│
├── emergency_reports.log   # 📝 Auto-generated log for emergencies
└── contacts.log            # 📝 Auto-generated log for contact forms
```
---



## 🚀 **Getting Started**



Follow these steps to set up the project locally.



### 1️⃣ Clone the Repository

```bash

git clone [https://github.com/BevanBenjamin/Capstone-Project.git](https://github.com/BevanBenjamin/Capstone-Project.git)

cd Capstone-Project
```



### 2️⃣ Install Dependencies

Ensure you have Python installed. Then run:



```bash

pip install -r requirements.txt
```



##Note: You may need ffmpeg installed on your system for audio processing.



### 3️⃣ Configure Environment Variables

Create a .env file in the root directory and add your credentials:



### 4️⃣ Run the Application

Start the FastAPI server:



```bash

uvicorn main:app --reload
```



### 5️⃣ Access the UI

Open your browser and navigate to: 👉 http://localhost:8000/ui



## API Endpoints

The backend exposes several key endpoints for the frontend to consume:



- **chatbot** – Handles logic for voice/text queries via Gemini.
- **seat-availability** – Fetches live seat data.
- **live-status** – Tracks real-time train location.
- **contact** – Processes contact form submissions.
- **report-emergency** – Critical endpoint for logging and emailing emergency reports.


---

## 📸 ScreenShots
![Main Page](https://github.com/user-attachments/assets/3fb12811-9210-49e0-be3a-8b9728c25fe1
)
![Features](https://github.com/user-attachments/assets/c3430e13-6a25-408f-b802-68575a19749d
)
![ChatBot](https://github.com/user-attachments/assets/db0cf213-a132-4ccd-9b33-9c62f5d77941
)
---

## 🤝 Contributors

**Krishna Kumar Jha** - AI ML Developer , Web Developer

**Bevan Benjamin** - Developer

**Harsh D Salian** - Developer


---

 ## 📬Contact

Have questions or want to contribute? Reach out!



<p align="center"> <i>Made with ❤️ for better railway journeys.</i> </p>


