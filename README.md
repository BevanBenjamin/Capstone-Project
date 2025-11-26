🚄 Railway Info System – AI Powered Travel Assistant

A smart, conversational Indian Railway assistant built using FastAPI, Gemini AI, Speech-to-Text, Real-time APIs, and a fully responsive modern UI.

🌟 Overview

Railway Info System is an intelligent, voice-enabled chatbot designed to help users access live railway data with ease.
It supports:

🎤 Voice & text chat

🚆 Live Train Status

🪑 Seat Availability

🎟️ Ticket Search

📍 Railway station information

⚠️ Emergency reporting

📩 Contact form with email notifications

🌐 Multilingual support

🤖 Smart AI replies via Google Gemini

The goal is to make railway information quick, accessible, conversational, and accurate, with a clean UX suitable for real-world deployment.

🎯 Features
🤖 AI Chatbot (Text + Voice)

Converts speech → text

Sends query to Gemini AI

Executes internal tools (train search, seat availability, live status)

Responds in natural language

🚉 Train Services

🔍 Search Trains

🪑 Check Seat Availability

🚦 Live Train Status

📅 PNR Status Lookup

🗺️ Station Information

🛡️ Emergency Reporting

Users can instantly report an emergency

Your system sends a real email alert to admin using Gmail SMTP

All reports logged to a server file for backup

💬 Contact Form With Email Alerts

Sends messages directly to your email

Logs every submission

Clean UI feedback

Spam-safe fallback

🔊 Voice Interaction

Users can click microphone

Speak their query

Gemini processes it

System replies via text + TTS (Text-to-Speech)

🌐 Multilingual Interface

English / Hindi toggle

Auto language detection

Gemini responds in the user’s language

🎨 Beautiful Frontend UI

Modern colors

Smooth animations

Floating chatbot

Emergency panel

Responsive & mobile-friendly

🛠️ Tech Stack
Frontend

HTML5

TailwindCSS

JavaScript

FontAwesome Icons

Fetch API

Responsive UI components

Backend

FastAPI (Python)

Uvicorn

Python Multipart

Pydantic

AI & Processing

Google Gemini API

Google Speech-to-Text

Google Text-to-Speech

Pydub

FFmpeg

Utilities

Requests

Dotenv

LangDetect

Geopy

📂 Project Structure
Capstone-Project/
│── main.py                # FastAPI backend + AI tool calling + endpoints
│── index.html             # Railway assistant UI
│── requirements.txt       # Project dependencies
│── static/                # Assets
│── emergency_reports.log  # Auto-generated emergency logs
│── contacts.log           # Auto-generated contact submissions
└── README.md              # ← You are here!

🔧 Core Endpoints
🟦 /chatbot/

Handles voice & text queries using Gemini function-calling.

🟦 /seat-availability/

Returns real-time seat availability (via external API).

🟦 /live-status/

Retrieves live train running status.

🟦 /contact/

Sends user messages directly to admin email.

🟦 /report-emergency/

Sends emergency details to email + logs them.

✨ Key Highlights
⚡ Real-Time Railway Data

Using external APIs + Gemini tools.

🎙️ Full Voice Pipeline

Speech → AI → Response → Optional TTS voice output.

💼 Production-Ready Email Integration

Contact form + emergency reports sent to your inbox.

🔄 Clean & Structured Code

Modular functions

Tool-based AI execution

Clear logging

Strong error handling

📸 Screenshots (Optional – Add Later)

You can add screenshots like:

/screenshots/
   home-page.png
   chatbot.png
   emergency-form.png
   train-results.png

🚀 How to Run Locally
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Add .env File
GOOGLE_API_KEY=your_api_key
PERSONAL_EMAIL=your_email@gmail.com
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587

3️⃣ Start Server
uvicorn main:app --reload

4️⃣ Open Website

➡️ Visit: http://localhost:8000/ui


🤝 Contributors

👨‍💻 Krishna Kumar Jha
B.Tech Student, Information Science & Technology
Presidency University, Bangalore
Developer – AI/ML, Web, Backend,

👨‍💻Bevan Benjamin
B.Tech Student, Information Science & Technology
Presidency University, Bangalore

👨‍💻Harsh D Salian
B.Tech Student, Information Science & Technology
Presidency University, Bangalore


⭐ Like this project? Give it a star!

If this project helped or inspired you, consider giving it a ⭐ on GitHub — it motivates further improvements!

📬 Contact

Want to collaborate, improve features, or hire for dev work?

📧 krishna7kumarjha@gmail.comurl)
📱 Open for contributions / feature requests
