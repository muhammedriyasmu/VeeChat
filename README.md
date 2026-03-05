🚀 VeeChat – Chat with YouTube Videos using RAG
---------------------------------------------

VeeChat is an AI-powered application that allows users to ask questions about YouTube videos using natural language.
The system uses Retrieval-Augmented Generation (RAG) to extract knowledge from video transcripts and generate accurate responses with a Large Language Model.

✨ Features
----------------

📺 YouTube Video Ingestion – Paste a YouTube URL

📝 Automatic Transcript Extraction

🔍 Semantic Search using Vector Embeddings

🤖 LLM-based Question Answering

🎥 Video Recommendation System

⚡ Groq LLM Integration for fast responses

💬 Simple chat interface


🧠 How It Works
------------------------

The system follows a RAG (Retrieval-Augmented Generation) pipeline:

User provides a YouTube video URL

The system extracts the transcript

Transcript is split into smaller chunks

Chunks are converted into vector embeddings

Relevant chunks are retrieved using semantic similarity

Retrieved context is passed to the LLM

The model generates a context-aware answer

This approach helps reduce hallucinations and improves answer accuracy by grounding the model with real data.

🛠 Tech Stack
---------------------

Python

Streamlit – User Interface

Embedchain – RAG pipeline

Groq LLM

YouTube Transcript API

Vector Embeddings

⚙️ Installation
------------------
1️⃣ Clone the repository

git clone https://github.com/muhammedriyasmu/VeeChat.git

cd VeeChat

2️⃣ Create a virtual environment

python -m venv venv

Activate it:

Windows

venv\Scripts\activate

Mac/Linux

source venv/bin/activate

3️⃣ Install dependencies

pip install -r requirements.txt

4️⃣ Add API Keys

Create a .env file in the root folder.

GROQ_API_KEY=your_groq_api_key

▶️ Run the Application

streamlit run streamlit_app.py

Open the app:

http://localhost:8501


🎓 Learning Outcomes
----------------------

Through this project, I explored:

Retrieval-Augmented Generation (RAG)

Embeddings & semantic search

Prompt engineering

LLM integration with real-world data

Debugging AI system pipelines


👨‍💻 Author
-----------------

Muhammed Riyas M.U

MSc Computer Science – MES Keveeyam College Valanchery

🌐 Portfolio  
https://muhammedriyasmu.github.io/
