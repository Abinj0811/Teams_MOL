# Teams_MOL ChatBot

This repo hosts a Teams-ready RAG chatbot. The active code lives in `ChatBot/`; the `chatApp/` folder contains older UI pieces.

## What’s inside (`ChatBot/`)
- `app.py` – aiohttp entrypoint exposing `/api/messages` plus simple debug/health endpoints.
- `bots/ThinkpalmRAGBot.py` – Bot Framework bot wiring the RAG pipeline.
- `graph/self_rag_graph.py` & `nodes/` – LangGraph-style nodes for query rewrite, retrieval, answer, regenerate, etc.
- `rag/thinkpalm_rag.py` – Retrieval + prompt orchestration for the bot.
- `models/chat_state.py` – Minimal conversation state model.
- `utils/` – helpers (logging, abbreviation expansion, chunk enrichment) and `shipping_glossary.json`.
- `main_graph_runner_copy.py` – quick script for running the graph locally.
- `config.py` – port/app ID settings (reads from `MicrosoftAppId`/`MicrosoftAppPassword` env vars).

## Quick start
1) `python3 -m venv .venv && source .venv/bin/activate`
2) Install deps: `pip install -r requirements.txt` and `pip install aiohttp botbuilder-core botbuilder-integration-aiohttp`
3) Set bot creds if needed: `export MicrosoftAppId=... MicrosoftAppPassword=...`
4) Run the bot: `cd ChatBot && python app.py` (defaults to port 3979)

## Work done 
1. Converted all modules to lanngraph 
2. App in Teams can be up by 
    a) Run python app.py
    b) Up ngrok server ( $http ngrok port_no) where port no is specified in app.py
    c) deploy manifest.yaml in Teams 
        update manifest.yaml : application id 
        zip the folder 
        upload in Teams
3. Tested all questions using "docs/InquiryList_MOLCT-GPM R2 - Testing by Babu 1.ods"
4. Added shipping glossery and full forms 
5. Added footer notes (in answer node). While formatting the answer is changing that needs a fix 

``Note : Add app configurations in config.py``
