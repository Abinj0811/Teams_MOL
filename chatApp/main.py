import os
from pathlib import Path
from fastapi import FastAPI, Request, Form, Depends, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

from user_manager import UserManager, get_user_from_header
from processors import extract_text_from_file
from embeddings import FAISSEmbeddingHandler
from chains.retrievel_chain import ThinkpalmCosmosRAGmethod2
load_dotenv()

# --- Config ---

DATA_ROOT = Path("./data")
ADMIN_FOLDER = DATA_ROOT / "admin"
ADMIN_UPLOADS = ADMIN_FOLDER / "uploads"
ADMIN_UPLOADS.mkdir(parents=True, exist_ok=True)

COSMOS_ENDPOINT=os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY=os.getenv("COSMOS_KEY")
COSMOS_DATABASE=os.getenv("COSMOS_DATABASE")
COSMOS_CONTAINER=os.getenv("COSMOS_CONTAINER")
CHAT_CONTAINER = os.getenv("CHAT_CONTAINER")


app = FastAPI(title="RAG + Admin Knowledge Base Demo")

# Templates & static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# User manager
user_manager = UserManager(data_root=DATA_ROOT, embedding_backend="ollama")

# --- Models ---
class AskRequest(BaseModel):
    question: str

class ChatRequest(BaseModel):
    user_id: str
    message: str
# =========================
# Admin routes
# =========================

@app.get("/admin", response_class=HTMLResponse)
def admin_panel(request: Request):
    """Render admin panel with upload form and list of uploaded files"""
    files = [f.name for f in ADMIN_UPLOADS.glob("*")]
    return templates.TemplateResponse("admin.html", {"request": request, "files": files})


@app.post("/admin/upload")
async def admin_upload_file(file: UploadFile = File(...)):
    """Admin uploads file, which becomes part of the shared knowledge base"""
    dest = ADMIN_UPLOADS / file.filename
    with open(dest, "wb") as f:
        f.write(await file.read())

    # Extract text and create markdown
    text = extract_text_from_file(dest)
    md_folder = ADMIN_FOLDER / "docs_markdown"
    md_folder.mkdir(parents=True, exist_ok=True)
    md_file = md_folder / f"{dest.stem}.md"
    md_file.write_text(text, encoding="utf-8")

    # Update FAISS embeddings
    handler = FAISSEmbeddingHandler(ADMIN_FOLDER)
    handler.create_faiss_index(str(md_file))

    return JSONResponse({"status": "ok", "file": file.filename})


@app.get("/admin/files")
def list_admin_files():
    """Return list of admin-uploaded files"""
    files = [f.name for f in ADMIN_UPLOADS.glob("*")]
    return {"files": files}


@app.delete("/admin/delete/{filename}")
def delete_admin_file(filename: str):
    """Delete an uploaded document (and optionally reindex embeddings)"""
    target_file = ADMIN_UPLOADS / filename
    md_file = (ADMIN_FOLDER / "docs_markdown" / f"{Path(filename).stem}.md")

    if not target_file.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Delete both original and markdown files
    os.remove(target_file)
    if md_file.exists():
        os.remove(md_file)

    # Optionally rebuild embeddings
    handler = FAISSEmbeddingHandler(ADMIN_FOLDER)
    md_files = list((ADMIN_FOLDER / "docs_markdown").glob("*.md"))
    if md_files:
        handler.create_faiss_index(*[str(f) for f in md_files])

    return {"status": "deleted", "file": filename}
# =========================
# User routes
# =========================

@app.get("/", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/history")
async def get_history(user_email: str):
    query = """
    SELECT TOP 5 c.user, c.assistant
    FROM c WHERE c.user_id = @user_id
    ORDER BY c.timestamp DESC
    """
    params = [{"name": "@user_id", "value": user_email}]
    items = list(rag_engine.history_container.query_items(
        query=query,
        parameters=params,
        enable_cross_partition_query=True
    ))
    items = items[::-1]  # chronological order

    history = [{"question": i.get("user", ""), "answer": i.get("assistant", "")} for i in items]
    return {"history": history}

@app.post("/login")
def login(email: str = Form(...)):
    user_manager.ensure_user(email)
    
        
    return JSONResponse({"status": "ok", "email": email})

@app.get("/chat-page", response_class=HTMLResponse)
def chat_page(request: Request, user_email: str):
    return templates.TemplateResponse("chat.html", {"request": request, "user_email": user_email})

rag_engine = ThinkpalmCosmosRAGmethod2(COSMOS_ENDPOINT, COSMOS_KEY, COSMOS_DATABASE, COSMOS_CONTAINER, CHAT_CONTAINER)



@app.post("/ask")
async def ask_question(req: AskRequest, user_email: str = Depends(get_user_from_header)):
    print(f"User {user_email} asked: {req.question}")
    answer = rag_engine.ask(user_email, req.question)
    
    return {"question": req.question, "answer": answer}
# =========================
# Health check
# =========================

@app.get("/health")
def health():
    return {"status": "ok", "knowledge_base": "admin shared", "users": len(user_manager.list_users())}
