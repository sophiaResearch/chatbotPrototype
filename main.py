"""
main.py - Optimizado para Qwen3-32B en Groq
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import re
import traceback
import shutil
from pathlib import Path
import subprocess

load_dotenv()

app = FastAPI(title="Chatbot RAG - Qwen3 + Groq")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# FUNCIÓN DE LIMPIEZA ESPECÍFICA PARA QWEN3
# ─────────────────────────────────────────────

def clean_qwen3_response(text):
    """
    Limpia el output de Qwen3 eliminando bloques de pensamiento
    """
    if not text:
        return "Lo siento, no pude generar una respuesta."
    
    # 1. Eliminar bloques <think>...</think>
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # 2. Eliminar <thinking>...</thinking>
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # 3. Eliminar [THOUGHT]...[/THOUGHT]
    text = re.sub(r'\[THOUGHT\].*?\[/THOUGHT\]', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # 4. Eliminar frases de introducción comunes
    intro_phrases = [
        r"(?i)^(déjame pensar|déjame ver|veamos|analizando|basado en el contexto|según la información)",
        r"(?i)^(revisando|procesando|pensando|razonando|ok|bien|entonces|ahora)",
    ]
    for pattern in intro_phrases:
        text = re.sub(pattern, "", text)
    
    # 5. Limpiar formato
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text if text else "Lo siento, no pude generar una respuesta."

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

print(" Cargando embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print(" Conectando a ChromaDB...")
vectorstore = Chroma(
    persist_directory="./chroma_db_pdf",
    embedding_function=embeddings
)

doc_count = vectorstore._collection.count()
print(f" Documentos en ChromaDB: {doc_count}")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print(" Configurando Groq API con Qwen3-32B...")
groq_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="qwen/qwen3-32b",  # ← Verifica el nombre exacto en tu cuenta Groq
    api_key=groq_key,
    temperature=0.5,  # Un poco más bajo para respuestas más directas
    max_tokens=1024
)

# ─────────────────────────────────────────────
# PROMPT OPTIMIZADO PARA QWEN3
# ─────────────────────────────────────────────

system_prompt = """
Eres un asistente virtual experto para el sitio web.

INSTRUCCIONES CRÍTICAS:
1. Responde SOLO con la información del contexto proporcionado.
2. NO muestres tu proceso de razonamiento interno.
3. NO uses etiquetas <think> o similares.
4. Ve DIRECTO a la respuesta final sin introducciones.
5. Sé claro, conciso y útil.
6. Si no encuentras la información, dilo amablemente.

Contexto: {context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# ─────────────────────────────────────────────
# CADENA RAG
# ─────────────────────────────────────────────

def rag_chain(input_text):
    print(f"\n [DEBUG] Pregunta: {input_text}")
    
    docs = retriever.invoke(input_text)
    print(f" Documentos encontrados: {len(docs)}")
    
    if len(docs) == 0:
        return "Lo siento, no encontré información relevante."
    
    context = "\n\n".join([doc.page_content for doc in docs])
    print(f" Contexto: {len(context)} caracteres")
    
    full_prompt = prompt.format(context=context, input=input_text)
    
    print(" Llamando a Qwen3-32B...")
    response = llm.invoke(full_prompt)
    
    return response.content

print("\n ¡Chatbot Qwen3 listo!\n")

# ─────────────────────────────────────────────
# MODELOS
# ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        print(f"\n{'='*50}")
        print(f"Nueva petición /chat")
        
        answer = rag_chain(request.message)
        answer = clean_qwen3_response(answer)  # ← LIMPIAR RESPUESTA QWEN3
        
        print(f"Respuesta limpia enviada")
        print(f"{'='*50}\n")
        
        return ChatResponse(response=answer)
    except Exception as e:
        print(f"\n {'='*50}")
        print(f" ERROR EN /chat: {str(e)}")
        traceback.print_exc()
        print(f" {'='*50}\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "docs": doc_count, "model": "qwen/qwen3-32b"}

# ─────────────────────────────────────────────
# UPLOAD PAGE
# ─────────────────────────────────────────────

@app.get("/upload")
async def upload_page():
    return FileResponse("static/upload.html")

# ─────────────────────────────────────────────
# PDF UPLOAD ENDPOINT
# ─────────────────────────────────────────────

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Sube un archivo PDF al directorio docs/
    """
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")

        docs_dir = Path("./docs")
        docs_dir.mkdir(exist_ok=True)

        file_path = docs_dir / file.filename

        if file_path.exists():
            raise HTTPException(status_code=400, detail=f"El archivo {file.filename} ya existe en docs/")

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = file_path.stat().st_size

        return {
            "status": "ok",
            "message": f"Archivo {file.filename} subido correctamente",
            "filename": file.filename,
            "size_bytes": file_size,
            "path": str(file_path)
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error al subir archivo: {str(e)}")

# ─────────────────────────────────────────────
# RE-INDEX ENDPOINT
# ─────────────────────────────────────────────

@app.post("/reindex")
async def reindex_pdfs():
    """
    Re-indexa todos los PDFs del directorio docs/ en ChromaDB
    """
    try:
        result = subprocess.run(
            ["python", "ingest.py"],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Error durante la indexación:\n{result.stderr}"
            )

        return {
            "status": "ok",
            "message": "PDFs re-indexados correctamente",
            "output": result.stdout
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="La indexación tardó demasiado")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ─────────────────────────────────────────────
# LIST UPLOADED PDFs
# ─────────────────────────────────────────────

@app.get("/list-pdfs")
async def list_pdfs():
    """
    Lista todos los PDFs en el directorio docs/
    """
    try:
        docs_dir = Path("./docs")
        pdf_files = [f.name for f in docs_dir.glob("*.pdf")]

        return {
            "status": "ok",
            "count": len(pdf_files),
            "files": sorted(pdf_files)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
    #Localhost