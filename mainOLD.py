"""
Backend del chatbot RAG con Groq API - Versión con Debug
"""

from fastapi import FastAPI, HTTPException
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
import traceback  # ← Para mostrar errores completos
import re

# Cargar variables de entorno
load_dotenv()

app = FastAPI(title="Chatbot RAG - Groq + Qwen")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# CONFIGURACIÓN CON VALIDACIÓN
# ─────────────────────────────────────────────

print("🔧 Cargando embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("🔧 Conectando a ChromaDB...")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Verificar que hay documentos
doc_count = vectorstore._collection.count()
print(f"📊 Documentos en ChromaDB: {doc_count}")
if doc_count == 0:
    print("⚠️  ADVERTENCIA: ChromaDB está vacío. Ejecuta: python ingest.py")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("🔧 Configurando Groq API...")
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    print("❌ ERROR: GROQ_API_KEY no encontrada. Verifica tu archivo .env")
else:
    print(f"✅ API Key cargada: {groq_key[:10]}...")

llm = ChatGroq(
    model="qwen/qwen3-32b",
    api_key=groq_key,
    temperature=0.7
)

# Prompt
system_prompt = """
Eres un asistente virtual experto para el sitio web.

INSTRUCCIONES CRÍTICAS:
1. Responde SOLO con la información del contexto proporcionado.
2. NO muestres tu proceso de razonamiento interno.
3. NO uses etiquetas <think> o similares.
4. Ve DIRECTO a la respuesta final sin introducciones.
5. Sé claro, conciso y útil.
6. Si no encuentras la información en el contexto, dilo amablemente.

Contexto: {context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# ─────────────────────────────────────────────
# CADENA RAG CON LOGS
# ─────────────────────────────────────────────

def rag_chain(input_text):
    print(f"\n🔍 [DEBUG] Pregunta recibida: {input_text}")
    
    # 1. Buscar documentos
    docs = retriever.invoke(input_text)
    print(f"📄 [DEBUG] Documentos encontrados: {len(docs)}")
    
    if len(docs) == 0:
        return "Lo siento, no encontré información relevante en la base de datos."
    
    # 2. Extraer contexto
    context = "\n\n".join([doc.page_content for doc in docs])
    print(f"📝 [DEBUG] Contexto: {len(context)} caracteres")
    
    # 3. Formatear prompt
    full_prompt = prompt.format(context=context, input=input_text)
    
    # 4. Llamar a Groq
    print("🤖 [DEBUG] Llamando a Groq API...")
    response = llm.invoke(full_prompt)
    print(f"✅ [DEBUG] Respuesta recibida")
    
    return response.content

print("\n✅ ¡Chatbot listo!\n")

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
        print(f"📨 Nueva petición /chat")
        answer = rag_chain(request.message)
        print(f"📤 Respuesta enviada al cliente")
        print(f"{'='*50}\n")
        return ChatResponse(response=answer)
    except Exception as e:
        # ← ESTO ES LO IMPORTANTE: Imprime el error completo
        print(f"\n❌ {'='*50}")
        print(f"❌ ERROR EN /chat: {str(e)}")
        print(f"❌ Traceback completo:")
        traceback.print_exc()
        print(f"❌ {'='*50}\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "docs": doc_count}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)