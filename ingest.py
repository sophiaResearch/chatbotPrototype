"""
ingest_pdf.py - Indexa PDFs en ChromaDB para RAG
"""

import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader  # Opción 1: Nativa
# from langchain_community.document_loaders import PyMuPDFLoader  # Opción 2: Más rápida (requiere pip install pymupdf)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# 🔹 OPCIÓN A: Lista manual de PDFs
PDF_FILES = [
    "docs/PoliticaAutoevaluaciónAutorregulaciónMejoramientoContinuoIntegral.pdf",
    "docs/PoliticaEticaIntegridadCientificaInvestigacionDesarrolloInnovacionCreacionArtisticaCulturalUNIMINUTO13Agosto2021.pdf",
    "docs/PoliticaPublicaciones13agosto2021.pdf",
    "docs/ProyectoEducativoInstitucionalUNIMINUTO2021.pdf"
    "docs/SistemaIDICUNIMINUTO5Marzo2021.pdf"
]

# 🔹 OPCIÓN B: Cargar todos los PDFs de una carpeta automáticamente
# PDF_DIR = "./mis_documentos"
# PDF_FILES = [str(f) for f in Path(PDF_DIR).glob("*.pdf")]

def main():
    print("📄 Iniciando carga de PDFs...")
    docs = []
    
    for pdf_path in PDF_FILES:
        if not os.path.exists(pdf_path):
            print(f"⚠️  No encontrado: {pdf_path}")
            continue
            
        try:
            # Opción recomendada para empezar:
            loader = PyPDFLoader(pdf_path)
            # Si quieres mayor velocidad y mejor manejo de layouts:
            # loader = PyMuPDFLoader(pdf_path)
            
            pages = loader.load()
            
            # Agregar metadata útil para que el bot cite la fuente
            for page in pages:
                page.metadata["source"] = pdf_path
                page.metadata["page"] = page.metadata.get("page", "?")
                
            docs.extend(pages)
            print(f"✅ {pdf_path} ({len(pages)} páginas)")
        except Exception as e:
            print(f"❌ Error en {pdf_path}: {e}")
            
    print(f"\n📚 Total de páginas cargadas: {len(docs)}")
    if len(docs) == 0:
        print("⛔ No se cargaron documentos. Revisa las rutas.")
        return

    # Chunking (igual que antes)
    print("✂️ Dividiendo en chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", "•", ".", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    print(f"✅ {len(splits)} chunks creados")

    # Embeddings & ChromaDB (idéntico a tu versión web)
    print("🔢 Generando embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("💾 Guardando en ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db_pdf"  # ← Carpeta diferente para no mezclar con web
    )
    
    print(f"\n🎉 ¡Indexación de PDFs completada!")
    print(f"📊 Chunks en la DB: {vectorstore._collection.count()}")

if __name__ == "__main__":
    main()