"""
Script para indexar las páginas de tu sitio web en ChromaDB
Ejecuta: python ingest.py
"""

import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# URLs del sitio web
URLS = [
    # "https://tusitioweb.com",
    # "https://tusitioweb.com/servicios",
    # "https://tusitioweb.com/contacto",
    # "https://tusitioweb.com/faq",
    # Agrega más URLs según necesites
    "https://www.uniminuto.edu/investigacion",
    "https://www.uniminuto.edu/portal-i-d-i-c"

]

def main():
    print("Iniciando scraping del sitio web...")
    
    # 1. Cargar documentos desde las URLs
    loader = WebBaseLoader(URLS)
    docs = loader.load()
    print(f"✅ {len(docs)} documentos cargados")
    
    # 2. Dividir en chunks
    print("✂️ Dividiendo texto en chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    print(f"✅ {len(splits)} chunks creados")
    
    # 3. Crear embeddings
    print("🔢 Generando embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    
    # 4. Guardar en ChromaDB
    print("💾 Guardando en ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print(f"✅ ¡Indexación completada!")
    print(f"📊 Total de chunks en la base de datos: {vectorstore._collection.count()}")

if __name__ == "__main__":
    main()