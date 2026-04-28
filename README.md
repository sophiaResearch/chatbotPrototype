# chatbotPrototype
A chatbot prototype for UNIMINUTO

# RAG Pipeline Scripts

Documentación y estructura de los scripts generados para el sistema de **Retrieval-Augmented Generation (RAG)**. Este repositorio contiene las utilidades necesarias para ingestar documentos, generar embeddings, realizar recuperación semántica y generar respuestas con LLM.

## Estructura de Scripts

| Archivo | Descripción |
|--------|-------------|
| `ingest.py` | Carga, limpia y divide documentos en chunks. Soporta PDF, TXT, DOCX y HTML. |
| `embed.py` | Genera embeddings a partir de los chunks y los almacena en una base de datos vectorial (ej. FAISS, Chroma, Qdrant, Pinecone). |
| `retrieve.py` | Recibe una consulta, genera su embedding y realiza búsqueda de similitud para recuperar los chunks más relevantes. |
| `generate.py` | Construye el prompt con el contexto recuperado y lo envía al LLM para generar una respuesta fundamentada. |
| `app.py` / `main.py` | Punto de entrada principal. Orquesta el flujo completo o expone una API/interfaz web. |
| `config.yaml` | Configuración centralizada: rutas, modelo de embeddings, LLM, hiperparámetros de chunking y retrieval. |
| `.env.example` | Variables de entorno requeridas (API keys, rutas de DB, etc.). |
| `test_rag.py` | Suite de evaluación: métricas de recuperación (Recall@K, MRR) y calidad de generación (BLEU, ROUGE, LLM-as-a-Judge). |



## Requisitos

- Python `>= 3.10`
- `pip` o `poetry` / `uv`
- Base de datos vectorial instalada o servicio cloud configurado
- Acceso a API de embeddings y LLM (OpenAI, Anthropic, Ollama, Hugging Face, etc.)

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/tu-repo-rag.git
cd tu-repo-rag

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env
# Edita .env con tus claves y rutas
