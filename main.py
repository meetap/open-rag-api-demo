import os
import uuid
import psycopg2
import numpy as np
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from psycopg2.extensions import register_adapter, AsIs
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pdfplumber

# Configuration
DB_HOST = os.environ.get('DB_HOST', 'your_db_host')
DB_NAME = os.environ.get('DB_NAME', 'your_db_name')
DB_USER = os.environ.get('DB_USER', 'your_db_user')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'your_db_password')

EMBEDDING_MODEL_NAME = os.environ.get('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-MiniLM-L6-v2')
LLM_MODEL_NAME = os.environ.get('LLM_MODEL', 'google/flan-t5-base')
MAX_PROMPT_LENGTH = int(os.environ.get('MAX_PROMPT_LENGTH', '2048'))
SERVICE_API_KEY = os.environ.get('SERVICE_API_KEY', 'your_service_api_key')

# Database Connection
conn = psycopg2.connect(
    host=DB_HOST,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)

# Register adapter for numpy.float32
def adapt_numpy_float32(numpy_float32):
    return AsIs(numpy_float32)
register_adapter(np.float32, adapt_numpy_float32)

# Register adapter for numpy arrays
def adapt_numpy_array(numpy_array):
    return AsIs(','.join(map(str, numpy_array)))
register_adapter(np.ndarray, adapt_numpy_array)

# Initialize FastAPI app
app = FastAPI()

# OAuth2 Scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Authentication Dependency
def api_key_auth(api_key: str = Depends(oauth2_scheme)):
    if api_key != SERVICE_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Forbidden"
        )

# Load Models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
llm_model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME).to(device)

# Initialize Database
def init_db():
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            text TEXT,
            embedding vector({embedding_model.get_sentence_embedding_dimension()}),
            file_name TEXT
        );
        """)
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents USING ivfflat (embedding);
        """)
        conn.commit()
init_db()

# Models
class QuestionBody(BaseModel):
    question: str

# Utilities
def get_embedding(text):
    embedding = embedding_model.encode([text])[0]
    return embedding

def tokenize_text(text, max_length=512):
    tokens = tokenizer.tokenize(text)
    return tokenizer.convert_tokens_to_string(tokens[:max_length])

# API Endpoints
# Uncomment the following line to enable API key authentication and comment the next line
# @app.post("/upload", dependencies=[Depends(api_key_auth)])
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    contents = await file.read()
    temp_file_path = f"/tmp/{uuid.uuid4()}.pdf"
    with open(temp_file_path, 'wb') as f:
        f.write(contents)

    texts = []
    with pdfplumber.open(temp_file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                texts.append(text)

    os.remove(temp_file_path)

    for text in texts:
        cleaned_text = tokenize_text(text)
        embedding = get_embedding(cleaned_text)
        embedding_str = '[' + ','.join(map(str, embedding.tolist())) + ']'
        with conn.cursor() as cur:
            cur.execute(f"""
            INSERT INTO documents (text, embedding, file_name)
            VALUES (%s, %s::vector, %s)
            """, (cleaned_text, embedding_str, file.filename))
    conn.commit()

    return {"status": "success", "message": "PDF uploaded and processed successfully"}

# Uncomment the following line to enable API key authentication and comment the next line
# @app.post("/ask", dependencies=[Depends(api_key_auth)])
@app.post("/ask")
async def ask_question(body: QuestionBody):
    question = body.question
    question_embedding = get_embedding(question)
    embedding_str = '[' + ','.join(map(str, question_embedding.tolist())) + ']'

    with conn.cursor() as cur:
        cur.execute(f"""
        SELECT text, file_name
        FROM documents
        ORDER BY embedding <-> %s::vector
        LIMIT 5;
        """, (embedding_str))
        results = cur.fetchall()

    contexts = [row[0] for row in results]
    file_names = list(set([row[1] for row in results]))

    prompt_start = "You are an assistant that provides answers based on the following information and explain about the information.\n\nInformation:\n"
    prompt = prompt_start + "\n\n".join(contexts) + f"\n\nQuestion: {question}\nAnswer:"

    input_ids = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=MAX_PROMPT_LENGTH).to(device)
    output_ids = llm_model.generate(
        input_ids,
        max_length=MAX_PROMPT_LENGTH,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return {
        "data": {
            "question": question,
            "answer": answer,
            "file_names": file_names
        },
        "status": "success",
        "message": "success"
    }
