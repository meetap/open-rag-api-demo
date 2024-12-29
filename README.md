# OpenRAG API Demo

This project is a FastAPI-based Chatbot AI application that provides an API for uploading PDF files, extracting text, generating embeddings, and answering questions based on the uploaded documents using open-source LLM models.

You can find a comprehensive slide presentation for this project in the following link: [https://docs.google.com/presentation/d/1jmwY-pLP_y-Qibolwzeee7xuLOtJHgtq1Ib89h1QIOU/edit?usp=sharing](https://docs.google.com/presentation/d/1jmwY-pLP_y-Qibolwzeee7xuLOtJHgtq1Ib89h1QIOU/edit?usp=sharing).

The slide presentation is a valuable resource for understanding the intricacies of the project. It is especially useful for:

- **Developers**: Looking to contribute or integrate the API into their applications.
- **Stakeholders**: Interested in the technical and functional aspects of the project.
- **Educators and Students**: Who want to learn about building AI-powered applications using open-source tools.

We encourage you to review the presentation to gain deeper insights into the project's capabilities, architecture, and potential applications.

## Table of Contents

- [OpenRAG API Demo](#openrag-api-demo)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
    - [1. Clone the Repository](#1-clone-the-repository)
    - [2. Set Up Python Environment](#2-set-up-python-environment)
      - [Using `uv` (Recommended)](#using-uv-recommended)
      - [Using `venv` (Alternative)](#using-venv-alternative)
    - [3. Install Dependencies](#3-install-dependencies)
    - [4. Install Transformer Models](#4-install-transformer-models)
    - [5. Set Up Environment Variables](#5-set-up-environment-variables)
    - [6. Set Up the Database](#6-set-up-the-database)
  - [Configuration](#configuration)
  - [Running the Application](#running-the-application)
  - [API Endpoints](#api-endpoints)
    - [Upload PDF](#upload-pdf)
    - [Ask Question](#ask-question)
  - [API Documentation](#api-documentation)
  - [Project Structure](#project-structure)
  - [Dependencies](#dependencies)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)

## Features

- **Upload PDFs**: Upload PDF files and extract text content.
- **Generate Embeddings**: Create text embeddings using Sentence Transformers.
- **Question Answering**: Answer questions based on the uploaded documents using an open-source LLM model.
- **Database Storage**: Store extracted texts and embeddings in a PostgreSQL database with vector support.
- **API Authentication**: Secure API endpoints using an API key.
- **Interactive Documentation**: Auto-generated API docs accessible through `/docs` endpoint.

## Prerequisites

- **Python**: Version 3.11.11
- **PostgreSQL**: With `vector` extension enabled.
- **uv**: Used as the package and project manager.
- **GPU (Optional)**: CUDA-enabled GPU for faster model inference.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/meetap/open-rag-api-demo.git
cd open-rag-api-demo
```

### 2. Set Up Python Environment

#### Using `uv` (Recommended)

Ensure you have `uv` installed:

```bash
pip install uv
```

Create a virtual environment and activate it:

```bash
uv venv create
uv venv activate
```

#### Using `venv` (Alternative)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies

Using `uv`:

```bash
uv pip install -e .
```

Or if you use poetry you can install them with:

```bash
poetry install
```

### 4. Install Transformer Models

Install the required models for embeddings and the LLM:

```bash
pip install sentence-transformers transformers torch
```

### 5. Set Up Environment Variables

Copy the sample environment file and modify it:

```bash
cp .env.sample .env
```

Edit `.env` with your configurations:

```dotenv
DB_HOST=your_db_host
DB_NAME=your_db_name
DB_USER=your_db_user
DB_PASSWORD=your_db_password
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-MiniLM-L6-v2
LLM_MODEL_NAME=google/flan-t5-base
MAX_PROMPT_LENGTH=2048
SERVICE_API_KEY=your_service_api_key
```

### 6. Set Up the Database

Ensure PostgreSQL is installed and running. Create a database and enable the `vector` extension:

```sql
CREATE DATABASE your_db_name;
\c your_db_name;
CREATE EXTENSION IF NOT EXISTS vector;
```

## Configuration

- **Database Configuration**: Update the database credentials in the `.env` file.
- **Model Configuration**: Set your preferred models in the `.env` file.
- **API Key**: Set a secure `SERVICE_API_KEY` in the `.env` file for API authentication.

## Running the Application

Start the FastAPI application using `uvicorn`:

```bash
uvicorn main:app --reload
```

The application will be accessible at `http://127.0.0.1:8000`.

## API Endpoints

### Upload PDF

- **Endpoint**: `/upload`
- **Method**: `POST`
- **Authentication**: Bearer Token (`SERVICE_API_KEY`)
- **Content-Type**: `multipart/form-data`
- **Parameter**:
  - `file` (required): The PDF file to upload.

**Sample Request using `curl`**:

```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -H "Authorization: Bearer your_service_api_key" \
  -F "file=@/path/to/your/file.pdf"
```

**Response**:

```json
{
  "status": "success",
  "message": "PDF uploaded and processed successfully"
}
```

### Ask Question

- **Endpoint**: `/ask`
- **Method**: `POST`
- **Authentication**: Bearer Token (`SERVICE_API_KEY`)
- **Content-Type**: `application/json`
- **Body**:

  ```json
  {
    "question": "Your question here"
  }
  ```

**Sample Request using `curl`**:

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Authorization: Bearer your_service_api_key" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the uploaded document?"}'
```

**Response**:

```json
{
  "data": {
    "question": "What is the main topic of the uploaded document?",
    "answer": "The main topic is...",
    "file_names": ["file.pdf"]
  },
  "status": "success",
  "message": "success"
}
```

## API Documentation

FastAPI provides interactive API documentation automatically generated from your code. 

![Swagger UI](https://fastapi.tiangolo.com/img/index/index-01-swagger-ui-simple.png)

Navigate to `http://127.0.0.1:8000/docs` to access the Swagger UI, an interactive interface to test your API endpoints.

Alternatively, visit `http://127.0.0.1:8000/redoc` for ReDoc documentation.

- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

These interfaces allow you to explore and test the API endpoints with ease.

## Project Structure

```bash
open-rag-api-demo/
├── main.py           # Main application file.
├── pyproject.toml    # Project dependencies and metadata.
├── .env.sample       # Sample environment configuration.
├── .python-version   # Python version specification.
├── LICENSE           # License file (MIT License).
└── README.md         # Project documentation.
```

## Dependencies

- **fastapi**: Web framework for building APIs with Python.
- **uvicorn**: ASGI server for running FastAPI applications.
- **psycopg2**: PostgreSQL database adapter.
- **numpy**: Library for numerical computations.
- **torch**: PyTorch library for machine learning models.
- **sentence-transformers**: Library for easy use of Transformer models for encoding sentences.
- **transformers**: State-of-the-art Machine Learning for Pytorch and TensorFlow.
- **pdfplumber**: Library for extracting information from PDF files.
- **python-multipart**: Required for form data parsing in FastAPI.

All dependencies are specified in the `pyproject.toml` file.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- **FastAPI**: For providing an easy-to-use web framework.
- **Hugging Face**: For the Transformers and Sentence Transformers libraries.
- **OpenAI**: For advancements in AI models and research.
- **pdfplumber**: For simplifying PDF text extraction.
- **Community Contributors**: For their continuous support and contributions.
