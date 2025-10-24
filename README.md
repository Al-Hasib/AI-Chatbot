# Document-Aware AI Chatbot

A production-ready AI chatbot with document retrieval, memory management, and real-time streaming built with Django, PostgreSQL, and LangGraph.

## Features

- 🤖 **AI Chat with Memory**: Conversational AI that remembers context across messages
- 📄 **Document Retrieval**: Upload `.txt` files and ask questions about them
- 🌊 **Streaming Responses**: Real-time token-by-token streaming via Server-Sent Events
- 🧠 **Semantic Search**: pgvector-powered vector similarity search
- 🔄 **LangGraph Orchestration**: Sophisticated AI workflow management
- 💾 **Persistent Storage**: All conversations and documents saved to PostgreSQL

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Client    │────▶│    Django    │────▶│  PostgreSQL │
│   (HTTP)    │◀────│   REST API   │◀────│  + pgvector │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                           ├──────▶ LangGraph (Orchestration)
                           ├──────▶ Groq API (LLM)
                           └──────▶ SentenceTransformers (Embeddings)
```

## Prerequisites

- Python 3.10+
- PostgreSQL 14+ with pgvector extension
- Groq API key (or OpenAI API key)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd chatbot_project
```

### 2. Set Up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up PostgreSQL with pgvector

```bash
# Install PostgreSQL (if not already installed)
# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# macOS
brew install postgresql

# Start PostgreSQL service
sudo service postgresql start  # Linux
brew services start postgresql  # macOS

# Create database and user
sudo -u postgres psql
```

In PostgreSQL shell:

```sql
CREATE DATABASE chatbot_db;
CREATE USER postgres WITH PASSWORD 'your-password';
GRANT ALL PRIVILEGES ON DATABASE chatbot_db TO postgres;

-- Connect to the database
\c chatbot_db

-- Install pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify installation
\dx
```

### 5. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
# Database
DB_NAME=chatbot_db
DB_USER=postgres
DB_PASSWORD=your-postgres-password
DB_HOST=localhost
DB_PORT=5432

# LLM API (get free key from https://console.groq.com)
GROQ_API_KEY=your-groq-api-key
```

### 6. Run Migrations

```bash
python manage.py makemigrations
python manage.py migrate
```

### 7. Create Superuser (Optional)

```bash
python manage.py createsuperuser
```

### 8. Start Development Server

```bash
python manage.py runserver
```

Server will start at `http://localhost:8000`

## API Endpoints

### 1. Chat (Streaming)

**POST** `/api/chat/`

Stream AI responses token-by-token using Server-Sent Events.

```bash
curl -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, how are you?",
    "session_id": null,
    "use_documents": true
  }'
```

**Request Body:**

```json
{
  "message": "Your question here",
  "session_id": "uuid-of-existing-session (optional)",
  "use_documents": true,
  "document_ids": ["doc-uuid-1", "doc-uuid-2"] (optional)
}
```

**Response:** Server-Sent Events stream

```
data: {"session_id": "123e4567-e89b-12d3-a456-426614174000"}

data: {"token": "Hello"}

data: {"token": "!"}

data: {"token": " How"}

data: {"done": true, "full_response": "Hello! How can I help you?"}
```

### 2. Upload Document

**POST** `/api/documents/`

Upload a `.txt` file for semantic search and retrieval.

```bash
curl -X POST http://localhost:8000/api/documents/ \
  -F "file=@document.txt"
```

**Response:**

```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "filename": "document.txt",
  "file_size": 5420,
  "uploaded_at": "2025-01-15T10:30:00Z",
  "chunk_count": 12
}
```

### 3. Get Session Messages

**GET** `/api/sessions/{session_id}/messages/`

Retrieve all messages from a conversation.

```bash
curl http://localhost:8000/api/sessions/123e4567-e89b-12d3-a456-426614174000/messages/
```

**Response:**

```json
[
  {
    "id": "msg-uuid-1",
    "role": "user",
    "content": "What is machine learning?",
    "tokens_used": 0,
    "created_at": "2025-01-15T10:30:00Z"
  },
  {
    "id": "msg-uuid-2",
    "role": "assistant",
    "content": "Machine learning is...",
    "tokens_used": 150,
    "created_at": "2025-01-15T10:30:05Z"
  }
]
```

### 4. List All Documents

**GET** `/api/documents/`

```bash
curl http://localhost:8000/api/documents/
```

### 5. List All Sessions

**GET** `/api/sessions/`

```bash
curl http://localhost:8000/api/sessions/
```

## Usage Examples

### Example 1: Simple Chat

```python
import requests
import json

url = "http://localhost:8000/api/chat/"
data = {
    "message": "What is the capital of France?"
}

response = requests.post(url, json=data, stream=True)

for line in response.iter_lines():
    if line:
        if line.startswith(b'data: '):
            event_data = json.loads(line[6:])
            if 'token' in event_data:
                print(event_data['token'], end='', flush=True)
```

### Example 2: Chat with Document Context

```python
# 1. Upload document
files = {'file': open('research_paper.txt', 'rb')}
doc_response = requests.post(
    'http://localhost:8000/api/documents/',
    files=files
)
doc_id = doc_response.json()['id']

# 2. Ask questions about the document
chat_data = {
    "message": "What are the main findings in this paper?",
    "use_documents": True,
    "document_ids": [doc_id]
}

response = requests.post(
    'http://localhost:8000/api/chat/',
    json=chat_data,
    stream=True
)

for line in response.iter_lines():
    if line and line.startswith(b'data: '):
        event_data = json.loads(line[6:])
        if 'token' in event_data:
            print(event_data['token'], end='', flush=True)
```

### Example 3: Continue Previous Conversation

```python
# First message
response1 = requests.post(
    'http://localhost:8000/api/chat/',
    json={"message": "My name is Alice"}
)
session_id = None
for line in response1.iter_lines():
    if line and line.startswith(b'data: '):
        data = json.loads(line[6:])
        if 'session_id' in data:
            session_id = data['session_id']
            break

# Follow-up message (remembers context)
response2 = requests.post(
    'http://localhost:8000/api/chat/',
    json={
        "message": "What's my name?",
        "session_id": session
```
