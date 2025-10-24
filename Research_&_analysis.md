# Research & Analysis: Document-Aware AI Chatbot

## 1. System Architecture Overview

### High-Level Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Client    │────▶│    Django    │────▶│  PostgreSQL │
│   (HTTP)    │◀────│   REST API   │◀────│   Database  │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                           ├──────▶ LangGraph (Orchestration)
                           │
                           ├──────▶ LLM API (Groq/OpenAI)
                           │
                           └──────▶ Vector Store (pgvector)
```

### Component Responsibilities

- **Django REST Framework**: HTTP handling, request validation, session management
- **PostgreSQL + pgvector**: Persistent storage for chat history, documents, and embeddings
- **LangGraph**: Orchestrates complex AI workflows (retrieval → contextualization → generation)
- **Sentence Transformers**: Generates embeddings for semantic document retrieval
- **LLM API**: Generates conversational responses with streaming support

### Design Philosophy

- **Stateless API**: Each request is self-contained; state lives in PostgreSQL
- **Async-first**: Use Django async views for streaming and concurrent operations
- **Modular pipeline**: LangGraph nodes separate concerns (retrieval, memory, generation)

---

## 2. Technology Justifications

### Django + Django REST Framework

**Why**:

- Mature ORM with excellent PostgreSQL support
- Built-in admin for debugging
- Strong ecosystem for production deployment
- Native async support (Django 4.1+) for streaming

**Alternative Considered**: FastAPI

- **Rejected because**: Django's ORM migrations and admin panel provide better developer experience for this use case

### PostgreSQL with pgvector Extension

**Why**:

- Native vector similarity search (cosine, L2)
- ACID compliance for chat history
- Single database for both structured data and embeddings
- Eliminates need for separate vector DB (Pinecone, Weaviate)

**Alternative Considered**: ChromaDB + SQLite

- **Rejected because**: Adds deployment complexity; pgvector sufficient for <1M vectors

### LangGraph

**Why**:

- Explicit state management for multi-step AI workflows
- Cyclic graph support (for iterative retrieval)
- Built-in streaming and error handling
- Better than raw LangChain for complex orchestration

**Key Workflow**:

```
User Query → [Retrieve Docs] → [Load History] → [Generate Response] → Stream
```

### Groq API (Primary Choice)

**Why**:

- Free tier with high rate limits
- Llama 3.1 models with 128k context window
- Native streaming support
- Faster inference than OpenAI for similar quality

**Fallback**: OpenAI GPT-4 Turbo, Anthropic Claude

### Sentence Transformers (all-MiniLM-L6-v2)

**Why**:

- Local embedding generation (no API costs)
- 384-dimensional vectors (efficient storage)
- Good balance of speed vs. quality
- 256 token context window (suitable for chunks)

---

## 3. Document Retrieval Plan

### Chunking Strategy

**Approach**: Semantic chunking with overlap

- **Primary**: Split by paragraphs (double newline `\n\n`)
- **Fallback**: Fixed 512-character chunks with 50-character overlap
- **Rationale**: Preserves context boundaries while ensuring no chunk exceeds embedding model limits

### Embedding Pipeline

1. **Preprocessing**: Strip extra whitespace, normalize line endings
2. **Chunking**: Apply semantic splitting
3. **Embedding**: Generate 384-dim vectors using SentenceTransformers
4. **Storage**: Store chunks and embeddings in PostgreSQL with pgvector

### Retrieval Strategy

**Algorithm**: Cosine similarity with hybrid ranking

```python
# PostgreSQL query with pgvector
SELECT chunk_text, 1 - (embedding <=> query_embedding) AS similarity
FROM document_chunks
WHERE document_id = ANY(user_documents)
ORDER BY similarity DESC
LIMIT 5
```

**Hybrid Ranking** (planned for v2):

- Combine vector similarity (70%) + BM25 keyword matching (30%)
- Rerank top 20 results using cross-encoder

### Context Injection

- Retrieve top 5 most relevant chunks
- Format as: `### Relevant Context:\n{chunk1}\n\n{chunk2}...`
- Prepend to user query before LLM generation

---

## 4. Chat Memory Design

### Memory Strategy: Sliding Window + Summarization

**Why not store all messages?**

- LLM context limits (even with 128k tokens)
- Costs scale linearly with history length
- Older messages lose relevance

**Implementation**:

1. **Short-term**: Last 10 messages (raw, full context)
2. **Long-term**: Summarize messages 11-50 into single context block
3. **Ancient**: Drop messages >50 (retrievable via `/messages/` endpoint)

### Database Schema

```sql
CREATE TABLE chat_message (
    id UUID PRIMARY KEY,
    session_id UUID REFERENCES chat_session(id),
    role VARCHAR(10),  -- 'user' or 'assistant'
    content TEXT,
    tokens_used INTEGER,
    created_at TIMESTAMP,
    metadata JSONB  -- For future: embeddings, citations
);

CREATE INDEX idx_session_time ON chat_message(session_id, created_at);
```

### Memory Loading in LangGraph

```python
def load_memory(state):
    recent = ChatMessage.objects.filter(
        session_id=state['session_id']
    ).order_by('-created_at')[:10]
  
    state['chat_history'] = [
        {"role": msg.role, "content": msg.content}
        for msg in reversed(recent)
    ]
    return state
```

---

## 5. Streaming Implementation

### Why Streaming?

- Improves perceived latency (time-to-first-token < 500ms)
- Better UX for long responses
- Reduces client timeout issues

### Technical Approach: Server-Sent Events (SSE)

**Advantages over WebSockets**:

- HTTP-based (no connection upgrade complexity)
- Auto-reconnection in browsers
- Compatible with Django async views

### Django Async View Implementation

```python
from django.http import StreamingHttpResponse
import asyncio

async def chat_stream(request):
    async def event_generator():
        async for chunk in langgraph_stream(query):
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
  
    return StreamingHttpResponse(
        event_generator(),
        content_type='text/event-stream'
    )
```

### LangGraph Streaming

LangGraph natively supports streaming via `astream_events`:

```python
async for event in graph.astream_events(input, version="v1"):
    if event["event"] == "on_chat_model_stream":
        yield event["data"]["chunk"].content
```

### Error Handling

- Send `data: {"error": "message"}\n\n` on failures
- Client closes connection on timeout
- Log partial completions for debugging

---

## 6. Scalability & Extensibility

### Scalability Considerations

#### Current Bottlenecks

1. **Embedding Generation**: CPU-bound, blocks request thread
   - **Solution**: Celery task queue for async processing
2. **Vector Search**: O(n) scan with pgvector (no HNSW index in free tier)
   - **Solution**: Add `USING ivfflat` index when >10k chunks
3. **LLM API Rate Limits**: Groq free tier = 30 req/min
   - **Solution**: Implement request queue with exponential backoff

#### Horizontal Scaling Strategy

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│ Django 1 │     │ Django 2 │     │ Django N │
└────┬─────┘     └────┬─────┘     └────┬─────┘
     └─────────┬──────┴──────┬──────────┘
               │             │
         ┌─────▼─────┐  ┌────▼────┐
         │ PostgreSQL│  │  Redis  │
         │ (primary) │  │ (cache) │
         └───────────┘  └─────────┘
```

- **Stateless Django instances** behind load balancer
- **PostgreSQL connection pooling** (PgBouncer)
- **Redis for session caching** (reduce DB load)

### Extensibility Features

#### 1. Pluggable LLM Backends

```python
# settings.py
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'groq')  # 'openai', 'claude', 'local'

# llm_factory.py
def get_llm():
    if settings.LLM_PROVIDER == 'groq':
        return ChatGroq(model="llama3-70b")
    elif settings.LLM_PROVIDER == 'openai':
        return ChatOpenAI(model="gpt-4-turbo")
```

#### 2. Multi-Modal Document Support (Planned)

- **PDF**: PyMuPDF for text extraction
- **DOCX**: python-docx parser
- **Images**: OCR with Tesseract + multimodal LLM (GPT-4V)

#### 3. Advanced Retrieval (RAG 2.0)

- **HyDE**: Generate hypothetical document for better retrieval
- **Query decomposition**: Break complex queries into sub-questions
- **Citation tracking**: Link LLM responses to source chunks

#### 4. Multi-User & Permissions

```python
# Future schema
class ChatSession:
    user = ForeignKey(User)
    shared_with = ManyToManyField(User)
    permissions = JSONField()  # read, write, admin
```

---

## 7. Testing & Validation

### Testing Strategy

#### Unit Tests (pytest + Django TestCase)

- **Models**: CRUD operations, constraints
- **Embeddings**: Chunk generation, similarity search
- **LangGraph**: Node execution, state transitions

```python
def test_document_chunking():
    doc = Document.objects.create(content="..." * 1000)
    chunks = create_chunks(doc)
    assert all(len(c) <= 512 for c in chunks)
    assert len(chunks) > 1
```

#### Integration Tests

- **API Endpoints**: Request/response validation
- **Streaming**: Verify SSE format, completion events
- **Retrieval Pipeline**: E2E from upload → query → context injection

```python
@pytest.mark.asyncio
async def test_chat_with_document():
    # Upload document
    doc_response = client.post('/api/document/', files={'file': ...})
  
    # Query with context
    response = client.post('/api/chat/', json={
        'message': 'What does the document say about AI?',
        'session_id': session_id
    })
  
    assert 'AI' in response.json()['text']
```

#### Load Testing (Locust)

- **Target**: 100 concurrent users, 10 msg/sec
- **Metrics**: p95 latency < 2s, error rate < 0.1%

### Validation Criteria

#### Functional Requirements

- ✅ Chat persists across sessions
- ✅ Document chunks retrieved accurately (MRR > 0.7)
- ✅ Streaming starts within 500ms
- ✅ Memory window includes last 10 messages

#### Quality Metrics

- **Retrieval Accuracy**: Manual evaluation of top-5 chunks (precision@5)
- **Response Quality**: LLM-as-judge (GPT-4 scores coherence, relevance)
- **Latency**: Time-to-first-token < 500ms, total response < 5s

### Monitoring & Observability

```python
# Metrics to track
- requests_per_minute
- llm_tokens_used
- retrieval_latency_p95
- streaming_error_rate
- database_connection_pool_usage
```

**Tools**: Django Debug Toolbar (dev), Prometheus + Grafana (prod)

---

## Conclusion

This architecture prioritizes:

1. **Simplicity**: Single database, minimal dependencies
2. **Performance**: Async streaming, efficient vector search
3. **Maintainability**: Clear separation of concerns via LangGraph
4. **Extensibility**: Pluggable components for future enhancements

**Trade-offs Acknowledged**:

- pgvector scales to ~1M vectors; larger deployments need dedicated vector DB
- CPU-bound embeddings may require GPU acceleration at scale
- Groq free tier limits production throughput (mitigated by caching)

**Next Steps**:

1. Implement core Django models + migrations
2. Build LangGraph retrieval pipeline
3. Add streaming endpoint with SSE
4. Integrate pgvector for semantic search
5. Comprehensive testing suite
