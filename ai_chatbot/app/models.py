import uuid
from django.db import models
from pgvector.django import VectorField


class ChatSession(models.Model):
    """Represents a conversation session between user and AI"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255, blank=True, default="New Chat")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    metadata = models.JSONField(default=dict, blank=True)
    
    class Meta:
        db_table = 'chat_session'
        ordering = ['-updated_at']
    
    def __str__(self):
        return f"Session {self.id} - {self.title}"


class ChatMessage(models.Model):
    """Stores individual messages in a chat session"""
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
        ('system', 'System'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        ChatSession,
        on_delete=models.CASCADE,
        related_name='messages'
    )
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    tokens_used = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    metadata = models.JSONField(default=dict, blank=True)
    
    class Meta:
        db_table = 'chat_message'
        ordering = ['created_at']
        indexes = [
            models.Index(fields=['session', 'created_at']),
        ]
    
    def __str__(self):
        return f"{self.role}: {self.content[:50]}"


class Document(models.Model):
    """Stores uploaded documents for retrieval"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    filename = models.CharField(max_length=255)
    content = models.TextField()
    file_size = models.IntegerField(default=0)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    metadata = models.JSONField(default=dict, blank=True)
    
    class Meta:
        db_table = 'document'
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"{self.filename} ({self.file_size} bytes)"


class DocumentChunk(models.Model):
    """Stores document chunks with embeddings for semantic search"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name='chunks'
    )
    chunk_text = models.TextField()
    chunk_index = models.IntegerField()
    embedding = VectorField(dimensions=384)  # all-MiniLM-L6-v2 produces 384-dim vectors
    metadata = models.JSONField(default=dict, blank=True)
    
    class Meta:
        db_table = 'document_chunk'
        ordering = ['document', 'chunk_index']
        indexes = [
            models.Index(fields=['document', 'chunk_index']),
        ]
    
    def __str__(self):
        return f"Chunk {self.chunk_index} of {self.document.filename}"