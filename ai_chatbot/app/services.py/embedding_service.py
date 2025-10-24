import re
from typing import List
from sentence_transformers import SentenceTransformer
from django.db.models import F
from pgvector.django import CosineDistance
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating and managing embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with sentence transformer model"""
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        logger.info(f"Loaded embedding model: {model_name}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for given text
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing)
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise
    
    @staticmethod
    def cosine_distance(query_embedding: List[float]):
        """
        Helper for pgvector cosine distance query
        
        Args:
            query_embedding: Query vector
            
        Returns:
            Django ORM expression for ordering by cosine distance
        """
        from ..models import DocumentChunk
        return CosineDistance(F('embedding'), query_embedding)
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Split text into chunks for embedding
        
        Strategy:
        1. First try semantic splitting (by paragraphs)
        2. Fall back to fixed-size chunks with overlap
        
        Args:
            text: Text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Character overlap between chunks
            
        Returns:
            List of text chunks
        """
        # Normalize text
        text = text.strip()
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
        text = re.sub(r' +', ' ', text)  # Normalize spaces
        
        # Try semantic chunking by paragraphs
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If paragraph fits in chunk, add it
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk += ("\n\n" if current_chunk else "") + para
            else:
                # Save current chunk if not empty
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If paragraph itself is too long, split it
                if len(para) > chunk_size:
                    para_chunks = EmbeddingService._split_long_text(
                        para, chunk_size, overlap
                    )
                    chunks.extend(para_chunks[:-1])
                    current_chunk = para_chunks[-1] if para_chunks else ""
                else:
                    current_chunk = para
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        # If no semantic chunks, fall back to fixed-size
        if not chunks:
            chunks = EmbeddingService._split_long_text(text, chunk_size, overlap)
        
        logger.info(f"Created {len(chunks)} chunks from text of length {len(text)}")
        return chunks
    
    @staticmethod
    def _split_long_text(text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split long text into fixed-size chunks with overlap"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > chunk_size * 0.7:  # At least 70% of chunk
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return chunks