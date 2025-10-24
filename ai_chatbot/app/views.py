from django.shortcuts import render

# Create your views here.
import json
import asyncio
from django.http import StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from rest_framework import viewsets, status
from rest_framework.decorators import action, api_view
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from asgiref.sync import sync_to_async
import logging

from .models import ChatSession, ChatMessage, Document, DocumentChunk
from .serializers import (
    ChatSessionSerializer, ChatMessageSerializer, 
    ChatRequestSerializer, DocumentSerializer, DocumentUploadSerializer
)
from .services.langgraph_service import LangGraphService
from .services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class ChatSessionViewSet(viewsets.ModelViewSet):
    """ViewSet for chat sessions"""
    queryset = ChatSession.objects.all()
    serializer_class = ChatSessionSerializer
    
    @action(detail=True, methods=['get'])
    def messages(self, request, pk=None):
        """Get all messages for a session"""
        session = self.get_object()
        messages = session.messages.all()
        serializer = ChatMessageSerializer(messages, many=True)
        return Response(serializer.data)


class DocumentViewSet(viewsets.ModelViewSet):
    """ViewSet for documents"""
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer
    parser_classes = (MultiPartParser, FormParser)
    
    def create(self, request, *args, **kwargs):
        """Upload and process a document"""
        serializer = DocumentUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        uploaded_file = serializer.validated_data['file']
        
        try:
            # Read file content
            content = uploaded_file.read().decode('utf-8')
            
            # Create document
            document = Document.objects.create(
                filename=uploaded_file.name,
                content=content,
                file_size=len(content)
            )
            
            # Process document: chunk and embed
            embedding_service = EmbeddingService()
            chunks = embedding_service.chunk_text(content)
            
            # Generate embeddings for all chunks
            embeddings = embedding_service.generate_embeddings_batch(chunks)
            
            # Create chunk records
            chunk_objects = []
            for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_objects.append(
                    DocumentChunk(
                        document=document,
                        chunk_text=chunk_text,
                        chunk_index=idx,
                        embedding=embedding
                    )
                )
            
            # Bulk create chunks
            DocumentChunk.objects.bulk_create(chunk_objects)
            
            logger.info(f"Created document {document.id} with {len(chunks)} chunks")
            
            # Return document details
            response_serializer = DocumentSerializer(document)
            return Response(
                response_serializer.data,
                status=status.HTTP_201_CREATED
            )
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return Response(
                {'error': f'Failed to process document: {str(e)}'},
                status=status.HTTP_400_BAD_REQUEST
            )


@method_decorator(csrf_exempt, name='dispatch')
@api_view(['POST'])
async def chat_stream_view(request):
    """
    Handle chat requests with streaming responses
    
    POST /api/chat/
    Body: {
        "message": "User message",
        "session_id": "uuid" (optional),
        "use_documents": true (optional),
        "document_ids": ["uuid1", "uuid2"] (optional)
    }
    
    Returns: Server-Sent Events stream
    """
    try:
        # Validate request
        serializer = ChatRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {'error': serializer.errors},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        data = serializer.validated_data
        user_message = data['message']
        session_id = data.get('session_id')
        use_documents = data.get('use_documents', True)
        document_ids = data.get('document_ids', [])
        
        # Create or get session
        if session_id:
            try:
                session = await sync_to_async(ChatSession.objects.get)(id=session_id)
            except ChatSession.DoesNotExist:
                session = await sync_to_async(ChatSession.objects.create)()
        else:
            session = await sync_to_async(ChatSession.objects.create)()
        
        # Save user message
        await sync_to_async(ChatMessage.objects.create)(
            session=session,
            role='user',
            content=user_message
        )
        
        # Create streaming response
        async def event_stream():
            """Generate SSE events"""
            try:
                # Send session ID first
                yield f"data: {json.dumps({'session_id': str(session.id)})}\n\n"
                
                # Initialize LangGraph service
                langgraph_service = LangGraphService()
                
                # Accumulate response for saving
                full_response = ""
                
                # Stream tokens from LLM
                async for token in langgraph_service.stream_response(
                    session_id=str(session.id),
                    user_message=user_message,
                    use_documents=use_documents,
                    document_ids=document_ids
                ):
                    full_response += token
                    yield f"data: {json.dumps({'token': token})}\n\n"
                
                # Save assistant response
                await sync_to_async(ChatMessage.objects.create)(
                    session=session,
                    role='assistant',
                    content=full_response
                )
                
                # Send completion event
                yield f"data: {json.dumps({'done': True, 'full_response': full_response})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in event stream: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        # Return streaming response
        response = StreamingHttpResponse(
            event_stream(),
            content_type='text/event-stream'
        )
        response['Cache-Control'] = 'no-cache'
        response['X-Accel-Buffering'] = 'no'
        return response
        
    except Exception as e:
        logger.error(f"Error in chat_stream_view: {e}")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
def chat_view(request):
    """
    Non-streaming chat endpoint (for testing)
    
    POST /api/chat-sync/
    """
    try:
        serializer = ChatRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {'error': serializer.errors},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        data = serializer.validated_data
        user_message = data['message']
        session_id = data.get('session_id')
        use_documents = data.get('use_documents', True)
        document_ids = data.get('document_ids', [])
        
        # Create or get session
        if session_id:
            try:
                session = ChatSession.objects.get(id=session_id)
            except ChatSession.DoesNotExist:
                session = ChatSession.objects.create()
        else:
            session = ChatSession.objects.create()
        
        # Save user message
        ChatMessage.objects.create(
            session=session,
            role='user',
            content=user_message
        )
        
        # Get response from LangGraph
        langgraph_service = LangGraphService()
        response_text = langgraph_service.get_response_sync(
            session_id=str(session.id),
            user_message=user_message,
            use_documents=use_documents,
            document_ids=document_ids
        )
        
        # Save assistant response
        ChatMessage.objects.create(
            session=session,
            role='assistant',
            content=response_text
        )
        
        return Response({
            'session_id': str(session.id),
            'message': response_text
        })
        
    except Exception as e:
        logger.error(f"Error in chat_view: {e}")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )