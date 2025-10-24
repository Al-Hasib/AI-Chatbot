import os
from typing import TypedDict, List, Dict, AsyncIterator
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from ..models import ChatMessage, DocumentChunk
from .embedding_service import EmbeddingService
import logging

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    """State for LangGraph workflow"""
    session_id: str
    user_message: str
    chat_history: List[Dict[str, str]]
    retrieved_context: str
    final_response: str
    use_documents: bool
    document_ids: List[str]


class LangGraphService:
    """Service for orchestrating AI chat with LangGraph"""
    
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.1-70b-versatile",
            temperature=0.7,
            api_key=os.getenv("GROQ_API_KEY"),
            streaming=True
        )
        self.embedding_service = EmbeddingService()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("load_history", self.load_history)
        workflow.add_node("retrieve_documents", self.retrieve_documents)
        workflow.add_node("generate_response", self.generate_response)
        
        # Define edges
        workflow.set_entry_point("load_history")
        workflow.add_edge("load_history", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()
    
    def load_history(self, state: GraphState) -> GraphState:
        """Load recent chat history from database"""
        try:
            if not state.get('session_id'):
                state['chat_history'] = []
                return state
            
            # Get last 10 messages for context
            messages = ChatMessage.objects.filter(
                session_id=state['session_id']
            ).order_by('-created_at')[:10]
            
            # Reverse to maintain chronological order
            state['chat_history'] = [
                {"role": msg.role, "content": msg.content}
                for msg in reversed(messages)
            ]
            
            logger.info(f"Loaded {len(state['chat_history'])} messages from history")
            
        except Exception as e:
            logger.error(f"Error loading history: {e}")
            state['chat_history'] = []
        
        return state
    
    def retrieve_documents(self, state: GraphState) -> GraphState:
        """Retrieve relevant document chunks"""
        try:
            if not state.get('use_documents', True):
                state['retrieved_context'] = ""
                return state
            
            # Generate embedding for user query
            query_embedding = self.embedding_service.generate_embedding(
                state['user_message']
            )
            
            # Build query for vector similarity search
            chunks_query = DocumentChunk.objects.all()
            
            # Filter by specific documents if provided
            if state.get('document_ids'):
                chunks_query = chunks_query.filter(
                    document_id__in=state['document_ids']
                )
            
            # Perform vector similarity search using pgvector
            # Using cosine distance: 1 - (embedding <=> query_embedding)
            chunks = chunks_query.order_by(
                self.embedding_service.cosine_distance(query_embedding)
            )[:5]
            
            # Format retrieved context
            if chunks:
                context_parts = []
                for i, chunk in enumerate(chunks, 1):
                    context_parts.append(
                        f"[Document {i}: {chunk.document.filename}]\n{chunk.chunk_text}"
                    )
                state['retrieved_context'] = "\n\n".join(context_parts)
                logger.info(f"Retrieved {len(chunks)} relevant chunks")
            else:
                state['retrieved_context'] = ""
                logger.info("No relevant documents found")
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            state['retrieved_context'] = ""
        
        return state
    
    def generate_response(self, state: GraphState) -> GraphState:
        """Generate AI response using LLM"""
        # This is a placeholder - actual streaming happens in the view
        # This node just validates the state
        state['final_response'] = ""
        return state
    
    async def stream_response(
        self, 
        session_id: str,
        user_message: str,
        use_documents: bool = True,
        document_ids: List[str] = None
    ) -> AsyncIterator[str]:
        """
        Stream AI response token by token
        
        This method processes the workflow and streams the LLM response
        """
        try:
            # Initialize state
            initial_state = {
                'session_id': session_id,
                'user_message': user_message,
                'chat_history': [],
                'retrieved_context': '',
                'final_response': '',
                'use_documents': use_documents,
                'document_ids': document_ids or []
            }
            
            # Run workflow up to response generation
            state = await self._run_workflow_async(initial_state)
            
            # Build messages for LLM
            messages = self._build_llm_messages(state)
            
            # Stream response from LLM
            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    yield chunk.content
                    
        except Exception as e:
            logger.error(f"Error in stream_response: {e}")
            yield f"\n\n[Error: {str(e)}]"
    
    async def _run_workflow_async(self, state: GraphState) -> GraphState:
        """Run the LangGraph workflow asynchronously"""
        # Run load_history
        state = self.load_history(state)
        
        # Run retrieve_documents
        state = self.retrieve_documents(state)
        
        return state
    
    def _build_llm_messages(self, state: GraphState) -> List:
        """Build message list for LLM"""
        messages = []
        
        # System message with instructions
        system_prompt = """You are a helpful AI assistant. You provide accurate, 
        concise, and friendly responses. If you're given document context, use it 
        to answer questions accurately. Always cite which document you're referencing."""
        
        if state['retrieved_context']:
            system_prompt += f"\n\nRelevant Context:\n{state['retrieved_context']}"
        
        messages.append(SystemMessage(content=system_prompt))
        
        # Add chat history
        for msg in state['chat_history']:
            if msg['role'] == 'user':
                messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                messages.append(AIMessage(content=msg['content']))
        
        # Add current user message
        messages.append(HumanMessage(content=state['user_message']))
        
        return messages
    
    def get_response_sync(
        self,
        session_id: str,
        user_message: str,
        use_documents: bool = True,
        document_ids: List[str] = None
    ) -> str:
        """
        Get complete AI response (non-streaming, for testing)
        """
        try:
            initial_state = {
                'session_id': session_id,
                'user_message': user_message,
                'chat_history': [],
                'retrieved_context': '',
                'final_response': '',
                'use_documents': use_documents,
                'document_ids': document_ids or []
            }
            
            # Run workflow
            state = self.load_history(initial_state)
            state = self.retrieve_documents(state)
            
            # Build messages
            messages = self._build_llm_messages(state)
            
            # Get response
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error in get_response_sync: {e}")
            return f"Error generating response: {str(e)}"