from rest_framework import serializers
from .models import ChatSession, ChatMessage, Document


class ChatMessageSerializer(serializers.ModelSerializer):
    """Serializer for chat messages"""
    class Meta:
        model = ChatMessage
        fields = ['id', 'role', 'content', 'tokens_used', 'created_at', 'metadata']
        read_only_fields = ['id', 'created_at', 'tokens_used']


class ChatSessionSerializer(serializers.ModelSerializer):
    """Serializer for chat sessions"""
    messages = ChatMessageSerializer(many=True, read_only=True)
    message_count = serializers.SerializerMethodField()
    
    class Meta:
        model = ChatSession
        fields = ['id', 'title', 'created_at', 'updated_at', 'metadata', 
                  'messages', 'message_count']
        read_only_fields = ['id', 'created_at', 'updated_at']
    
    def get_message_count(self, obj):
        return obj.messages.count()


class ChatRequestSerializer(serializers.Serializer):
    """Serializer for incoming chat requests"""
    message = serializers.CharField(required=True, max_length=4000)
    session_id = serializers.UUIDField(required=False, allow_null=True)
    use_documents = serializers.BooleanField(default=True)
    document_ids = serializers.ListField(
        child=serializers.UUIDField(),
        required=False,
        allow_empty=True
    )


class DocumentSerializer(serializers.ModelSerializer):
    """Serializer for documents"""
    chunk_count = serializers.SerializerMethodField()
    
    class Meta:
        model = Document
        fields = ['id', 'filename', 'file_size', 'uploaded_at', 
                  'metadata', 'chunk_count']
        read_only_fields = ['id', 'uploaded_at', 'file_size']
    
    def get_chunk_count(self, obj):
        return obj.chunks.count()


class DocumentUploadSerializer(serializers.Serializer):
    """Serializer for document upload"""
    file = serializers.FileField(required=True)
    
    def validate_file(self, value):
        """Validate uploaded file"""
        if not value.name.endswith('.txt'):
            raise serializers.ValidationError("Only .txt files are supported")
        
        # Check file size (max 10MB)
        if value.size > 10 * 1024 * 1024:
            raise serializers.ValidationError("File size must be less than 10MB")
        
        return value