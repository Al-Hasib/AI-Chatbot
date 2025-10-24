from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ChatSessionViewSet, DocumentViewSet, chat_stream_view, chat_view

router = DefaultRouter()
router.register(r'sessions', ChatSessionViewSet, basename='session')
router.register(r'documents', DocumentViewSet, basename='document')

urlpatterns = [
    path('', include(router.urls)),
    path('chat/', chat_stream_view, name='chat-stream'),
    path('chat-sync/', chat_view, name='chat-sync'),
]