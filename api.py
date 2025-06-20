"""
Minimal production-ready chatbot API with memory using Pinecone.
"""
import os
import json
import time
import uvicorn
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv

# Import the Memory Agent functionality
from my_agent import process_query_with_memory, memory_service

# -----------------------------------
# Environment and Configuration Setup
# -----------------------------------

load_dotenv()

# -----------------------------------
# Database Setup
# -----------------------------------

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chat_history.db")

engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class ChatMessage(Base):
    """Database model for storing chat messages"""
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(String, index=True, nullable=False)
    speaker = Column(String, nullable=False)  # "User" or "Assistant"
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Database helper functions
def load_conversation_history(chat_id: str, db: Session) -> List[Tuple[str, str]]:
    """Retrieve conversation history for a given chat_id from the database."""
    messages = db.query(ChatMessage).filter(ChatMessage.chat_id == chat_id).order_by(ChatMessage.created_at).all()
    return [(msg.speaker, msg.message) for msg in messages]

def save_message(chat_id: str, speaker: str, message: str, db: Session) -> None:
    """Save a chat message to the database."""
    new_msg = ChatMessage(chat_id=chat_id, speaker=speaker, message=message)
    db.add(new_msg)
    db.commit()

# -----------------------------------
# Request and Response Models
# -----------------------------------

class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="Unique identifier for the user")
    message: str = Field(..., min_length=1, description="The user's message")

    @field_validator("user_id")
    @classmethod
    def user_id_not_empty(cls, v):
        if not v.strip():
            raise ValueError("user_id must not be empty or whitespace")
        return v

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v):
        if not v.strip():
            raise ValueError("message must not be empty or whitespace")
        return v

class ChatResponse(BaseModel):
    user_id: str
    message: str
    response: str
    conversation_history: List[Tuple[str, str]]
    memory_retrieved: bool = Field(default=False, description="Whether memories were retrieved for this response")
    memory_count: int = Field(default=0, description="Number of memories retrieved")

class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: str

# -----------------------------------
# FastAPI Application
# -----------------------------------

app = FastAPI(
    title="Memory Chatbot API",
    description="Production-ready chatbot with memory using Pinecone vector database",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only - restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------
# API Endpoints
# -----------------------------------

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint to check if API is running"""
    return HealthResponse(
        status="online",
        message="Memory Chatbot API is running",
        timestamp=datetime.now().isoformat()
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="API is operational",
        timestamp=datetime.now().isoformat()
    )

@app.get("/info")
async def api_info():
    """
    Get comprehensive API information and usage examples
    """
    return {
        "name": "Memory Chatbot API",
        "version": "1.0.0",
        "description": "Production-ready chatbot with autonomous memory using Pinecone vector database",
        "features": {
            "memory_storage": "Every message automatically stored in Pinecone vector database",
            "autonomous_decisions": "AI decides when to retrieve memories based on message content",
            "conversation_history": "SQLite database for structured conversation storage",
            "memory_analytics": "Endpoints for testing and analyzing memory functionality"
        },
        "endpoints": {
            "chat": {
                "POST /chat": "Main chat endpoint with full memory functionality",
                "POST /chat/simple": "Lightweight chat without database persistence",
                "GET /chat/history/{user_id}": "Get conversation history",
                "DELETE /chat/history/{user_id}": "Clear conversation history"
            },
            "memory": {
                "GET /memories/test/{user_id}?query=text": "Test what memories would be retrieved",
                "GET /memories/stats/{user_id}": "Get memory statistics for a user",
                "POST /memories/store": "Manually store a memory for testing"
            },
            "system": {
                "GET /": "Root endpoint",
                "GET /health": "Health check",
                "GET /info": "This endpoint - API information"
            }
        },
        "examples": {
            "basic_chat": {
                "url": "POST /chat",
                "body": {
                    "user_id": "user123",
                    "message": "Hello! I love chocolate ice cream."
                }
            },
            "memory_trigger": {
                "url": "POST /chat", 
                "body": {
                    "user_id": "user123",
                    "message": "What's my favorite ice cream?"
                },
                "note": "This will trigger memory retrieval due to 'my favorite' keywords"
            },
            "test_memory": {
                "url": "GET /memories/test/user123?query=favorite ice cream",
                "note": "Test what memories would be retrieved for this query"
            }
        },
        "memory_triggers": [
            "remember", "recall", "mentioned", "discussed", "my favorite", 
            "what do i", "what's my", "tell me about me", "we talked"
        ]
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Process a chat message with autonomous memory functionality
    
    The agent will:
    1. Store the user's message in Pinecone vector database
    2. Autonomously decide whether to retrieve relevant memories based on message content
    3. Generate a response using retrieved context if relevant
    4. Store the response in memory for future reference
    5. Return metadata about memory usage
    """
    try:
        # Load conversation history from database
        conversation_history = load_conversation_history(request.user_id, db)
        
        # Save user message to database
        save_message(request.user_id, "User", request.message, db)
        
        # Process through memory-enabled agent
        response = await process_query_with_memory(
            user_id=request.user_id,
            message=request.message,
            conversation_history=conversation_history
        )
        
        # Check if any memories exist for this user (for metadata)
        memories = memory_service.retrieve_memories(request.user_id, request.message, top_k=3)
        memory_count = len(memories)
        would_retrieve = memory_count > 0
        
        # Save assistant response to database
        save_message(request.user_id, "Assistant", response, db)
        
        # Get updated conversation history
        updated_history = load_conversation_history(request.user_id, db)
        
        return ChatResponse(
            user_id=request.user_id,
            message=request.message,
            response=response,
            conversation_history=updated_history,
            memory_retrieved=would_retrieve,
            memory_count=memory_count
        )
        
    except Exception as e:
        print(f"Error in chat_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/chat/history/{user_id}")
async def get_chat_history(user_id: str, db: Session = Depends(get_db)):
    """
    Get conversation history for a specific user
    """
    try:
        conversation_history = load_conversation_history(user_id, db)
        
        return {
            "user_id": user_id,
            "conversation_history": conversation_history,
            "message_count": len(conversation_history)
        }
        
    except Exception as e:
        print(f"Error retrieving chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat/history/{user_id}")
async def clear_chat_history(user_id: str, db: Session = Depends(get_db)):
    """
    Clear conversation history for a specific user from database
    Note: This does not clear memories from Pinecone vector database
    """
    try:
        # Delete messages from database
        db.query(ChatMessage).filter(ChatMessage.chat_id == user_id).delete()
        db.commit()
        
        return {
            "user_id": user_id,
            "message": "Chat history cleared from database",
            "note": "Memories in vector database are preserved"
        }
        
    except Exception as e:
        print(f"Error clearing chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------
# Main Entry Point
# -----------------------------------

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
