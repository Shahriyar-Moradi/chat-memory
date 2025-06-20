import os
import time
import uuid
import hashlib
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from agents import Agent, Runner, function_tool, ModelSettings
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
try:
    from pinecone import ServerlessSpec
except ImportError:
    ServerlessSpec = None


import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure API clients
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
MEMORY_INDEX_NAME = os.getenv("MEMORY_INDEX_NAME", "chatbot-memory")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing required API keys. Please set OPENAI_API_KEY and PINECONE_API_KEY in your .env file")

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Model configuration
MODEL = os.getenv('MODEL_CHOICE', 'gpt-4o-mini')

# --- Initialize Pinecone for memory storage ---
def initialize_memory_index():
    """Initialize Pinecone index for storing conversation memories"""
    print(f"Initializing Pinecone memory index: {MEMORY_INDEX_NAME}")
    
    # Check if index exists, create if not
    existing_indexes = pc.list_indexes().names()
    if MEMORY_INDEX_NAME not in existing_indexes:
        print(f"Creating new Pinecone index: {MEMORY_INDEX_NAME}")
        
        # Create index based on available ServerlessSpec
        if ServerlessSpec is not None:
            pc.create_index(
                name=MEMORY_INDEX_NAME,
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        else:
            # Fallback for older pinecone versions or when ServerlessSpec is not available
            try:
                # Try with minimal parameters for older versions
                pc.create_index(
                    name=MEMORY_INDEX_NAME,
                    dimension=1536,
                    metric="cosine",
                    spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}  # Dictionary format fallback
                )
            except Exception as e:
                logger.warning(f"Failed to create index with serverless spec: {str(e)}")
                # Ultimate fallback - this might not work with newer Pinecone versions
                logger.info("Attempting basic index creation...")
                pc.create_index(
                    name=MEMORY_INDEX_NAME,
                    dimension=1536,
                    metric="cosine",
                    spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
                )
        
    # Wait for index to be ready
    while not pc.describe_index(MEMORY_INDEX_NAME).status.get('ready', False):
        print("Waiting for Pinecone index to be ready...")
        time.sleep(1)
        
    index = pc.Index(MEMORY_INDEX_NAME)
    print("Pinecone memory index is ready")
    return index

memory_index = initialize_memory_index()

# --- Memory Functions ---
def get_embedding(text: str) -> List[float]:
    """Generate embedding for text using OpenAI"""
    try:
        response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        return []

# Legacy function - now delegates to MemoryService
def store_message_in_memory(user_id: str, message: str, message_type: str = "user") -> None:
    """Store a message in Pinecone memory with metadata (legacy function)"""
    memory_service.store_message(user_id, message, message_type)

class MemoryService:
    """Service for managing memory storage and retrieval using Pinecone"""
    
    def __init__(self):
        self.index = memory_index
        
    def store_message(self, user_id: str, message: str, message_type: str = "user"):
        """Store a message in Pinecone memory with metadata"""
        try:
            # Create unique ID for this message
            timestamp = str(int(time.time()))
            message_id = f"{user_id}_{message_type}_{timestamp}_{hashlib.md5(message.encode()).hexdigest()[:8]}"
            
            # Generate embedding
            embedding = get_embedding(message)
            if not embedding:
                logger.warning("Failed to generate embedding for message")
                return
                
            # Store in Pinecone with better metadata structure
            self.index.upsert(
                vectors=[{
                    "id": message_id,
                    "values": embedding,
                    "metadata": {
                        "user_id": user_id,
                        "message": message,
                        "message_type": message_type,
                        "timestamp": timestamp
                    }
                }]
            )
            logger.info(f"Stored {message_type} message in memory for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error storing message in memory: {str(e)}")
    
    def get_all_user_memories(self, user_id: str, limit: int = 50) -> List[dict]:
        """Get all memories for a user for debugging purposes"""
        try:
            # Create a dummy query to search all vectors for this user
            dummy_embedding = [0.0] * 1536  # All zeros
            
            results = self.index.query(
                vector=dummy_embedding,
                top_k=limit,
                filter={"user_id": user_id},
                include_metadata=True
            )
            
            memories_info = []
            for match in results.matches:
                if hasattr(match, 'metadata') and match.metadata:
                    memories_info.append({
                        'id': match.id,
                        'score': match.score,
                        'message': match.metadata.get("message", ""),
                        'message_type': match.metadata.get("message_type", ""),
                        'timestamp': match.metadata.get("timestamp", "")
                    })
            
            return memories_info
            
        except Exception as e:
            logger.error(f"Error getting all user memories: {str(e)}")
            return []
    
    def retrieve_memories(self, user_id: str, query: str, top_k: int = 5) -> List[str]:
        """Retrieve relevant memories for a user based on query"""
        try:
            # Generate embedding for the query
            query_embedding = get_embedding(query)
            if not query_embedding:
                logger.warning("Failed to generate embedding for query")
                return []
                
            # Search for relevant memories
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter={"user_id": user_id},
                include_metadata=True
            )
            
            logger.info(f"Found {len(results.matches)} potential matches for user {user_id}")
            
            # Process results with more lenient threshold
            memories = []
            for i, match in enumerate(results.matches):
                logger.info(f"Match {i+1}: Score={match.score:.3f}")
                
                # Lower threshold to 0.5 for better recall, and include score info
                if match.score > 0.5:  
                    if hasattr(match, 'metadata') and match.metadata:
                        message = match.metadata.get("message", "")
                        message_type = match.metadata.get("message_type", "unknown")
                        timestamp = match.metadata.get("timestamp", "")
                        
                        if message:
                            # Add metadata for debugging
                            memory_info = f"{message} [Type: {message_type}, Score: {match.score:.3f}]"
                            memories.append(message)  # Just the message for now
                            logger.info(f"Added memory: {message[:50]}... (Score: {match.score:.3f})")
            
            logger.info(f"Retrieved {len(memories)} memories above threshold")
            return memories
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {str(e)}")
            return []

# Initialize memory service
memory_service = MemoryService()

@function_tool
def retrieve_relevant_memories(query: str, user_id: str, top_k: int = 5) -> str:
    """
    Retrieve relevant past memories for the user based on the current query.
    Use this when the user asks about past conversations or when context from previous interactions would be helpful.
    
    Args:
        query: The current user message or query to find relevant memories for
        user_id: The user's unique identifier
        top_k: Number of relevant memories to retrieve (default: 5)
        
    Returns:
        A formatted string containing relevant past memories
    """
    try:
        memories = memory_service.retrieve_memories(user_id, query, top_k)
        
        if not memories:
            return "No relevant past conversations found."
            
        # Format memories for better context
        formatted_memories = []
        for i, memory in enumerate(memories[:3], 1):  # Limit to top 3
            formatted_memories.append(f"{i}. {memory}")
            
        return "Relevant past memories:\n" + "\n".join(formatted_memories)
        
    except Exception as e:
        logger.error(f"Error in retrieve_relevant_memories tool: {str(e)}")
        return "Unable to retrieve memories at this time."



# --- Memory-enabled chatbot agent ---
memory_chatbot = Agent(
    name="Memory Chatbot",
    instructions="""
    You are a helpful AI assistant with memory capabilities.

    CRITICAL: On every new message, you MUST first decide whether to call the retrieve_relevant_memories tool to check for relevant past conversations.

    DECISION CRITERIA:
    - Use retrieve_relevant_memories tool when user asks about:
      • Past conversations ("remember", "recall", "what did we discuss")
      • Personal preferences ("my favorite", "what do I like", "tell me about me")
      • Previous interactions ("last time", "you mentioned", "we talked")

    WORKFLOW:
    1. First, evaluate if memories are relevant to the current message
    2. If relevant, call retrieve_relevant_memories tool to get past context
    3. Use retrieved memories naturally in your response
    4. Be conversational and don't explicitly mention memory lookups

    Always be helpful and maintain conversation continuity using available memory context when available and relevant.
    """,
    model=MODEL,
    tools=[retrieve_relevant_memories],
    model_settings=ModelSettings(temperature=0.3),
)

# --- Main processing function ---
async def process_query_with_memory(user_id: str, message: str, conversation_history: List[Tuple[str, str]] = None) -> str:
    """
    Process a user query with autonomous memory storage and retrieval
    
    Args:
        user_id: Unique identifier for the user
        message: The user's message
        conversation_history: Optional conversation history for context
        
    Returns:
        The assistant's response
    """
    try:
        # Store the user's message in memory
        memory_service.store_message(user_id, message, "user")
        
        # Build context for the agent
        full_query = f"User ID: {user_id}\n"
        
        # Add recent conversation history if available
        if conversation_history:
            recent_context = "\n".join([f"{speaker}: {msg}" for speaker, msg in conversation_history[-5:]])
            full_query += f"Recent conversation:\n{recent_context}\n\n"
        
        # Add current message
        full_query += f"Current message: {message}"
        
        # Process through the memory-enabled agent
        result = await Runner.run(memory_chatbot, full_query)
        response = result.final_output if hasattr(result, 'final_output') else str(result)
        
        # Store the assistant's response in memory
        memory_service.store_message(user_id, response, "assistant")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query with memory: {str(e)}")
        return "I apologize, but I'm having trouble processing your request right now. Please try again."


