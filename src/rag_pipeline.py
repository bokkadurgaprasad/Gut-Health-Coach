"""
Main RAG pipeline orchestration for gut health coaching
"""
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass, asdict

# Import LangChain components
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.callbacks import get_openai_callback

# Import local components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import *
from config.prompts import *
from config.medical_disclaimers import *
from src.vectorstore import GutHealthVectorStore
from src.llm_interface import BioMistralInterface, MedicalResponseProcessor
from src.tone_engineering import EmpathyEngine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    """Structured RAG response"""
    user_query: str
    assistant_response: str
    retrieved_context: str
    response_time: float
    sources_used: List[str]
    empathy_score: float
    safety_flags: List[str]
    tokens_used: int
    timestamp: str

class GutHealthRAGPipeline:
    """
    Complete RAG pipeline for gut health coaching with empathy
    """
    
    def __init__(self):
        """Initialize the RAG pipeline"""
        self.vector_store = None
        self.llm_interface = None
        self.response_processor = None
        self.empathy_engine = None
        self.conversation_memory = None
        self.conversation_history = []
        
        # Performance metrics
        self.total_queries = 0
        self.average_response_time = 0.0
        self.success_rate = 0.0
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all pipeline components"""
        try:
            logger.info("Initializing RAG pipeline components...")
            
            # Initialize vector store
            logger.info("Loading vector store...")
            self.vector_store = GutHealthVectorStore()
            
            # Try to load existing vector store
            try:
                self.vector_store.load_vector_store(str(FAISS_INDEX_DIR))
                logger.info("Vector store loaded from disk")
            except:
                logger.info("Creating new vector store from embeddings...")
                embeddings_dir = EMBEDDINGS_DIR
                latest_embeddings = max(embeddings_dir.glob("embeddings_*.npy"))
                latest_metadata = max(embeddings_dir.glob("metadata_*.json"))
                
                self.vector_store.create_from_embeddings(
                    str(latest_embeddings),
                    str(latest_metadata)
                )
                self.vector_store.save_vector_store(str(FAISS_INDEX_DIR))
                logger.info("Vector store created and saved")
            
            # Initialize LLM interface
            logger.info("Loading BioMistral model...")
            self.llm_interface = BioMistralInterface()
            
            # Initialize response processor
            logger.info("Setting up empathy processing...")
            self.response_processor = MedicalResponseProcessor()
            self.empathy_engine = EmpathyEngine()
            
            # Initialize conversation memory
            self.conversation_memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            logger.info("All RAG pipeline components initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise
    
    def process_query(self, user_query: str, user_id: str = "default") -> RAGResponse:
        """
        Process a user query through the complete RAG pipeline
        
        Args:
            user_query: User's question about gut health
            user_id: User identifier for conversation tracking
            
        Returns:
            RAGResponse object with complete response data
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing query: {user_query[:50]}...")
            
            # Step 1: Retrieve relevant context
            logger.info("Retrieving relevant medical context...")
            context = self.vector_store.get_context_for_query(user_query)
            
            # Step 2: Get sources for citation
            search_results = self.vector_store.similarity_search_with_relevance_scores(
                user_query, k=5
            )
            sources_used = [result.source for result in search_results]
            
            # Step 3: Generate empathetic response
            logger.info("Generating empathetic response...")
            
            # Create RAG prompt
            rag_prompt = RAG_PROMPT_TEMPLATE.format(
                context=context,
                question=user_query
            )
            
            # Generate response using BioMistral
            model_response = self.llm_interface.generate_response(
                rag_prompt,
                system_prompt=MEDICAL_SYSTEM_PROMPT
            )
            
            # Step 4: Enhance with empathy
            logger.info("Enhancing response with empathy...")
            enhanced_response = self.response_processor.enhance_response_with_empathy(
                model_response.content, user_query
            )
            
            # Step 5: Apply empathy scoring
            empathy_score = self.empathy_engine.calculate_empathy_score(
                enhanced_response, user_query
            )
            
            # Step 6: Safety checks
            safety_flags = self.empathy_engine.check_safety_flags(
                user_query, enhanced_response
            )
            
            # Step 7: Update conversation memory
            self.conversation_memory.chat_memory.add_user_message(user_query)
            self.conversation_memory.chat_memory.add_ai_message(enhanced_response)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Create response object
            rag_response = RAGResponse(
                user_query=user_query,
                assistant_response=enhanced_response,
                retrieved_context=context,
                response_time=response_time,
                sources_used=sources_used,
                empathy_score=empathy_score,
                safety_flags=safety_flags,
                tokens_used=model_response.tokens_used,
                timestamp=datetime.now().isoformat()
            )
            
            # Update metrics
            self._update_metrics(response_time, success=True)
            
            # Log conversation
            self._log_conversation(rag_response)
            
            logger.info(f"Query processed successfully in {response_time:.2f}s")
            return rag_response
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            
            # Create error response
            error_response = RAGResponse(
                user_query=user_query,
                assistant_response=f"I apologize, but I'm having trouble processing your question right now. Please try again or consult with a healthcare professional. {PRIMARY_DISCLAIMER}",
                retrieved_context="",
                response_time=time.time() - start_time,
                sources_used=[],
                empathy_score=0.0,
                safety_flags=["system_error"],
                tokens_used=0,
                timestamp=datetime.now().isoformat()
            )
            
            self._update_metrics(time.time() - start_time, success=False)
            return error_response
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get formatted conversation history"""
        try:
            messages = self.conversation_memory.chat_memory.messages
            history = []
            
            for message in messages:
                if isinstance(message, HumanMessage):
                    history.append({
                        "role": "user",
                        "content": message.content,
                        "timestamp": datetime.now().isoformat()
                    })
                elif isinstance(message, AIMessage):
                    history.append({
                        "role": "assistant", 
                        "content": message.content,
                        "timestamp": datetime.now().isoformat()
                    })
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []
    
    def clear_conversation_history(self):
        """Clear conversation memory"""
        try:
            self.conversation_memory.clear()
            self.conversation_history = []
            logger.info("Conversation history cleared")
        except Exception as e:
            logger.error(f"Failed to clear conversation history: {e}")
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics"""
        return {
            "total_queries_processed": self.total_queries,
            "average_response_time": self.average_response_time,
            "success_rate": self.success_rate,
            "vector_store_size": len(self.vector_store.vector_store.docstore._dict) if self.vector_store.vector_store else 0,
            "model_loaded": self.llm_interface.model is not None,
            "last_updated": datetime.now().isoformat()
        }
    
    def _update_metrics(self, response_time: float, success: bool):
        """Update performance metrics"""
        self.total_queries += 1
        
        # Update average response time
        if self.total_queries == 1:
            self.average_response_time = response_time
        else:
            self.average_response_time = (
                (self.average_response_time * (self.total_queries - 1) + response_time) 
                / self.total_queries
            )
        
        # Update success rate
        if success:
            self.success_rate = (
                (self.success_rate * (self.total_queries - 1) + 1.0) 
                / self.total_queries
            )
        else:
            self.success_rate = (
                (self.success_rate * (self.total_queries - 1) + 0.0) 
                / self.total_queries
            )
    
    def _log_conversation(self, response: RAGResponse):
        """Log conversation for analysis"""
        try:
            log_entry = {
                "timestamp": response.timestamp,
                "user_query": response.user_query,
                "response_length": len(response.assistant_response),
                "response_time": response.response_time,
                "empathy_score": response.empathy_score,
                "safety_flags": response.safety_flags,
                "sources_used": response.sources_used,
                "tokens_used": response.tokens_used
            }
            
            self.conversation_history.append(log_entry)
            
            # Save to file periodically
            if len(self.conversation_history) % 10 == 0:
                self._save_conversation_log()
                
        except Exception as e:
            logger.error(f"Failed to log conversation: {e}")
    
    def _save_conversation_log(self):
        """Save conversation log to file"""
        try:
            log_file = Path("logs") / f"conversation_log_{datetime.now().strftime('%Y%m%d')}.json"
            log_file.parent.mkdir(exist_ok=True)
            
            with open(log_file, 'w') as f:
                json.dump(self.conversation_history, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save conversation log: {e}")

def main():
    """
    Test the complete RAG pipeline
    """
    try:
        # Initialize pipeline
        logger.info("🏗️ Setting up RAG pipeline...")
        rag_pipeline = GutHealthRAGPipeline()
        
        # Test queries
        test_queries = [
            "I've been experiencing bloating and gas after meals. What could be causing this?",
            "What are some natural ways to improve gut health?",
            "I have IBS and I'm wondering about the best diet approaches.",
            "Can stress really affect my digestive system?",
            "I'm having severe abdominal pain - should I be worried?"
        ]
        
        logger.info("🧪 Testing RAG pipeline with sample queries...")
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Test {i}/{len(test_queries)}: {query}")
            logger.info(f"{'='*60}")
            
            # Process query
            response = rag_pipeline.process_query(query)
            
            # Display results
            logger.info(f"Response time: {response.response_time:.2f}s")
            logger.info(f"Empathy score: {response.empathy_score:.2f}")
            logger.info(f"Sources used: {', '.join(response.sources_used)}")
            logger.info(f"Safety flags: {response.safety_flags}")
            logger.info(f"Response preview:\n{response.assistant_response[:300]}...")
        
        # Display pipeline metrics
        logger.info(f"\n{'='*60}")
        logger.info("Pipeline Performance Metrics:")
        logger.info(f"{'='*60}")
        
        metrics = rag_pipeline.get_pipeline_metrics()
        for key, value in metrics.items():
            logger.info(f"{key}: {value}")
        
        logger.info("\nRAG pipeline testing completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline testing failed: {e}")
        raise

if __name__ == "__main__":
    main()
