"""
Chat interface management for gut health coach
"""
from typing import List, Dict, Any, Optional, Tuple
import gradio as gr
from datetime import datetime
import logging

# Import local components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.rag_pipeline import GutHealthRAGPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GutHealthChatInterface:
    """
    Chat interface handler for gut health coaching
    """
    
    def __init__(self, rag_pipeline: GutHealthRAGPipeline):
        """
        Initialize chat interface
        
        Args:
            rag_pipeline: RAG pipeline instance
        """
        self.rag_pipeline = rag_pipeline
        self.session_stats = {
            "messages_sent": 0,
            "session_start": datetime.now(),
            "topics_discussed": [],
            "user_satisfaction": None
        }
        
        # Welcome message
        self.welcome_message = """
        👋 **Welcome to your Gut Health Coach!**
        
        I'm here to provide empathetic, evidence-based guidance for your gut health concerns. I can help you understand:
        
        • **Digestive symptoms** and their potential causes
        • **Gut-friendly foods** and dietary approaches  
        • **Lifestyle factors** that impact digestive health
        • **When to seek professional medical care**
        
        Feel free to ask me anything about your gut health. I'm here to listen and help! 💚
        
        **What would you like to know about your gut health today?**
        """
    
    def get_welcome_message(self) -> List[Dict[str, str]]:
        """
        Get welcome message for new conversations
        
        Returns:
            List containing welcome message
        """
        return [{"role": "assistant", "content": self.welcome_message}]
    
    def process_message(self, message: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Process user message and return updated conversation history
        
        Args:
            message: User message
            history: Current conversation history
            
        Returns:
            Updated conversation history
        """
        try:
            # Validate input
            if not message.strip():
                return history
            
            # Process through RAG pipeline
            response = self.rag_pipeline.process_query(message)
            
            # Update session stats
            self._update_session_stats(message, response)
            
            # Add to history
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response.assistant_response})
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            
            # Add error response
            error_response = ("I apologize, but I encountered an error processing your message. "
                            "Please try again or consult with a healthcare professional for your concerns.")
            
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_response})
            
            return history
    
    def _update_session_stats(self, message: str, response) -> None:
        """Update session statistics"""
        try:
            self.session_stats["messages_sent"] += 1
            
            # Extract topics from message
            topics = self._extract_topics(message)
            for topic in topics:
                if topic not in self.session_stats["topics_discussed"]:
                    self.session_stats["topics_discussed"].append(topic)
                    
        except Exception as e:
            logger.error(f"Failed to update session stats: {e}")
    
    def _extract_topics(self, message: str) -> List[str]:
        """Extract health topics from message"""
        try:
            topics = []
            message_lower = message.lower()
            
            # Common gut health topics
            topic_keywords = {
                "bloating": ["bloating", "bloated", "gas", "gassy"],
                "ibs": ["ibs", "irritable bowel", "spastic colon"],
                "constipation": ["constipation", "constipated", "hard stool"],
                "diarrhea": ["diarrhea", "loose stool", "frequent bowel"],
                "acid_reflux": ["acid reflux", "heartburn", "gerd"],
                "probiotics": ["probiotics", "good bacteria", "microbiome"],
                "diet": ["diet", "food", "eating", "nutrition"],
                "stress": ["stress", "anxiety", "worried", "nervous"],
                "pain": ["pain", "cramps", "cramping", "ache"]
            }
            
            for topic, keywords in topic_keywords.items():
                if any(keyword in message_lower for keyword in keywords):
                    topics.append(topic)
            
            return topics
            
        except Exception as e:
            logger.error(f"Failed to extract topics: {e}")
            return []
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get current session statistics
        
        Returns:
            Dictionary of session statistics
        """
        try:
            session_duration = datetime.now() - self.session_stats["session_start"]
            
            return {
                **self.session_stats,
                "session_duration_minutes": session_duration.total_seconds() / 60,
                "pipeline_metrics": self.rag_pipeline.get_pipeline_metrics()
            }
            
        except Exception as e:
            logger.error(f"Failed to get session stats: {e}")
            return {"error": str(e)}
    
    def reset_session(self) -> None:
        """Reset session statistics"""
        try:
            self.session_stats = {
                "messages_sent": 0,
                "session_start": datetime.now(),
                "topics_discussed": [],
                "user_satisfaction": None
            }
            
            # Clear RAG pipeline conversation history
            self.rag_pipeline.clear_conversation_history()
            
        except Exception as e:
            logger.error(f"Failed to reset session: {e}")
    
    def export_conversation(self, history: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Export conversation with metadata
        
        Args:
            history: Conversation history
            
        Returns:
            Export data dictionary
        """
        try:
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "conversation": history,
                "session_stats": self.get_session_stats(),
                "pipeline_metrics": self.rag_pipeline.get_pipeline_metrics()
            }
            
            return export_data
            
        except Exception as e:
            logger.error(f"Failed to export conversation: {e}")
            return {"error": str(e)}
