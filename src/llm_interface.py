"""
BioMistral-7B model interface for medical conversations
"""
import os
import requests
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from dataclasses import dataclass
from datetime import datetime

try:
    from llama_cpp import Llama
except ImportError:
    print("llama-cpp-python not installed. Please install with: pip install llama-cpp-python")
    Llama = None

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import *
from config.prompts import *
from config.medical_disclaimers import *

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelResponse:
    """Structured model response"""
    content: str
    tokens_used: int
    response_time: float
    model_name: str
    temperature: float

class BioMistralInterface:
    """
    Interface for BioMistral-7B model using llama-cpp-python
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize BioMistral interface
        
        Args:
            model_path: Path to BioMistral GGUF model file
        """
        self.model_path = model_path or self._get_model_path()
        self.model = None
        self.model_config = {
            'n_ctx': 2048,  # Context window
            'n_threads': 4,  # CPU threads
            'n_gpu_layers': 0,  # CPU only
            'verbose': False
        }
        self.generation_config = {
            'temperature': 0.3,  # Conservative for medical accuracy
            'top_p': 0.8,
            'top_k': 40,
            'repeat_penalty': 1.1,
            'max_tokens': 600
        }
        
        # Initialize model
        self._initialize_model()
    
    def _get_model_path(self) -> str:
        """Get the path to BioMistral model"""
        possible_paths = [
            MODELS_DIR / "biomistral-7b-q4.gguf",
            MODELS_DIR / "BioMistral-7B.Q4_K_M.gguf",
            Path.cwd() / "models" / "biomistral-7b-q4.gguf"
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        # If no model found, provide download instructions
        logger.warning("BioMistral model not found. Please download from:")
        logger.warning("https://huggingface.co/MaziyarPanahi/BioMistral-7B-GGUF")
        logger.warning("Place the Q4_K_M.gguf file in the models/ directory")
        
        return str(possible_paths[0])  # Return first path for error handling
    
    def _initialize_model(self):
        """Initialize the BioMistral model"""
        try:
            if not Llama:
                raise ImportError("llama-cpp-python not available")
            
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            logger.info(f"Loading BioMistral model from: {self.model_path}")
            
            self.model = Llama(
                model_path=self.model_path,
                **self.model_config
            )
            
            logger.info("BioMistral model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize BioMistral model: {e}")
            logger.info("Fallback: Using OpenAI-compatible API if available")
            self.model = None
    
    def generate_response(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Generate response using BioMistral model
        
        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            **kwargs: Additional generation parameters
            
        Returns:
            ModelResponse object
        """
        start_time = datetime.now()
        
        try:
            # Merge generation config with kwargs
            gen_config = {**self.generation_config, **kwargs}
            
            # Format prompt with system context
            if system_prompt:
                formatted_prompt = f"<s>[INST] {system_prompt}\n\nUser: {prompt} [/INST]"
            else:
                formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            
            if self.model:
                # Use local model
                response = self.model(
                    formatted_prompt,
                    **gen_config
                )
                
                content = response['choices'][0]['text']
                tokens_used = response['usage']['total_tokens']
                
            else:
                # Fallback to simulated response for testing
                content = self._generate_fallback_response(prompt)
                tokens_used = len(content.split())
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            
            return ModelResponse(
                content=content.strip(),
                tokens_used=tokens_used,
                response_time=response_time,
                model_name="BioMistral-7B",
                temperature=gen_config.get('temperature', 0.3)
            )
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            # Return error response
            return ModelResponse(
                content="I apologize, but I'm having trouble generating a response right now. Please try again or consult with a healthcare professional for your concerns.",
                tokens_used=0,
                response_time=(datetime.now() - start_time).total_seconds(),
                model_name="BioMistral-7B",
                temperature=0.3
            )
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate fallback response for testing"""
        return f"""I understand you're asking about gut health. While I'd love to provide specific guidance, I'm currently in testing mode. 

For your question about "{prompt[:50]}...", I'd recommend:

1. **Consult a healthcare provider** - They can give personalized advice based on your specific situation
2. **Consider seeing a gastroenterologist** - For specialized digestive health concerns
3. **Keep a symptom diary** - Track what you eat and how you feel

{PRIMARY_DISCLAIMER}

Please remember that gut health is complex and individual. What works for one person may not work for another, so professional medical guidance is always best."""

class MedicalResponseProcessor:
    """
    Processes and enhances medical responses with empathy and safety
    """
    
    def __init__(self):
        self.empathy_templates = EMPATHY_TEMPLATES
        self.disclaimers = SITUATIONAL_DISCLAIMERS
        self.red_flags = RED_FLAG_SYMPTOMS
    
    def enhance_response_with_empathy(self, response: str, user_query: str) -> str:
        """
        Enhance response with empathetic elements
        
        Args:
            response: Original model response
            user_query: User's original query
            
        Returns:
            Enhanced response with empathy
        """
        try:
            # Check for red flag symptoms
            needs_urgent_care = self._check_for_red_flags(user_query)
            
            if needs_urgent_care:
                return self._generate_urgent_care_response(user_query)
            
            # Add empathetic introduction
            empathetic_intro = self._get_empathetic_intro(user_query)
            
            # Add appropriate disclaimer
            disclaimer = self._get_appropriate_disclaimer(user_query, response)
            
            # Combine elements
            enhanced_response = f"{empathetic_intro}\n\n{response}\n\n{disclaimer}"
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Failed to enhance response: {e}")
            return response
    
    def _check_for_red_flags(self, query: str) -> bool:
        """Check if query contains red flag symptoms"""
        query_lower = query.lower()
        return any(symptom in query_lower for symptom in self.red_flags)
    
    def _get_empathetic_intro(self, query: str) -> str:
        """Generate empathetic introduction"""
        # Simple template selection based on query content
        if any(word in query.lower() for word in ['worried', 'concerned', 'scared', 'anxious']):
            return "I understand this is concerning for you, and it's completely natural to seek answers about your health."
        elif any(word in query.lower() for word in ['pain', 'hurt', 'uncomfortable']):
            return "I can hear that you're experiencing discomfort, and I want to help you understand what might be happening."
        else:
            return "Thank you for reaching out about your gut health. I'm here to provide helpful, evidence-based information."
    
    def _get_appropriate_disclaimer(self, query: str, response: str) -> str:
        """Select appropriate disclaimer based on context"""
        query_lower = query.lower()
        response_lower = response.lower()
        
        if any(word in query_lower for word in ['emergency', 'urgent', 'severe']):
            return self.disclaimers['emergency_situations']
        elif any(word in query_lower for word in ['medication', 'drug', 'supplement']):
            return self.disclaimers['medication_interactions']
        elif any(word in response_lower for word in ['treatment', 'therapy', 'cure']):
            return self.disclaimers['treatment_suggestions']
        else:
            return self.disclaimers['symptom_assessment']
    
    def _generate_urgent_care_response(self, query: str) -> str:
        """Generate urgent care response for red flag symptoms"""
        return f"""**IMPORTANT: Seek Immediate Medical Attention**

I notice you're describing symptoms that may require urgent medical evaluation. Please:

1. **Contact emergency services (911)** if symptoms are severe
2. **Visit your nearest emergency room** for immediate assessment
3. **Call your doctor immediately** if they're available

Your symptoms warrant professional medical evaluation that cannot be provided through this chat.

{self.disclaimers['emergency_situations']}

Please prioritize your health and seek immediate professional care."""

def main():
    """
    Test the BioMistral interface
    """
    try:
        # Initialize interface
        llm_interface = BioMistralInterface()
        response_processor = MedicalResponseProcessor()
        
        # Test queries
        test_queries = [
            "I've been experiencing bloating for the past week. What could be causing this?",
            "What are some natural ways to improve gut health?",
            "I have severe abdominal pain and blood in my stool",
            "Can probiotics help with IBS symptoms?"
        ]
        
        logger.info("🧪 Testing BioMistral interface:")
        
        for query in test_queries:
            logger.info(f"\n📝 Query: {query}")
            
            # Generate response
            response = llm_interface.generate_response(
                query,
                system_prompt=MEDICAL_SYSTEM_PROMPT
            )
            
            # Enhance with empathy
            enhanced_response = response_processor.enhance_response_with_empathy(
                response.content, query
            )
            
            logger.info(f"Response time: {response.response_time:.2f}s")
            logger.info(f"Tokens used: {response.tokens_used}")
            logger.info(f"Enhanced response:\n{enhanced_response[:200]}...")
        
        logger.info("\nBioMistral interface testing completed!")
        
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        raise

if __name__ == "__main__":
    main()
