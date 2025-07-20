"""
Empathy and tone engineering for medical conversations
"""
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime

# Import configuration
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import *
from config.prompts import *
from config.medical_disclaimers import *

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EmpathyMetrics:
    """Metrics for empathy analysis"""
    empathy_score: float
    validation_present: bool
    reassurance_present: bool
    actionable_guidance: bool
    appropriate_boundaries: bool
    emotional_words_count: int
    question_complexity: str

class EmpathyEngine:
    """
    Engine for analyzing and enhancing empathetic responses
    """
    
    def __init__(self):
        """Initialize empathy engine"""
        self.empathy_keywords = {
            'validation': [
                'understand', 'valid', 'natural', 'common', 'normal',
                'makes sense', 'understandable', 'legitimate', 'reasonable'
            ],
            'reassurance': [
                'help', 'support', 'manageable', 'treatable', 'hope',
                'better', 'improve', 'relief', 'solutions', 'positive'
            ],
            'emotional_support': [
                'concern', 'worry', 'anxious', 'scared', 'frustrated',
                'difficult', 'challenging', 'overwhelming', 'stressful'
            ],
            'empowerment': [
                'control', 'choice', 'decision', 'steps', 'action',
                'progress', 'improvement', 'empowered', 'capable'
            ]
        }
        
        self.medical_urgency_indicators = [
            'severe', 'intense', 'emergency', 'urgent', 'immediate',
            'worsening', 'blood', 'fever', 'persistent', 'chronic'
        ]
        
        self.emotional_intensity_words = [
            'very', 'extremely', 'really', 'incredibly', 'terribly',
            'awful', 'terrible', 'horrible', 'devastating', 'overwhelming'
        ]
    
    def calculate_empathy_score(self, response: str, user_query: str) -> float:
        """
        Calculate empathy score for a response
        
        Args:
            response: AI response to analyze
            user_query: Original user query
            
        Returns:
            Empathy score (0.0 to 1.0)
        """
        try:
            response_lower = response.lower()
            query_lower = user_query.lower()
            
            # Initialize score components
            validation_score = 0.0
            reassurance_score = 0.0
            personalization_score = 0.0
            boundary_score = 0.0
            
            # Check for validation elements
            validation_count = sum(1 for word in self.empathy_keywords['validation'] 
                                 if word in response_lower)
            validation_score = min(validation_count * 0.15, 0.3)
            
            # Check for reassurance elements
            reassurance_count = sum(1 for word in self.empathy_keywords['reassurance'] 
                                  if word in response_lower)
            reassurance_score = min(reassurance_count * 0.1, 0.25)
            
            # Check for personalization (addressing user's specific concern)
            personalization_score = self._calculate_personalization_score(
                response_lower, query_lower
            )
            
            # Check for appropriate boundaries
            boundary_score = self._calculate_boundary_score(response_lower)
            
            # Calculate total empathy score
            total_score = validation_score + reassurance_score + personalization_score + boundary_score
            
            # Normalize to 0.0-1.0 range
            empathy_score = min(total_score, 1.0)
            
            logger.debug(f"Empathy score calculated: {empathy_score:.3f}")
            return empathy_score
            
        except Exception as e:
            logger.error(f"Failed to calculate empathy score: {e}")
            return 0.5  # Default middle score
    
    def _calculate_personalization_score(self, response: str, query: str) -> float:
        """Calculate personalization score"""
        try:
            # Check if response addresses specific symptoms mentioned
            query_symptoms = self._extract_symptoms_from_query(query)
            
            personalization_score = 0.0
            
            # Award points for addressing specific symptoms
            for symptom in query_symptoms:
                if symptom in response:
                    personalization_score += 0.05
            
            # Award points for using "you" to address the user directly
            you_count = response.count('you')
            personalization_score += min(you_count * 0.02, 0.1)
            
            # Award points for acknowledging emotional state
            emotional_acknowledgment = any(word in response for word in 
                                         self.empathy_keywords['emotional_support'])
            if emotional_acknowledgment:
                personalization_score += 0.1
            
            return min(personalization_score, 0.25)
            
        except Exception as e:
            logger.error(f"Failed to calculate personalization score: {e}")
            return 0.0
    
    def _calculate_boundary_score(self, response: str) -> float:
        """Calculate appropriate medical boundary score"""
        try:
            boundary_score = 0.0
            
            # Check for medical disclaimers
            disclaimer_present = any(phrase in response for phrase in [
                'disclaimer', 'medical professional', 'healthcare provider',
                'doctor', 'not intended to replace', 'consult'
            ])
            
            if disclaimer_present:
                boundary_score += 0.1
            
            # Check for appropriate language (not overly medical)
            overly_medical = sum(1 for word in ['diagnosis', 'treatment', 'prescription', 
                                               'cure', 'disease'] if word in response)
            
            if overly_medical <= 2:  # Appropriate medical language usage
                boundary_score += 0.1
            
            return min(boundary_score, 0.2)
            
        except Exception as e:
            logger.error(f"Failed to calculate boundary score: {e}")
            return 0.0
    
    def _extract_symptoms_from_query(self, query: str) -> List[str]:
        """Extract potential symptoms from user query"""
        try:
            common_symptoms = [
                'bloating', 'gas', 'pain', 'cramping', 'nausea', 'vomiting',
                'diarrhea', 'constipation', 'heartburn', 'acid reflux',
                'indigestion', 'stomach ache', 'abdominal pain', 'discomfort',
                'burning', 'inflammation', 'swelling', 'fatigue', 'tired'
            ]
            
            found_symptoms = []
            for symptom in common_symptoms:
                if symptom in query.lower():
                    found_symptoms.append(symptom)
            
            return found_symptoms
            
        except Exception as e:
            logger.error(f"Failed to extract symptoms: {e}")
            return []
    
    def check_safety_flags(self, user_query: str, response: str) -> List[str]:
        """
        Check for safety flags in query and response
        
        Args:
            user_query: User's query
            response: AI response
            
        Returns:
            List of safety flags
        """
        try:
            safety_flags = []
            
            query_lower = user_query.lower()
            response_lower = response.lower()
            
            # Check for red flag symptoms in query
            for red_flag in RED_FLAG_SYMPTOMS:
                if red_flag in query_lower:
                    safety_flags.append(f"red_flag_symptom: {red_flag}")
            
            # Check for emergency language
            emergency_words = ['emergency', 'urgent', 'severe', 'immediate']
            for word in emergency_words:
                if word in query_lower:
                    safety_flags.append(f"emergency_language: {word}")
            
            # Check if response appropriately handles urgency
            if safety_flags and 'emergency' not in response_lower and 'urgent' not in response_lower:
                safety_flags.append("inadequate_urgency_response")
            
            # Check for inappropriate medical advice
            inappropriate_advice = ['diagnose', 'prescribe', 'cure', 'treatment for']
            for advice in inappropriate_advice:
                if advice in response_lower:
                    safety_flags.append(f"inappropriate_advice: {advice}")
            
            # Check for missing disclaimers on medical topics
            medical_topics = ['symptoms', 'condition', 'disease', 'treatment']
            if any(topic in response_lower for topic in medical_topics):
                if not any(disclaimer in response_lower for disclaimer in 
                          ['disclaimer', 'medical professional', 'healthcare provider']):
                    safety_flags.append("missing_medical_disclaimer")
            
            return safety_flags
            
        except Exception as e:
            logger.error(f"Failed to check safety flags: {e}")
            return ["safety_check_error"]
    
    def enhance_response_empathy(self, response: str, user_query: str) -> str:
        """
        Enhance response with additional empathy elements
        
        Args:
            response: Original response
            user_query: User's query
            
        Returns:
            Enhanced response with improved empathy
        """
        try:
            # Calculate current empathy score
            current_score = self.calculate_empathy_score(response, user_query)
            
            # If score is already high, return original
            if current_score >= 0.8:
                return response
            
            # Add empathetic introduction if missing
            if not self._has_empathetic_introduction(response):
                empathetic_intro = self._generate_empathetic_introduction(user_query)
                response = f"{empathetic_intro}\n\n{response}"
            
            # Add validation if missing
            if not self._has_validation(response):
                validation = self._generate_validation(user_query)
                response = f"{validation}\n\n{response}"
            
            # Add reassurance if appropriate
            if self._needs_reassurance(user_query) and not self._has_reassurance(response):
                reassurance = self._generate_reassurance(user_query)
                response = f"{response}\n\n{reassurance}"
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to enhance response empathy: {e}")
            return response
    
    def _has_empathetic_introduction(self, response: str) -> bool:
        """Check if response has empathetic introduction"""
        intro_indicators = ['understand', 'hear', 'see', 'know', 'appreciate']
        first_sentence = response.split('.')[0].lower()
        return any(indicator in first_sentence for indicator in intro_indicators)
    
    def _has_validation(self, response: str) -> bool:
        """Check if response contains validation"""
        validation_indicators = self.empathy_keywords['validation']
        return any(indicator in response.lower() for indicator in validation_indicators)
    
    def _has_reassurance(self, response: str) -> bool:
        """Check if response contains reassurance"""
        reassurance_indicators = self.empathy_keywords['reassurance']
        return any(indicator in response.lower() for indicator in reassurance_indicators)
    
    def _needs_reassurance(self, query: str) -> bool:
        """Check if query suggests need for reassurance"""
        worry_indicators = ['worried', 'concerned', 'scared', 'anxious', 'afraid']
        return any(indicator in query.lower() for indicator in worry_indicators)
    
    def _generate_empathetic_introduction(self, query: str) -> str:
        """Generate empathetic introduction"""
        if 'worried' in query.lower() or 'concerned' in query.lower():
            return "I understand this is concerning for you, and seeking answers about your health shows great self-awareness."
        elif 'pain' in query.lower() or 'hurt' in query.lower():
            return "I can see you're experiencing discomfort, and I want to help you understand what might be happening."
        else:
            return "Thank you for reaching out about your gut health. I'm here to provide helpful, evidence-based information."
    
    def _generate_validation(self, query: str) -> str:
        """Generate validation statement"""
        return "Your concern is completely valid, and you're not alone in experiencing this type of digestive issue."
    
    def _generate_reassurance(self, query: str) -> str:
        """Generate reassurance statement"""
        return "While digestive issues can be concerning, many people find significant relief through evidence-based approaches and appropriate medical care."
    
    def analyze_conversation_quality(self, conversation_history: List[Dict]) -> Dict[str, Any]:
        """
        Analyze overall conversation quality
        
        Args:
            conversation_history: List of conversation turns
            
        Returns:
            Quality analysis metrics
        """
        try:
            if not conversation_history:
                return {"error": "No conversation history provided"}
            
            total_turns = len(conversation_history)
            assistant_turns = [turn for turn in conversation_history if turn.get('role') == 'assistant']
            
            # Calculate average empathy score
            empathy_scores = []
            for turn in assistant_turns:
                # Find corresponding user turn
                user_turn = None
                for i, conv_turn in enumerate(conversation_history):
                    if conv_turn == turn and i > 0:
                        user_turn = conversation_history[i-1]
                        break
                
                if user_turn and user_turn.get('role') == 'user':
                    score = self.calculate_empathy_score(
                        turn.get('content', ''), 
                        user_turn.get('content', '')
                    )
                    empathy_scores.append(score)
            
            avg_empathy = sum(empathy_scores) / len(empathy_scores) if empathy_scores else 0.0
            
            # Calculate other quality metrics
            analysis = {
                "total_conversation_turns": total_turns,
                "assistant_responses": len(assistant_turns),
                "average_empathy_score": avg_empathy,
                "empathy_score_distribution": {
                    "high (>0.8)": sum(1 for score in empathy_scores if score > 0.8),
                    "medium (0.5-0.8)": sum(1 for score in empathy_scores if 0.5 <= score <= 0.8),
                    "low (<0.5)": sum(1 for score in empathy_scores if score < 0.5)
                },
                "conversation_quality_grade": self._get_quality_grade(avg_empathy),
                "recommendations": self._get_quality_recommendations(avg_empathy, empathy_scores)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze conversation quality: {e}")
            return {"error": str(e)}
    
    def _get_quality_grade(self, avg_empathy: float) -> str:
        """Get quality grade based on empathy score"""
        if avg_empathy >= 0.8:
            return "Excellent"
        elif avg_empathy >= 0.6:
            return "Good"
        elif avg_empathy >= 0.4:
            return "Fair"
        else:
            return "Needs Improvement"
    
    def _get_quality_recommendations(self, avg_empathy: float, scores: List[float]) -> List[str]:
        """Get recommendations for improving conversation quality"""
        recommendations = []
        
        if avg_empathy < 0.6:
            recommendations.append("Increase validation and acknowledgment of user concerns")
        
        if avg_empathy < 0.5:
            recommendations.append("Add more reassuring and supportive language")
        
        if len(scores) > 0 and max(scores) - min(scores) > 0.4:
            recommendations.append("Maintain more consistent empathy levels across responses")
        
        if not recommendations:
            recommendations.append("Continue maintaining high empathy standards")
        
        return recommendations

def main():
    """
    Test the empathy engine
    """
    try:
        # Initialize empathy engine
        empathy_engine = EmpathyEngine()
        
        # Test scenarios
        test_scenarios = [
            {
                "user_query": "I'm really worried about my bloating. It's been going on for weeks.",
                "response": "I understand your concern about persistent bloating. This is a common digestive issue that many people experience. There are several potential causes including dietary factors, stress, and gut microbiome imbalances. I'd recommend keeping a food diary and considering consultation with a healthcare provider for proper evaluation."
            },
            {
                "user_query": "I have severe abdominal pain and blood in my stool",
                "response": "These symptoms require immediate medical attention. Please contact emergency services or visit an emergency room right away. Blood in stool combined with severe pain can indicate serious conditions that need professional evaluation."
            },
            {
                "user_query": "What are some natural ways to improve gut health?",
                "response": "Great question! There are many evidence-based natural approaches including eating diverse fiber-rich foods, managing stress, getting adequate sleep, and considering probiotics. A balanced diet with plenty of vegetables, fruits, and whole grains supports a healthy gut microbiome."
            }
        ]
        
        logger.info("🧪 Testing empathy engine...")
        
        for i, scenario in enumerate(test_scenarios, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Test Scenario {i}")
            logger.info(f"{'='*60}")
            
            user_query = scenario["user_query"]
            response = scenario["response"]
            
            # Calculate empathy score
            empathy_score = empathy_engine.calculate_empathy_score(response, user_query)
            
            # Check safety flags
            safety_flags = empathy_engine.check_safety_flags(user_query, response)
            
            # Enhance response
            enhanced_response = empathy_engine.enhance_response_empathy(response, user_query)
            
            # Display results
            logger.info(f"📝 User Query: {user_query}")
            logger.info(f"💗 Empathy Score: {empathy_score:.3f}")
            logger.info(f"⚠️ Safety Flags: {safety_flags}")
            logger.info(f"📈 Enhanced Response:\n{enhanced_response[:200]}...")
        
        logger.info("\n✅ Empathy engine testing completed!")
        
    except Exception as e:
        logger.error(f"❌ Testing failed: {e}")
        raise

if __name__ == "__main__":
    main()
