"""
Critical test questions evaluation for gut health coach
Tests the 10 mandatory questions from the project requirements
"""
import json
import time
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

# Import your RAG components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.rag_pipeline import GutHealthRAGPipeline
from src.tone_engineering import EmpathyEngine
from config.settings import CRITICAL_TEST_QUESTIONS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CriticalQuestionsEvaluator:
    """
    Evaluator for the 10 critical test questions
    """
    
    def __init__(self):
        """Initialize the evaluator"""
        self.rag_pipeline = GutHealthRAGPipeline()
        self.empathy_engine = EmpathyEngine()
        self.results = []
        
        # Expected answer categories for evaluation
        self.answer_categories = {
            "I've been bloated for three days — what should I do?": {
                "category": "symptom_advice",
                "expected_elements": ["dietary_factors", "medical_consultation", "timeline_concern"],
                "urgency_level": "moderate"
            },
            "How does gut health affect sleep?": {
                "category": "gut_brain_connection",
                "expected_elements": ["microbiome_influence", "sleep_quality", "scientific_connection"],
                "urgency_level": "low"
            },
            "What are the best probiotics for lactose intolerance?": {
                "category": "treatment_guidance",
                "expected_elements": ["specific_strains", "lactose_digestion", "evidence_based"],
                "urgency_level": "low"
            },
            "What does mucus in stool indicate?": {
                "category": "symptom_interpretation",
                "expected_elements": ["possible_causes", "when_to_worry", "professional_consultation"],
                "urgency_level": "moderate_high"
            },
            "I feel nauseous after eating fermented foods. Is that normal?": {
                "category": "food_reaction",
                "expected_elements": ["fermentation_effects", "individual_variation", "adaptation_advice"],
                "urgency_level": "low"
            },
            "Should I fast if my gut is inflamed?": {
                "category": "dietary_intervention",
                "expected_elements": ["inflammation_response", "fasting_caution", "professional_guidance"],
                "urgency_level": "high"
            },
            "Can antibiotics damage gut flora permanently?": {
                "category": "medication_effects",
                "expected_elements": ["microbiome_disruption", "recovery_potential", "restoration_methods"],
                "urgency_level": "moderate"
            },
            "How do I know if I have SIBO?": {
                "category": "condition_diagnosis",
                "expected_elements": ["symptoms_description", "testing_options", "professional_diagnosis"],
                "urgency_level": "moderate_high"
            },
            "What are signs that my gut is healing?": {
                "category": "recovery_indicators",
                "expected_elements": ["positive_symptoms", "timeline_expectations", "monitoring_progress"],
                "urgency_level": "low"
            },
            "Why do I feel brain fog after eating sugar?": {
                "category": "gut_brain_symptoms",
                "expected_elements": ["blood_sugar_impact", "inflammation_connection", "dietary_adjustment"],
                "urgency_level": "low"
            }
        }
    
    def evaluate_single_question(self, question: str) -> Dict[str, Any]:
        """
        Evaluate a single critical question
        
        Args:
            question: The test question
            
        Returns:
            Evaluation results dictionary
        """
        try:
            start_time = time.time()
            
            # Get RAG response
            logger.info(f"Evaluating: {question[:50]}...")
            response = self.rag_pipeline.process_query(question)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Get expected criteria
            expected = self.answer_categories.get(question, {})
            
            # Evaluate medical accuracy
            medical_accuracy = self._evaluate_medical_accuracy(
                question, response.assistant_response, expected
            )
            
            # Evaluate empathy score
            empathy_score = response.empathy_score
            
            # Evaluate completeness
            completeness_score = self._evaluate_completeness(
                response.assistant_response, expected.get("expected_elements", [])
            )
            
            # Evaluate safety handling
            safety_score = self._evaluate_safety_handling(
                question, response, expected.get("urgency_level", "low")
            )
            
            # Calculate overall score
            overall_score = (
                medical_accuracy * 0.3 +
                empathy_score * 0.25 +
                completeness_score * 0.25 +
                safety_score * 0.2
            )
            
            evaluation = {
                "question": question,
                "response": response.assistant_response,
                "response_time": response_time,
                "medical_accuracy": medical_accuracy,
                "empathy_score": empathy_score,
                "completeness_score": completeness_score,
                "safety_score": safety_score,
                "overall_score": overall_score,
                "sources_used": response.sources_used,
                "safety_flags": response.safety_flags,
                "tokens_used": response.tokens_used,
                "expected_category": expected.get("category", "unknown"),
                "urgency_level": expected.get("urgency_level", "unknown"),
                "evaluation_timestamp": datetime.now().isoformat()
            }
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Evaluation failed for question: {question[:50]}... Error: {e}")
            return {
                "question": question,
                "error": str(e),
                "overall_score": 0.0,
                "evaluation_timestamp": datetime.now().isoformat()
            }
    
    def _evaluate_medical_accuracy(self, question: str, response: str, expected: Dict) -> float:
        """Evaluate medical accuracy of response"""
        try:
            accuracy_score = 0.0
            
            # Check for evidence-based information
            evidence_indicators = [
                "studies", "research", "evidence", "clinical", "medical literature"
            ]
            if any(indicator in response.lower() for indicator in evidence_indicators):
                accuracy_score += 0.2
            
            # Check for appropriate medical language
            if not any(overly_confident in response.lower() for overly_confident in 
                      ["definitely", "certainly", "always", "never", "cure", "guaranteed"]):
                accuracy_score += 0.2
            
            # Check for condition-specific accuracy
            category = expected.get("category", "")
            if category == "symptom_interpretation" and "healthcare provider" in response.lower():
                accuracy_score += 0.3
            elif category == "treatment_guidance" and "individual" in response.lower():
                accuracy_score += 0.3
            elif category == "condition_diagnosis" and "professional diagnosis" in response.lower():
                accuracy_score += 0.3
            else:
                accuracy_score += 0.2  # Baseline for reasonable response
            
            # Check for appropriate disclaimers
            if any(disclaimer in response.lower() for disclaimer in 
                  ["medical advice", "healthcare professional", "consult", "disclaimer"]):
                accuracy_score += 0.3
            
            return min(accuracy_score, 1.0)
            
        except Exception as e:
            logger.error(f"Failed to evaluate medical accuracy: {e}")
            return 0.5
    
    def _evaluate_completeness(self, response: str, expected_elements: List[str]) -> float:
        """Evaluate completeness of response"""
        try:
            if not expected_elements:
                return 0.8  # No specific expectations
            
            elements_covered = 0
            response_lower = response.lower()
            
            element_keywords = {
                "dietary_factors": ["food", "diet", "eating", "nutrition"],
                "medical_consultation": ["doctor", "healthcare", "medical", "consult"],
                "timeline_concern": ["days", "weeks", "persistent", "ongoing"],
                "microbiome_influence": ["bacteria", "microbiome", "gut flora"],
                "sleep_quality": ["sleep", "rest", "insomnia"],
                "scientific_connection": ["research", "studies", "connection"],
                "specific_strains": ["lactobacillus", "bifidobacterium", "strains"],
                "lactose_digestion": ["lactose", "dairy", "enzyme"],
                "evidence_based": ["evidence", "research", "studies"],
                "possible_causes": ["causes", "reasons", "factors"],
                "when_to_worry": ["concerning", "urgent", "serious"],
                "professional_consultation": ["doctor", "healthcare", "professional"],
                "fermentation_effects": ["fermented", "fermentation", "bacteria"],
                "individual_variation": ["individual", "person", "varies"],
                "adaptation_advice": ["gradually", "slowly", "adapt"],
                "inflammation_response": ["inflammation", "inflammatory", "healing"],
                "fasting_caution": ["careful", "caution", "consult"],
                "professional_guidance": ["guidance", "supervision", "medical"],
                "microbiome_disruption": ["disruption", "damage", "imbalance"],
                "recovery_potential": ["recover", "restore", "rebuild"],
                "restoration_methods": ["probiotics", "prebiotics", "restore"],
                "symptoms_description": ["symptoms", "signs", "indicators"],
                "testing_options": ["test", "testing", "diagnosis"],
                "professional_diagnosis": ["diagnosis", "professional", "testing"],
                "positive_symptoms": ["improvement", "better", "healing"],
                "timeline_expectations": ["time", "weeks", "months"],
                "monitoring_progress": ["track", "monitor", "progress"],
                "blood_sugar_impact": ["blood sugar", "glucose", "insulin"],
                "inflammation_connection": ["inflammation", "inflammatory", "response"],
                "dietary_adjustment": ["diet", "avoid", "reduce"]
            }
            
            for element in expected_elements:
                if element in element_keywords:
                    keywords = element_keywords[element]
                    if any(keyword in response_lower for keyword in keywords):
                        elements_covered += 1
            
            return min(elements_covered / len(expected_elements), 1.0)
            
        except Exception as e:
            logger.error(f"Failed to evaluate completeness: {e}")
            return 0.5
    
    def _evaluate_safety_handling(self, question: str, response, urgency_level: str) -> float:
        """Evaluate safety and urgency handling"""
        try:
            safety_score = 0.0
            response_text = response.assistant_response.lower()
            
            # Check urgency handling
            if urgency_level == "high":
                if any(urgent in response_text for urgent in 
                      ["urgent", "immediate", "emergency", "right away"]):
                    safety_score += 0.4
                if "healthcare provider" in response_text or "doctor" in response_text:
                    safety_score += 0.4
            elif urgency_level == "moderate_high":
                if "healthcare" in response_text or "medical" in response_text:
                    safety_score += 0.3
                if "concerning" in response_text or "evaluation" in response_text:
                    safety_score += 0.3
            elif urgency_level == "moderate":
                if "consider" in response_text and "healthcare" in response_text:
                    safety_score += 0.4
            else:  # low urgency
                safety_score += 0.4  # Baseline for appropriate tone
            
            # Check for safety flags handling
            if response.safety_flags:
                if any("red_flag" in flag for flag in response.safety_flags):
                    if "emergency" in response_text or "urgent" in response_text:
                        safety_score += 0.4
                    else:
                        safety_score -= 0.2  # Penalty for missing urgency
            
            # Check for appropriate disclaimers
            if any(disclaimer in response_text for disclaimer in 
                  ["not medical advice", "healthcare professional", "consult"]):
                safety_score += 0.2
            
            return min(safety_score, 1.0)
            
        except Exception as e:
            logger.error(f"Failed to evaluate safety handling: {e}")
            return 0.5
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """
        Run evaluation on all critical questions
        
        Returns:
            Complete evaluation results
        """
        try:
            logger.info("🧪 Starting critical questions evaluation...")
            
            # Evaluate each question
            for question in CRITICAL_TEST_QUESTIONS:
                evaluation = self.evaluate_single_question(question)
                self.results.append(evaluation)
                
                logger.info(f"✅ Evaluated: {question[:40]}... Score: {evaluation.get('overall_score', 0.0):.3f}")
            
            # Calculate aggregate metrics
            aggregate_metrics = self._calculate_aggregate_metrics()
            
            # Generate detailed report
            report = self._generate_evaluation_report(aggregate_metrics)
            
            # Save results
            self._save_evaluation_results(report)
            
            logger.info("🎯 Critical questions evaluation completed!")
            return report
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            raise
    
    def _calculate_aggregate_metrics(self) -> Dict[str, Any]:
        """Calculate aggregate performance metrics"""
        try:
            valid_results = [r for r in self.results if 'error' not in r]
            
            if not valid_results:
                return {"error": "No valid results to analyze"}
            
            # Calculate averages
            avg_medical_accuracy = sum(r['medical_accuracy'] for r in valid_results) / len(valid_results)
            avg_empathy_score = sum(r['empathy_score'] for r in valid_results) / len(valid_results)
            avg_completeness = sum(r['completeness_score'] for r in valid_results) / len(valid_results)
            avg_safety_score = sum(r['safety_score'] for r in valid_results) / len(valid_results)
            avg_overall_score = sum(r['overall_score'] for r in valid_results) / len(valid_results)
            avg_response_time = sum(r['response_time'] for r in valid_results) / len(valid_results)
            
            # Performance categorization
            performance_grade = self._get_performance_grade(avg_overall_score)
            
            # Category-specific analysis
            category_performance = {}
            for result in valid_results:
                category = result.get('expected_category', 'unknown')
                if category not in category_performance:
                    category_performance[category] = []
                category_performance[category].append(result['overall_score'])
            
            category_averages = {
                cat: sum(scores) / len(scores) 
                for cat, scores in category_performance.items()
            }
            
            return {
                "total_questions": len(CRITICAL_TEST_QUESTIONS),
                "successful_evaluations": len(valid_results),
                "failed_evaluations": len(self.results) - len(valid_results),
                "average_medical_accuracy": avg_medical_accuracy,
                "average_empathy_score": avg_empathy_score,
                "average_completeness": avg_completeness,
                "average_safety_score": avg_safety_score,
                "average_overall_score": avg_overall_score,
                "average_response_time": avg_response_time,
                "performance_grade": performance_grade,
                "category_performance": category_averages,
                "recommendations": self._generate_recommendations(avg_overall_score, category_averages)
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate aggregate metrics: {e}")
            return {"error": str(e)}
    
    def _get_performance_grade(self, score: float) -> str:
        """Get performance grade based on overall score"""
        if score >= 0.9:
            return "A+ (Excellent)"
        elif score >= 0.8:
            return "A (Very Good)"
        elif score >= 0.7:
            return "B+ (Good)"
        elif score >= 0.6:
            return "B (Satisfactory)"
        elif score >= 0.5:
            return "C (Needs Improvement)"
        else:
            return "D (Poor - Requires Major Improvements)"
    
    def _generate_recommendations(self, overall_score: float, category_performance: Dict) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if overall_score < 0.7:
            recommendations.append("Overall performance needs improvement - focus on medical accuracy and completeness")
        
        # Category-specific recommendations
        for category, score in category_performance.items():
            if score < 0.6:
                if category == "symptom_interpretation":
                    recommendations.append("Improve symptom interpretation responses with clearer medical guidance")
                elif category == "treatment_guidance":
                    recommendations.append("Enhance treatment guidance with more evidence-based recommendations")
                elif category == "condition_diagnosis":
                    recommendations.append("Strengthen condition diagnosis responses with professional consultation emphasis")
        
        if not recommendations:
            recommendations.append("Excellent performance! Continue maintaining high standards")
        
        return recommendations
    
    def _generate_evaluation_report(self, metrics: Dict) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        return {
            "evaluation_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_questions_evaluated": len(self.results),
                "aggregate_metrics": metrics
            },
            "detailed_results": self.results,
            "performance_analysis": {
                "strengths": self._identify_strengths(metrics),
                "areas_for_improvement": self._identify_weaknesses(metrics),
                "interview_readiness": self._assess_interview_readiness(metrics)
            }
        }
    
    def _identify_strengths(self, metrics: Dict) -> List[str]:
        """Identify system strengths"""
        strengths = []
        
        if metrics.get("average_empathy_score", 0) > 0.7:
            strengths.append("Strong empathetic communication")
        
        if metrics.get("average_safety_score", 0) > 0.8:
            strengths.append("Excellent safety and urgency handling")
        
        if metrics.get("average_medical_accuracy", 0) > 0.7:
            strengths.append("Good medical accuracy and evidence-based responses")
        
        if metrics.get("average_response_time", 300) < 60:
            strengths.append("Fast response times")
        
        return strengths or ["System shows baseline functionality"]
    
    def _identify_weaknesses(self, metrics: Dict) -> List[str]:
        """Identify areas for improvement"""
        weaknesses = []
        
        if metrics.get("average_empathy_score", 0) < 0.6:
            weaknesses.append("Empathy scoring needs improvement")
        
        if metrics.get("average_completeness", 0) < 0.6:
            weaknesses.append("Response completeness needs enhancement")
        
        if metrics.get("average_medical_accuracy", 0) < 0.6:
            weaknesses.append("Medical accuracy requires improvement")
        
        if metrics.get("average_response_time", 0) > 120:
            weaknesses.append("Response times are too slow")
        
        return weaknesses
    
    def _assess_interview_readiness(self, metrics: Dict) -> Dict[str, Any]:
        """Assess readiness for interview demonstration"""
        overall_score = metrics.get("average_overall_score", 0)
        
        if overall_score >= 0.8:
            readiness = "READY"
            confidence = "HIGH"
            message = "System performs excellently and is interview-ready"
        elif overall_score >= 0.7:
            readiness = "MOSTLY_READY"
            confidence = "MEDIUM-HIGH"
            message = "System performs well with minor areas for improvement"
        elif overall_score >= 0.6:
            readiness = "NEEDS_WORK"
            confidence = "MEDIUM"
            message = "System needs optimization before interview"
        else:
            readiness = "NOT_READY"
            confidence = "LOW"
            message = "Significant improvements needed before interview"
        
        return {
            "status": readiness,
            "confidence_level": confidence,
            "message": message,
            "overall_score": overall_score
        }
    
    def _save_evaluation_results(self, report: Dict) -> None:
        """Save evaluation results to files"""
        try:
            # Create results directory
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save complete report as JSON
            report_file = results_dir / f"critical_questions_evaluation_{timestamp}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Save summary as CSV for easy analysis
            summary_data = []
            for result in self.results:
                if 'error' not in result:
                    summary_data.append({
                        'question': result['question'][:50] + '...',
                        'overall_score': result['overall_score'],
                        'medical_accuracy': result['medical_accuracy'],
                        'empathy_score': result['empathy_score'],
                        'completeness_score': result['completeness_score'],
                        'safety_score': result['safety_score'],
                        'response_time': result['response_time'],
                        'category': result['expected_category']
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_file = results_dir / f"evaluation_summary_{timestamp}.csv"
                summary_df.to_csv(summary_file, index=False)
            
            logger.info(f"✅ Results saved to {results_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")

def main():
    """
    Run critical questions evaluation
    """
    try:
        evaluator = CriticalQuestionsEvaluator()
        results = evaluator.run_full_evaluation()
        
        # Display key metrics
        metrics = results["evaluation_summary"]["aggregate_metrics"]
        print("\n" + "="*60)
        print("🎯 CRITICAL QUESTIONS EVALUATION RESULTS")
        print("="*60)
        print(f"Overall Score: {metrics['average_overall_score']:.3f}")
        print(f"Performance Grade: {metrics['performance_grade']}")
        print(f"Medical Accuracy: {metrics['average_medical_accuracy']:.3f}")
        print(f"Empathy Score: {metrics['average_empathy_score']:.3f}")
        print(f"Average Response Time: {metrics['average_response_time']:.1f}s")
        
        readiness = results["performance_analysis"]["interview_readiness"]
        print(f"\nInterview Readiness: {readiness['status']}")
        print(f"Confidence Level: {readiness['confidence_level']}")
        print(f"Assessment: {readiness['message']}")
        
        print("\n📊 Results saved to 'results/' directory")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
