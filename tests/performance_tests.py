"""
Performance and stress testing for gut health coach
"""
import time
import asyncio
import threading
import statistics
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import logging
import psutil
import concurrent.futures

# Import RAG components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.rag_pipeline import GutHealthRAGPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceTestSuite:
    """
    Comprehensive performance testing for RAG pipeline
    """
    
    def __init__(self):
        """Initialize performance test suite"""
        self.rag_pipeline = GutHealthRAGPipeline()
        self.test_queries = [
            "What causes bloating?",
            "How to improve gut health?",
            "What are symptoms of IBS?",
            "Foods to avoid with SIBO",
            "How does stress affect digestion?",
            "What are probiotics benefits?",
            "Signs of leaky gut syndrome",
            "How to heal gut naturally?",
            "What is gut microbiome?",
            "Can diet cure IBS?"
        ]
        self.performance_results = {
            "response_times": [],
            "memory_usage": [],
            "cpu_usage": [],
            "throughput_metrics": {},
            "error_rates": {},
            "concurrent_performance": {}
        }
    
    def test_response_time_consistency(self, iterations: int = 20) -> Dict[str, Any]:
        """
        Test response time consistency across multiple queries
        
        Args:
            iterations: Number of test iterations
            
        Returns:
            Response time metrics
        """
        try:
            logger.info(f"🚀 Testing response time consistency ({iterations} iterations)...")
            
            response_times = []
            errors = 0
            
            for i in range(iterations):
                query = self.test_queries[i % len(self.test_queries)]
                
                try:
                    start_time = time.time()
                    response = self.rag_pipeline.process_query(query)
                    end_time = time.time()
                    
                    response_time = end_time - start_time
                    response_times.append(response_time)
                    
                    logger.info(f"Iteration {i+1}/{iterations}: {response_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Query failed at iteration {i+1}: {e}")
                    errors += 1
            
            # Calculate statistics
            if response_times:
                metrics = {
                    "total_iterations": iterations,
                    "successful_queries": len(response_times),
                    "failed_queries": errors,
                    "success_rate": len(response_times) / iterations,
                    "mean_response_time": statistics.mean(response_times),
                    "median_response_time": statistics.median(response_times),
                    "min_response_time": min(response_times),
                    "max_response_time": max(response_times),
                    "std_dev_response_time": statistics.stdev(response_times) if len(response_times) > 1 else 0,
                    "response_times": response_times
                }
                
                self.performance_results["response_times"] = metrics
                return metrics
            else:
                return {"error": "No successful queries completed"}
                
        except Exception as e:
            logger.error(f"Response time test failed: {e}")
            return {"error": str(e)}
    
    def test_memory_usage(self, monitoring_duration: int = 300) -> Dict[str, Any]:
        """
        Monitor memory usage during operation
        
        Args:
            monitoring_duration: Duration to monitor in seconds
            
        Returns:
            Memory usage metrics
        """
        try:
            logger.info(f"🧠 Monitoring memory usage for {monitoring_duration}s...")
            
            memory_readings = []
            start_time = time.time()
            
            # Start background query processing
            def background_queries():
                while time.time() - start_time < monitoring_duration:
                    try:
                        query = self.test_queries[int(time.time()) % len(self.test_queries)]
                        self.rag_pipeline.process_query(query)
                        time.sleep(5)  # Space out queries
                    except Exception as e:
                        logger.error(f"Background query failed: {e}")
            
            # Start background thread
            query_thread = threading.Thread(target=background_queries)
            query_thread.daemon = True
            query_thread.start()
            
            # Monitor memory usage
            while time.time() - start_time < monitoring_duration:
                memory_info = psutil.virtual_memory()
                process = psutil.Process()
                process_memory = process.memory_info()
                
                reading = {
                    "timestamp": time.time() - start_time,
                    "total_memory_gb": memory_info.total / (1024**3),
                    "available_memory_gb": memory_info.available / (1024**3),
                    "memory_percent": memory_info.percent,
                    "process_memory_mb": process_memory.rss / (1024**2),
                    "process_memory_percent": process.memory_percent()
                }
                
                memory_readings.append(reading)
                time.sleep(10)  # Sample every 10 seconds
            
            # Calculate memory metrics
            if memory_readings:
                process_memory_values = [r["process_memory_mb"] for r in memory_readings]
                
                metrics = {
                    "monitoring_duration": monitoring_duration,
                    "total_readings": len(memory_readings),
                    "average_process_memory_mb": statistics.mean(process_memory_values),
                    "peak_process_memory_mb": max(process_memory_values),
                    "min_process_memory_mb": min(process_memory_values),
                    "memory_growth_mb": max(process_memory_values) - min(process_memory_values),
                    "readings": memory_readings
                }
                
                self.performance_results["memory_usage"] = metrics
                return metrics
            else:
                return {"error": "No memory readings collected"}
                
        except Exception as e:
            logger.error(f"Memory usage test failed: {e}")
            return {"error": str(e)}
    
    def test_concurrent_load(self, concurrent_users: int = 5, queries_per_user: int = 10) -> Dict[str, Any]:
        """
        Test performance under concurrent load
        
        Args:
            concurrent_users: Number of concurrent users to simulate
            queries_per_user: Number of queries per user
            
        Returns:
            Concurrent load metrics
        """
        try:
            logger.info(f"🔄 Testing concurrent load ({concurrent_users} users, {queries_per_user} queries each)...")
            
            def user_session(user_id: int) -> Dict[str, Any]:
                """Simulate a user session"""
                session_results = {
                    "user_id": user_id,
                    "queries_completed": 0,
                    "total_time": 0,
                    "response_times": [],
                    "errors": 0
                }
                
                session_start = time.time()
                
                for i in range(queries_per_user):
                    try:
                        query = self.test_queries[i % len(self.test_queries)]
                        
                        query_start = time.time()
                        response = self.rag_pipeline.process_query(query)
                        query_end = time.time()
                        
                        response_time = query_end - query_start
                        session_results["response_times"].append(response_time)
                        session_results["queries_completed"] += 1
                        
                    except Exception as e:
                        logger.error(f"User {user_id} query {i+1} failed: {e}")
                        session_results["errors"] += 1
                
                session_results["total_time"] = time.time() - session_start
                return session_results
            
            # Run concurrent user sessions
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
                futures = [executor.submit(user_session, user_id) for user_id in range(concurrent_users)]
                session_results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            total_time = time.time() - start_time
            
            # Aggregate results
            total_queries = sum(session["queries_completed"] for session in session_results)
            total_errors = sum(session["errors"] for session in session_results)
            all_response_times = []
            
            for session in session_results:
                all_response_times.extend(session["response_times"])
            
            metrics = {
                "concurrent_users": concurrent_users,
                "queries_per_user": queries_per_user,
                "total_test_time": total_time,
                "total_queries_attempted": concurrent_users * queries_per_user,
                "total_queries_completed": total_queries,
                "total_errors": total_errors,
                "success_rate": total_queries / (concurrent_users * queries_per_user),
                "throughput_qps": total_queries / total_time,
                "average_response_time": statistics.mean(all_response_times) if all_response_times else 0,
                "median_response_time": statistics.median(all_response_times) if all_response_times else 0,
                "session_details": session_results
            }
            
            self.performance_results["concurrent_performance"] = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Concurrent load test failed: {e}")
            return {"error": str(e)}
    
    def test_error_recovery(self) -> Dict[str, Any]:
        """
        Test error recovery and system resilience
        
        Returns:
            Error recovery metrics
        """
        try:
            logger.info("🛡️ Testing error recovery...")
            
            # Test scenarios that might cause errors
            error_test_cases = [
                "",  # Empty query
                "a" * 10000,  # Very long query
                "!@#$%^&*()",  # Special characters only
                "What is the meaning of life, the universe, and everything?" * 100,  # Extremely long
                None,  # None input (will be converted to string)
            ]
            
            error_results = []
            
            for i, test_case in enumerate(error_test_cases):
                try:
                    start_time = time.time()
                    
                    # Convert None to string for processing
                    query = str(test_case) if test_case is not None else ""
                    
                    response = self.rag_pipeline.process_query(query)
                    response_time = time.time() - start_time
                    
                    error_results.append({
                        "test_case": f"Case {i+1}",
                        "query_length": len(query),
                        "status": "success",
                        "response_time": response_time,
                        "response_length": len(response.assistant_response) if hasattr(response, 'assistant_response') else 0
                    })
                    
                except Exception as e:
                    error_results.append({
                        "test_case": f"Case {i+1}",
                        "query_length": len(str(test_case)) if test_case is not None else 0,
                        "status": "error",
                        "error": str(e),
                        "response_time": 0
                    })
            
            # Calculate error recovery metrics
            successful_recoveries = sum(1 for result in error_results if result["status"] == "success")
            
            metrics = {
                "total_error_cases": len(error_test_cases),
                "successful_recoveries": successful_recoveries,
                "recovery_rate": successful_recoveries / len(error_test_cases),
                "error_details": error_results
            }
            
            self.performance_results["error_rates"] = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error recovery test failed: {e}")
            return {"error": str(e)}
    
    def run_full_performance_suite(self) -> Dict[str, Any]:
        """
        Run complete performance test suite
        
        Returns:
            Complete performance report
        """
        try:
            logger.info("🧪 Starting full performance test suite...")
            
            # Test 1: Response Time Consistency
            logger.info("\n1️⃣ Testing response time consistency...")
            response_time_results = self.test_response_time_consistency(iterations=10)
            
            # Test 2: Memory Usage Monitoring  
            logger.info("\n2️⃣ Monitoring memory usage...")
            memory_results = self.test_memory_usage(monitoring_duration=60)  # 1 minute
            
            # Test 3: Concurrent Load Testing
            logger.info("\n3️⃣ Testing concurrent load...")
            concurrent_results = self.test_concurrent_load(concurrent_users=3, queries_per_user=5)
            
            # Test 4: Error Recovery
            logger.info("\n4️⃣ Testing error recovery...")
            error_recovery_results = self.test_error_recovery()
            
            # Generate comprehensive report
            performance_report = {
                "test_timestamp": datetime.now().isoformat(),
                "response_time_consistency": response_time_results,
                "memory_usage_monitoring": memory_results,
                "concurrent_load_testing": concurrent_results,
                "error_recovery_testing": error_recovery_results,
                "overall_assessment": self._assess_overall_performance()
            }
            
            # Save performance report
            self._save_performance_report(performance_report)
            
            logger.info("✅ Full performance test suite completed!")
            return performance_report
            
        except Exception as e:
            logger.error(f"❌ Performance test suite failed: {e}")
            return {"error": str(e)}
    
    def _assess_overall_performance(self) -> Dict[str, Any]:
        """Assess overall system performance"""
        try:
            assessment = {
                "response_time_grade": "Unknown",
                "memory_efficiency_grade": "Unknown", 
                "concurrency_grade": "Unknown",
                "reliability_grade": "Unknown",
                "overall_grade": "Unknown",
                "recommendations": []
            }
            
            # Response time assessment
            rt_metrics = self.performance_results.get("response_times", {})
            if rt_metrics and "mean_response_time" in rt_metrics:
                mean_rt = rt_metrics["mean_response_time"]
                if mean_rt < 30:
                    assessment["response_time_grade"] = "A (Excellent)"
                elif mean_rt < 60:
                    assessment["response_time_grade"] = "B (Good)"
                elif mean_rt < 120:
                    assessment["response_time_grade"] = "C (Acceptable)"
                else:
                    assessment["response_time_grade"] = "D (Needs Improvement)"
                    assessment["recommendations"].append("Optimize response time - consider model quantization or caching")
            
            # Concurrency assessment
            concurrent_metrics = self.performance_results.get("concurrent_performance", {})
            if concurrent_metrics and "success_rate" in concurrent_metrics:
                success_rate = concurrent_metrics["success_rate"]
                if success_rate > 0.95:
                    assessment["concurrency_grade"] = "A (Excellent)"
                elif success_rate > 0.90:
                    assessment["concurrency_grade"] = "B (Good)"
                elif success_rate > 0.80:
                    assessment["concurrency_grade"] = "C (Acceptable)"
                else:
                    assessment["concurrency_grade"] = "D (Needs Improvement)"
                    assessment["recommendations"].append("Improve concurrent handling - consider connection pooling")
            
            # Memory efficiency assessment  
            memory_metrics = self.performance_results.get("memory_usage", {})
            if memory_metrics and "memory_growth_mb" in memory_metrics:
                growth = memory_metrics["memory_growth_mb"]
                if growth < 100:
                    assessment["memory_efficiency_grade"] = "A (Excellent)"
                elif growth < 500:
                    assessment["memory_efficiency_grade"] = "B (Good)"
                elif growth < 1000:
                    assessment["memory_efficiency_grade"] = "C (Acceptable)"
                else:
                    assessment["memory_efficiency_grade"] = "D (Memory Leak Detected)"
                    assessment["recommendations"].append("Investigate potential memory leaks")
            
            # Error recovery assessment
            error_metrics = self.performance_results.get("error_rates", {})
            if error_metrics and "recovery_rate" in error_metrics:
                recovery_rate = error_metrics["recovery_rate"]
                if recovery_rate > 0.80:
                    assessment["reliability_grade"] = "A (Excellent)"
                elif recovery_rate > 0.60:
                    assessment["reliability_grade"] = "B (Good)"
                elif recovery_rate > 0.40:
                    assessment["reliability_grade"] = "C (Acceptable)"
                else:
                    assessment["reliability_grade"] = "D (Needs Improvement)"
                    assessment["recommendations"].append("Improve error handling and recovery mechanisms")
            
            # Overall grade calculation
            grades = [g for g in [
                assessment["response_time_grade"],
                assessment["memory_efficiency_grade"],
                assessment["concurrency_grade"],
                assessment["reliability_grade"]
            ] if g != "Unknown"]
            
            if grades:
                # Simple average of letter grades
                grade_values = {"A": 4, "B": 3, "C": 2, "D": 1}
                avg_grade = sum(grade_values.get(g[0], 2) for g in grades) / len(grades)
                
                if avg_grade >= 3.5:
                    assessment["overall_grade"] = "A (Interview Ready)"
                elif avg_grade >= 2.5:
                    assessment["overall_grade"] = "B (Minor Optimizations Needed)"  
                elif avg_grade >= 1.5:
                    assessment["overall_grade"] = "C (Major Improvements Required)"
                else:
                    assessment["overall_grade"] = "D (Not Ready for Production)"
            
            return assessment
            
        except Exception as e:
            logger.error(f"Failed to assess performance: {e}")
            return {"error": str(e)}
    
    def _save_performance_report(self, report: Dict) -> None:
        """Save performance report to files"""
        try:
            # Create results directory
            results_dir = Path("results") 
            results_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save complete report
            report_file = results_dir / f"performance_report_{timestamp}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"✅ Performance report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save performance report: {e}")

def main():
    """
    Run performance test suite
    """
    try:
        test_suite = PerformanceTestSuite()
        results = test_suite.run_full_performance_suite()
        
        # Display summary
        print("\n" + "="*60)
        print("⚡ PERFORMANCE TEST RESULTS")
        print("="*60)
        
        if "overall_assessment" in results:
            assessment = results["overall_assessment"]
            print(f"Overall Grade: {assessment.get('overall_grade', 'Unknown')}")
            print(f"Response Time: {assessment.get('response_time_grade', 'Unknown')}")
            print(f"Memory Efficiency: {assessment.get('memory_efficiency_grade', 'Unknown')}")
            print(f"Concurrency: {assessment.get('concurrency_grade', 'Unknown')}")
            print(f"Reliability: {assessment.get('reliability_grade', 'Unknown')}")
            
            if assessment.get('recommendations'):
                print("\nRecommendations:")
                for rec in assessment['recommendations']:
                    print(f"• {rec}")
        
        print("\n📊 Detailed results saved to 'results/' directory")
        
    except Exception as e:
        print(f"❌ Performance testing failed: {e}")
        raise

if __name__ == "__main__":
    main()
