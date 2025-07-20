"""
Main Gradio application for gut health coach
"""
import gradio as gr
import time
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import local components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import *
from src.rag_pipeline import GutHealthRAGPipeline
from ui.chat_interface import GutHealthChatInterface

# Load custom CSS
css_file = Path(__file__).parent / "styling.css"
custom_css = css_file.read_text() if css_file.exists() else ""

class GutHealthCoachApp:
    """
    Main Gradio application for gut health coaching
    """
    
    def __init__(self):
        """Initialize the application"""
        self.rag_pipeline = None
        self.chat_interface = None
        self.app_metrics = {
            "total_sessions": 0,
            "total_queries": 0,
            "average_response_time": 0.0,
            "user_satisfaction": 0.0
        }
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize RAG pipeline and chat interface"""
        try:
            print("🚀 Initializing Gut Health Coach...")
            
            # Initialize RAG pipeline
            self.rag_pipeline = GutHealthRAGPipeline()
            
            # Initialize chat interface
            self.chat_interface = GutHealthChatInterface(self.rag_pipeline)
            
            print("✅ Gut Health Coach initialized successfully!")
            
        except Exception as e:
            print(f"❌ Failed to initialize components: {e}")
            raise
    
    def create_interface(self) -> gr.Blocks:
        """
        Create the main Gradio interface
        
        Returns:
            Gradio Blocks interface
        """
        try:
            with gr.Blocks(
                title="Gut Health Coach - Your Empathetic AI Health Assistant",
                theme=gr.themes.Soft(
                    primary_hue="green",
                    secondary_hue="blue",
                    neutral_hue="slate"
                ),
                css=custom_css
            ) as interface:
                
                # Header
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div style="text-align: center; padding: 20px;">
                            <h1 style="color: #2E8B57; font-size: 2.5em; margin-bottom: 10px;">
                                🌱 Gut Health Coach
                            </h1>
                            <p style="color: #666; font-size: 1.2em; margin-bottom: 20px;">
                                Your empathetic AI health assistant for gut health guidance
                            </p>
                            <div style="background: linear-gradient(135deg, #E8F5E8 0%, #F0F8FF 100%); 
                                        padding: 15px; border-radius: 10px; margin: 20px 0;">
                                <p style="color: #2E8B57; font-weight: bold; margin: 0;">
                                    ✨ Powered by BioMistral-7B • 🔬 Medical-grade accuracy • 💝 August AI-style empathy
                                </p>
                            </div>
                        </div>
                        """)
                
                # Main chat interface
                with gr.Row():
                    with gr.Column(scale=3):
                        # Chat interface
                        chatbot = gr.Chatbot(
                            label="Gut Health Coach",
                            height=500,
                            show_label=False,
                            container=True,
                            type="messages"
                        )
                        
                        # Message input
                        with gr.Row():
                            msg_input = gr.Textbox(
                                placeholder="Ask me about your gut health concerns...",
                                scale=4,
                                container=False,
                                show_label=False
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1)
                        
                        # Quick action buttons
                        with gr.Row():
                            quick_actions = [
                                "What causes bloating?",
                                "Foods for gut health",
                                "Signs of healthy gut",
                                "Stress and digestion"
                            ]
                            
                            for action in quick_actions:
                                btn = gr.Button(action, variant="secondary", size="sm")
                                btn.click(
                                    lambda x=action: self.chat_interface.process_message(x, []),
                                    inputs=[],
                                    outputs=[chatbot]
                                )
                    
                    # Side panel
                    with gr.Column(scale=1):
                        # Information panel
                        with gr.Accordion("ℹ️ About This Coach", open=False):
                            gr.Markdown("""
                            **Your Gut Health Coach provides:**
                            - Evidence-based gut health information
                            - Empathetic, supportive guidance
                            - Safety-first medical approach
                            - Personalized recommendations
                            
                            **Sources:**
                            - Mayo Clinic
                            - NIH/NIDDK
                            - Healthline
                            
                            **⚠️ Important:** This is educational information only. 
                            Always consult healthcare professionals for medical advice.
                            """)
                        
                        # Metrics display
                        with gr.Accordion("📊 Session Stats", open=False):
                            metrics_display = gr.JSON(
                                label="Pipeline Metrics",
                                value=self.rag_pipeline.get_pipeline_metrics()
                            )
                        
                        # Conversation controls
                        with gr.Accordion("🔧 Controls", open=False):
                            clear_btn = gr.Button("Clear Conversation", variant="secondary")
                            export_btn = gr.Button("Export Chat", variant="secondary")
                            
                            # Feedback
                            gr.Markdown("**Rate this conversation:**")
                            rating = gr.Slider(
                                minimum=1, maximum=5, step=1, value=5,
                                label="Helpfulness (1-5)"
                            )
                            feedback_text = gr.Textbox(
                                placeholder="Any feedback or suggestions?",
                                label="Feedback",
                                lines=3
                            )
                            submit_feedback = gr.Button("Submit Feedback", variant="primary")
                
                # Set up interactions
                def respond_to_message(message, history):
                    """Handle user message and generate response"""
                    try:
                        # Process message through RAG pipeline
                        response = self.rag_pipeline.process_query(message)
                        
                        # Update metrics
                        self.app_metrics["total_queries"] += 1
                        
                        # Format response for chat
                        formatted_response = self._format_response_for_chat(response)
                        
                        # Update history
                        history.append({"role": "user", "content": message})
                        history.append({"role": "assistant", "content": formatted_response})
                        
                        return "", history
                        
                    except Exception as e:
                        error_msg = f"I apologize, but I encountered an error: {str(e)}\n\nPlease try again or consult with a healthcare professional."
                        history.append({"role": "user", "content": message})
                        history.append({"role": "assistant", "content": error_msg})
                        return "", history
                
                def clear_conversation():
                    """Clear the conversation"""
                    self.rag_pipeline.clear_conversation_history()
                    return []
                
                def export_conversation(history):
                    """Export conversation to JSON"""
                    try:
                        export_data = {
                            "timestamp": datetime.now().isoformat(),
                            "conversation": history,
                            "metrics": self.rag_pipeline.get_pipeline_metrics()
                        }
                        
                        filename = f"gut_health_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        
                        return gr.File.update(
                            value=json.dumps(export_data, indent=2),
                            visible=True,
                            label=filename
                        )
                        
                    except Exception as e:
                        return f"Export failed: {str(e)}"
                
                def submit_user_feedback(rating_value, feedback_text_value):
                    """Handle user feedback submission"""
                    try:
                        feedback_data = {
                            "timestamp": datetime.now().isoformat(),
                            "rating": rating_value,
                            "feedback": feedback_text_value,
                            "session_metrics": self.rag_pipeline.get_pipeline_metrics()
                        }
                        
                        # Save feedback (in real app, this would go to a database)
                        feedback_file = Path("feedback") / f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        feedback_file.parent.mkdir(exist_ok=True)
                        
                        with open(feedback_file, 'w') as f:
                            json.dump(feedback_data, f, indent=2)
                        
                        return "Thank you for your feedback! 🙏"
                        
                    except Exception as e:
                        return f"Feedback submission failed: {str(e)}"
                
                # Wire up interactions
                msg_input.submit(respond_to_message, [msg_input, chatbot], [msg_input, chatbot])
                send_btn.click(respond_to_message, [msg_input, chatbot], [msg_input, chatbot])
                
                clear_btn.click(clear_conversation, outputs=[chatbot])
                
                submit_feedback.click(
                    submit_user_feedback,
                    inputs=[rating, feedback_text],
                    outputs=[gr.Textbox(label="Feedback Status")]
                )
                
                # Add example conversations
                examples = gr.Examples(
                    examples=[
                        ["I've been experiencing bloating after meals. What could be causing this?"],
                        ["What are some natural ways to improve gut health?"],
                        ["I have IBS and I'm looking for dietary advice."],
                        ["How does stress affect digestive health?"],
                        ["What are the signs of a healthy gut microbiome?"]
                    ],
                    inputs=[msg_input],
                    outputs=[chatbot],
                    fn=lambda x: respond_to_message(x, []),
                    cache_examples=False
                )
                
                # Footer
                gr.HTML("""
                <div style="text-align: center; padding: 20px; color: #666; font-size: 0.9em;">
                    <p><strong>⚠️ Medical Disclaimer:</strong> This AI assistant provides educational information only. 
                    Always consult with qualified healthcare professionals for medical advice, diagnosis, or treatment.</p>
                    <p>Built with 💝 for better gut health • Open source RAG pipeline • BioMistral-7B + PubMedBERT</p>
                </div>
                """)
            
            return interface
            
        except Exception as e:
            print(f"❌ Failed to create interface: {e}")
            raise
    
    def _format_response_for_chat(self, response) -> str:
        """Format RAG response for chat display"""
        try:
            formatted = response.assistant_response
            
            # Add metadata footer
            footer = f"\n\n---\n📊 **Response Info**: {response.response_time:.2f}s • "
            footer += f"💗 Empathy: {response.empathy_score:.2f} • "
            footer += f"📚 Sources: {len(response.sources_used)}"
            
            if response.safety_flags:
                footer += f" • ⚠️ Flags: {len(response.safety_flags)}"
            
            return formatted + footer
            
        except Exception as e:
            print(f"Failed to format response: {e}")
            return response.assistant_response if hasattr(response, 'assistant_response') else str(response)
    
    def launch(self, **kwargs):
        """
        Launch the Gradio application
        
        Args:
            **kwargs: Additional arguments for gr.launch()
        """
        try:
            # Create interface
            interface = self.create_interface()
            
            # Default launch settings
            launch_settings = {
                "server_name": "0.0.0.0",
                "server_port": 7860,
                "share": False,
                "debug": False,
                "show_error": True,
                "quiet": False
            }
            
            # Update with provided kwargs
            launch_settings.update(kwargs)
            
            print("🚀 Launching Gut Health Coach...")
            print(f"📊 Pipeline metrics: {self.rag_pipeline.get_pipeline_metrics()}")
            
            # Launch the interface
            interface.launch(**launch_settings)
            
        except Exception as e:
            print(f"❌ Failed to launch application: {e}")
            raise

def main():
    """
    Main entry point for the application
    """
    try:
        # Create and launch the app
        app = GutHealthCoachApp()
        app.launch(
            share=False,  # Set to True for public sharing
            debug=True,   # Enable debug mode for development
            show_error=True
        )
        
    except Exception as e:
        print(f"❌ Application failed to start: {e}")
        raise

if __name__ == "__main__":
    main()
