# 🌱 Gut Health Coach - Empathetic AI Health Assistant

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![RAG](https://img.shields.io/badge/RAG-Pipeline-green.svg)](https://github.com/yourusername/gut-health-coach)
[![BioMistral](https://img.shields.io/badge/Model-BioMistral--7B-red.svg)](https://huggingface.co/MaziyarPanahi/BioMistral-7B-GGUF)

An intelligent, empathetic AI health assistant specifically designed for gut health guidance. This system combines cutting-edge medical AI with retrieval-augmented generation (RAG) to provide personalized, evidence-based health information in an August AI-style conversational manner.

## 🎯 Project Overview

The Gut Health Coach is a sophisticated RAG-based system that delivers empathetic, medically accurate responses to gut health queries. Built with production-ready architecture, it integrates specialized medical language models with authoritative health sources to create a trusted health companion.

### ✨ Key Features

- **🧠 Medical Domain Expertise**: Powered by BioMistral-7B, achieving 57.3% accuracy on medical benchmarks
- **💝 Quantified Empathy**: Custom empathy scoring engine (0-1.0 scale) for August AI-style responses
- **📚 Authoritative Sources**: Curated medical content from Mayo Clinic, NIH/NIDDK, and Healthline
- **⚡ Fast Retrieval**: FAISS vector database with PubMedBERT embeddings for biomedical understanding
- **🛡️ Safety First**: Comprehensive medical boundary detection and emergency protocols
- **📊 Real-time Metrics**: Performance monitoring with response times, empathy scores, and source attribution
- **🖥️ User-Friendly Interface**: Professional Gradio web application with responsive design

### Core Components

- **Data Layer**: Ethical web scraping from authoritative medical sources
- **Vector Store**: FAISS database with 768-dimensional PubMedBERT embeddings
- **Language Model**: BioMistral-7B (4.37GB) for specialized medical responses
- **Empathy Engine**: Custom algorithms for emotional intelligence scoring
- **User Interface**: Gradio-based chat interface with real-time metrics

## 🚀 Quick Start

### Prerequisites

- Python 3.11 (recommended for optimal compatibility)
- 8GB+ RAM (16GB recommended for optimal performance)
- 10GB free storage space
- Internet connection for model download

### Installation Guide

#### Step 1: Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/gut-health-coach.git
cd gut-health-coach

# Create virtual environment (Python 3.11 recommended)
python -m venv .venv

# Activate virtual environment
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

#### Step 2: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Additional packages for performance testing (optional)
pip install psutil
```

#### Step 3: Download BioMistral-7B Model

**Important**: The BioMistral-7B model (4.37GB) is not included in the repository due to size constraints.

```bash
# Install hugging face CLI
pip install huggingface-hub

# Download the model
huggingface-cli download MaziyarPanahi/BioMistral-7B-GGUF BioMistral-7B.Q4_K_M.gguf --local-dir ./models --local-dir-use-symlinks False

# Rename for consistency
cd models
mv BioMistral-7B.Q4_K_M.gguf biomistral-7b-q4.gguf
cd ..
```

**Alternative Download Method**:
```python
# Using Python script
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="MaziyarPanahi/BioMistral-7B-GGUF",
    filename="BioMistral-7B.Q4_K_M.gguf",
    local_dir="./models",
    local_dir_use_symlinks=False
)
```

### 🎯 Running the Application

#### Phase 1: Data Collection and Processing

```bash
cd src/

# 1. Collect medical data from authoritative sources
python data_collection.py

# 2. Process and chunk the collected data
python preprocessing.py

# 3. Generate PubMedBERT embeddings
python embeddings.py
```

**Expected Output**:
- ~20 medical articles from Mayo Clinic, NIH, Healthline
- ~200-300 semantic chunks
- 768-dimensional embeddings with quality validation

#### Phase 2: RAG Pipeline Setup

```bash
# 4. Initialize FAISS vector store
python vectorstore.py

# 5. Test LLM interface
python llm_interface.py

# 6. Test complete RAG pipeline
python rag_pipeline.py

# 7. Test empathy engine
python tone_engineering.py
```

#### Phase 3: Launch Web Interface

```bash
cd ../ui/

# Launch Gradio application
python gradio_app.py
```

The application will be available at: **http://localhost:7860**

## 📁 Project Structure

```
gut_health_coach/
├── data/
│   ├── raw/                    # Original scraped medical documents
│   ├── processed/              # Cleaned and preprocessed data
│   └── chunks/                 # Document chunks for embedding
├── models/
│   ├── embeddings/             # PubMedBERT embeddings cache
│   └── biomistral-7b-q4.gguf  # BioMistral-7B model (download required)
├── vectorstore/
│   ├── faiss_index/           # FAISS vector database files
│   └── metadata.json          # Document metadata
├── src/
│   ├── data_collection.py      # Ethical web scraping
│   ├── preprocessing.py        # Text cleaning and chunking
│   ├── embeddings.py          # PubMedBERT embedding generation
│   ├── vectorstore.py         # FAISS database management
│   ├── llm_interface.py       # BioMistral model interface
│   ├── rag_pipeline.py        # Main RAG orchestration
│   └── tone_engineering.py    # Empathy and tone management
├── ui/
│   ├── gradio_app.py          # Main Gradio interface
│   ├── chat_interface.py      # Chat logic and memory
│   └── styling.css            # Custom UI styling
├── tests/
│   ├── critical_questions.py  # Evaluation framework
│   └── performance_tests.py   # Performance benchmarking
├── config/
│   ├── settings.py            # Configuration management
│   ├── prompts.py             # Prompt templates
│   └── medical_disclaimers.py # Safety and legal text
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🧪 Testing and Evaluation

### Critical Questions Evaluation

```bash
cd tests/

# Run comprehensive evaluation on 10 critical medical questions
python critical_questions.py
```

**Evaluation Metrics**:
- Medical Accuracy Score
- Empathy Score (0-1.0 scale)
- Response Completeness
- Safety Handling
- Overall Performance Grade

### Performance Testing

```bash
# Run performance benchmarks
python performance_tests.py
```

**Performance Metrics**:
- Response time consistency
- Memory usage monitoring
- Concurrent load testing
- Error recovery assessment

## 🔧 Configuration

### Key Configuration Files

- **`config/settings.py`**: Main configuration with data sources, processing parameters, and model settings
- **`config/prompts.py`**: Empathetic response templates and system prompts
- **`config/medical_disclaimers.py`**: Safety boundaries and medical disclaimers

### Customization Options

```python
# Adjust response generation parameters
GENERATION_CONFIG = {
    'temperature': 0.3,      # Conservative for medical accuracy
    'max_tokens': 512,       # Response length
    'top_p': 0.8,           # Nucleus sampling
    'top_k': 40             # Top-k sampling
}

# Modify empathy scoring weights
EMPATHY_WEIGHTS = {
    'validation': 0.3,       # Acknowledging user concerns
    'reassurance': 0.25,     # Providing comfort
    'personalization': 0.25, # Addressing specific needs
    'boundaries': 0.2        # Medical safety limits
}
```

## 📊 Technical Specifications

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8GB | 16GB |
| **Storage** | 10GB free | 20GB free |
| **CPU** | 4 cores | 8+ cores |
| **GPU** | Not required | Optional (CUDA support) |
| **Python** | 3.11+ | 3.11.x |

### Model Performance

| Metric | BioMistral-7B | General LLMs |
|--------|---------------|--------------|
| **Medical Benchmark Accuracy** | 57.3% | 55.9% |
| **USMLE Performance** | Specialized | General |
| **Inference Speed** | 10-20 tokens/sec (CPU) | Similar |
| **Memory Usage** | ~4GB RAM | Similar |

## 🎯 Usage Examples

### Basic Query Example

```python
# Example interaction
user_query = "I've been experiencing bloating after meals. What could be causing this?"

# System processes through:
# 1. Vector retrieval from medical knowledge base
# 2. Context-aware response generation
# 3. Empathy scoring and safety checks
# 4. Response with sources and metrics

response = {
    "content": "I understand your concern about bloating after meals...",
    "empathy_score": 0.78,
    "sources": ["Mayo Clinic", "Healthline"],
    "response_time": "45.2s",
    "safety_flags": []
}
```

## 🛡️ Safety and Ethics

### Medical Boundaries

- **No Diagnosis**: System explicitly avoids providing medical diagnoses
- **Professional Referrals**: Recommends healthcare consultation for concerning symptoms
- **Emergency Detection**: Identifies red flag symptoms requiring immediate care
- **Source Attribution**: All medical information traceable to authoritative sources

### Data Privacy

- **No Data Storage**: User conversations are not permanently stored
- **Local Processing**: All computation happens locally (no external API calls)
- **GDPR Compliance**: Privacy-by-design architecture
- **Ethical Scraping**: Respects robots.txt and terms of service

## 🚧 Known Limitations and Future Improvements

### Current Limitations

- **Response Time**: 45-90 seconds per query (optimization in progress)
- **Knowledge Cutoff**: Limited to scraped medical content
- **Language Support**: English only
- **Hardware Requirements**: Requires 8GB+ RAM for optimal performance

### Planned Improvements

- [ ] **Performance Optimization**: Target <30 second response times
- [ ] **Knowledge Base Expansion**: Additional medical sources and specialties
- [ ] **Multi-language Support**: Spanish and other major languages
- [ ] **Mobile Optimization**: Responsive design improvements
- [ ] **Caching Layer**: Implement response caching for common queries
- [ ] **API Development**: RESTful API for third-party integrations

## 📈 Evaluation Results

### Critical Questions Performance

| Metric | Score | Grade |
|--------|-------|-------|
| **Overall Performance** | 0.537 | C+ |
| **Medical Accuracy** | 0.600 | B- |
| **Empathy Score** | 0.629 | B- |
| **Safety Handling** | 0.750 | B+ |
| **Response Time** | 88.0s | Needs Improvement |

*Note: Continuous improvements are being made to enhance performance metrics*

## 🤝 Contributing

I welcome contributions to improve the Gut Health Coach! Here's how you can help:

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`python tests/critical_questions.py`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Contribution Areas

- **Bug Fixes**: Report and fix issues
- **Performance**: Optimize response times and memory usage
- **Content**: Improve medical knowledge base
- **Internationalization**: Add multi-language support
- **Testing**: Expand test coverage and evaluation metrics

## References and Citations

### Medical Sources

- **Mayo Clinic**: Digestive health and gastroenterology information
- **NIH/NIDDK**: National Institute of Diabetes and Digestive and Kidney Diseases
- **Healthline**: Evidence-based health and wellness information

### Technical References

- **BioMistral-7B**: [MaziyarPanahi/BioMistral-7B-GGUF](https://huggingface.co/MaziyarPanahi/BioMistral-7B-GGUF)
- **PubMedBERT**: [NeuML/pubmedbert-base-embeddings](https://huggingface.co/NeuML/pubmedbert-base-embeddings)
- **FAISS**: Facebook AI Similarity Search for efficient vector operations
- **LangChain**: Framework for developing applications with large language models


## ⚠️ Medical Disclaimer

**Important**: This AI assistant provides educational information only and is not intended to replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for personalized medical guidance. In case of medical emergencies, contact emergency services immediately.

---

<div align="center">

**Made with ❤️ for better gut health**

[⭐ Star this repo](https://github.com/bokkadurgaprasad/gut-health-coach) | [Report Bug](https://github.com/bokkadurgaprasad/gut-health-coach/issues) | [💡 Request Feature](https://github.com/bokkadurgaprasad/gut-health-coach/issues)

</div>
