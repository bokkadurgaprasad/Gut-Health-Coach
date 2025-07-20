"""
Configuration settings for the Gut Health Coach RAG system
"""
import os
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHUNKS_DATA_DIR = DATA_DIR / "chunks"
MODELS_DIR = PROJECT_ROOT / "models"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
FAISS_INDEX_DIR = VECTORSTORE_DIR / "faiss_index"

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CHUNKS_DATA_DIR, 
                  EMBEDDINGS_DIR, FAISS_INDEX_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data collection settings with authoritative medical sources
DATA_SOURCES = {
    "mayo_clinic": {
        "base_url": "https://www.mayoclinic.org",
        "gut_health_urls": [
            "/diseases-conditions/gastritis/symptoms-causes/syc-20355807",
            "/diseases-conditions/gastroparesis/symptoms-causes/syc-20453808",
            "/diseases-conditions/inflammatory-bowel-disease/symptoms-causes/syc-20353315",
            "/diseases-conditions/irritable-bowel-syndrome/symptoms-causes/syc-20360016",
            "/diseases-conditions/ulcerative-colitis/symptoms-causes/syc-20353326",
            "/diseases-conditions/crohns-disease/symptoms-causes/syc-20353304",
            "/diseases-conditions/celiac-disease/symptoms-causes/syc-20352220",
            "/diseases-conditions/diverticulitis/symptoms-causes/syc-20371758",
            "/diseases-conditions/gerd/symptoms-causes/syc-20361940",
            "/diseases-conditions/lactose-intolerance/symptoms-causes/syc-20374232"
        ]
    },
    "healthline": {
        "base_url": "https://www.healthline.com",
        "gut_health_urls": [
            "/health/gut-health",
            "/nutrition/ways-to-improve-digestion",
            "/health/digestive-health/foods-for-digestion",
            "/health/food-nutrition/leaky-gut-diet",
            "/health/digestive-health/probiotics-and-digestive-health",
            "/nutrition/gut-microbiome-and-health",
            "/health/digestive-health/ibs-diet",
            "/nutrition/foods-that-help-digestion",
            "/health/digestive-health/sibo-diet",
            "/nutrition/fermented-foods"
        ]
    },
    "niddk": {
        "base_url": "https://www.niddk.nih.gov",
        "gut_health_urls": [
            "/health-information/digestive-diseases/irritable-bowel-syndrome",
            "/health-information/digestive-diseases/crohns-disease",
            "/health-information/digestive-diseases/ulcerative-colitis",
            "/health-information/digestive-diseases/celiac-disease",
            "/health-information/digestive-diseases/gastritis",
            "/health-information/digestive-diseases/gastroparesis",
            "/health-information/digestive-diseases/gas",
            "/health-information/digestive-diseases/constipation",
            "/health-information/digestive-diseases/diarrhea",
            "/health-information/digestive-diseases/hemorrhoids"
        ]
    }
}

# Ethical web scraping configuration
SCRAPING_CONFIG = {
    "headers": {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    },
    "delay_between_requests": 2.0,  # Respectful rate limiting
    "timeout": 30,
    "max_retries": 3,
    "respect_robots_txt": True
}

# Text processing optimized for medical content
PROCESSING_CONFIG = {
    "chunk_size": 750,  # Optimal for PubMedBERT
    "chunk_overlap": 75,  # Context preservation
    "min_chunk_size": 100,
    "max_chunk_size": 1000,
    "remove_empty_chunks": True,
    "language": "en"
}

# PubMedBERT embedding configuration
EMBEDDING_CONFIG = {
    "model_name": "NeuML/pubmedbert-base-embeddings",
    "model_cache_dir": str(EMBEDDINGS_DIR),
    "batch_size": 32,
    "max_seq_length": 512,
    "normalize_embeddings": True
}

# FAISS vector database settings
FAISS_CONFIG = {
    "index_type": "IndexFlatL2",
    "dimension": 768,  # PubMedBERT dimension
    "nlist": 100,
    "nprobe": 10,
    "metric_type": "METRIC_L2"
}

# Critical test questions for evaluation
CRITICAL_TEST_QUESTIONS = [
    "I've been bloated for three days — what should I do?",
    "How does gut health affect sleep?",
    "What are the best probiotics for lactose intolerance?",
    "What does mucus in stool indicate?",
    "I feel nauseous after eating fermented foods. Is that normal?",
    "Should I fast if my gut is inflamed?",
    "Can antibiotics damage gut flora permanently?",
    "How do I know if I have SIBO?",
    "What are signs that my gut is healing?",
    "Why do I feel brain fog after eating sugar?"
]

# Empathy patterns for August AI-style responses
EMPATHY_PATTERNS = {
    "validation": [
        "Your concern is completely valid",
        "What you're experiencing is more common than you think",
        "It's understandable that you're worried about this"
    ],
    "reassurance": [
        "This is something we can work on together",
        "There are effective approaches to help with this",
        "Small changes can make a significant difference"
    ],
    "actionability": [
        "Here are some steps you can take",
        "Let's start with some gentle approaches",
        "Consider trying these evidence-based strategies"
    ],
    "boundaries": [
        "I recommend consulting with your healthcare provider",
        "For persistent symptoms, please seek professional care"
    ]
}
