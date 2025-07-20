"""
Text preprocessing and chunking for gut health medical content
Optimized for PubMedBERT embeddings and medical terminology
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
from datetime import datetime
import logging
from tqdm import tqdm
import sys
import hashlib

# Import configuration
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import *

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalTextCleaner:
    """Medical text cleaning and normalization"""

    def __init__(self):
        self.medical_abbreviations = {
            'GI': 'gastrointestinal',
            'IBD': 'inflammatory bowel disease',
            'IBS': 'irritable bowel syndrome',
            'GERD': 'gastroesophageal reflux disease',
            'SIBO': 'small intestinal bacterial overgrowth',
            'PPI': 'proton pump inhibitor',
            'NSAID': 'nonsteroidal anti-inflammatory drug',
            'H2': 'histamine-2',
            'FDA': 'Food and Drug Administration',
            'MD': 'medical doctor',
            'RD': 'registered dietitian'
        }

        self.medical_terms = {
            'conditions': [
                'gastritis', 'gastroparesis', 'ulcerative colitis', 'crohns disease',
                'celiac disease', 'diverticulitis', 'lactose intolerance', 'dysbiosis',
                'leaky gut syndrome', 'functional dyspepsia', 'microscopic colitis'
            ],
            'symptoms': [
                'abdominal pain', 'bloating', 'constipation', 'diarrhea', 'nausea',
                'vomiting', 'heartburn', 'acid reflux', 'gas', 'cramping',
                'food intolerance', 'malabsorption', 'weight loss'
            ],
            'treatments': [
                'probiotics', 'prebiotics', 'fiber supplement', 'elimination diet',
                'low fodmap diet', 'fermented foods', 'digestive enzymes',
                'anti-inflammatory diet', 'stress management', 'exercise'
            ]
        }

    def clean_text(self, text: str) -> str:
        """Clean and normalize medical text"""
        if not text:
            return ""

        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text.strip())

        # Expand medical abbreviations
        for abbr, full_form in self.medical_abbreviations.items():
            text = re.sub(rf'\b{abbr}\b', full_form, text, flags=re.IGNORECASE)

        # Remove non-essential punctuation but keep medical formatting
        text = re.sub(r'[^\w\s.,;:()\-/%]', '', text)

        # Fix common OCR/formatting errors
        text = re.sub(r'\s+([.,;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.,;:])([^\s])', r'\1 \2', text)  # Add space after punctuation

        # Normalize medical terminology
        text = self._normalize_medical_terms(text)

        return text.strip()

    def _normalize_medical_terms(self, text: str) -> str:
        """Normalize medical terminology for consistency"""
        # Common medical term variations
        normalizations = {
            'gastro-intestinal': 'gastrointestinal',
            'anti-inflammatory': 'anti-inflammatory',
            'pre-biotic': 'prebiotic',
            'pro-biotic': 'probiotic',
            'Crohn\'s disease': 'Crohns disease',
            'H. pylori': 'Helicobacter pylori'
        }

        for variant, normalized in normalizations.items():
            text = re.sub(variant, normalized, text, flags=re.IGNORECASE)

        return text

    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities from text"""
        entities = {category: [] for category in self.medical_terms.keys()}

        text_lower = text.lower()

        for category, terms in self.medical_terms.items():
            for term in terms:
                if term.lower() in text_lower:
                    entities[category].append(term)

        return entities

class TextChunker:
    """Intelligent text chunking for medical content"""

    def __init__(self, config: Dict):
        self.chunk_size = config['chunk_size']
        self.chunk_overlap = config['chunk_overlap']
        self.min_chunk_size = config['min_chunk_size']
        self.max_chunk_size = config['max_chunk_size']

    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """Split text into overlapping chunks"""
        if not text or len(text.split()) < self.min_chunk_size:
            return []

        # Split into sentences
        sentences = self._split_sentences(text)

        chunks = []
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            # Check if adding this sentence would exceed chunk size
            if current_word_count + sentence_words > self.chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.split()) >= self.min_chunk_size:
                    chunks.append(self._create_chunk(chunk_text, metadata, len(chunks)))

                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_word_count = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_words

        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text.split()) >= self.min_chunk_size:
                chunks.append(self._create_chunk(chunk_text, metadata, len(chunks)))

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences with medical awareness"""
        # Protect medical abbreviations
        protected_patterns = [
            r'Dr\.',
            r'Mr\.',
            r'Mrs\.',
            r'Ms\.',
            r'vs\.',
            r'i\.e\.',
            r'e\.g\.',
            r'etc\.',
            r'\d+\.\d+',  # Numbers with decimals
            r'\b[A-Z]\.\s*[A-Z]\.',  # Initials
        ]

        # Temporarily replace protected patterns
        protected_text = text
        replacements = {}

        for i, pattern in enumerate(protected_patterns):
            matches = re.findall(pattern, protected_text)
            for match in matches:
                placeholder = f"PROTECTED_{i}_{len(replacements)}"
                replacements[placeholder] = match
                protected_text = protected_text.replace(match, placeholder)

        # Split sentences
        sentences = re.split(r'[.!?]+\s+', protected_text)

        # Restore protected patterns
        for i, sentence in enumerate(sentences):
            for placeholder, original in replacements.items():
                sentences[i] = sentence.replace(placeholder, original)

        return [s.strip() for s in sentences if s.strip()]

    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap"""
        overlap_words = self.chunk_overlap
        overlap_sentences = []
        current_words = 0

        for sentence in reversed(sentences):
            sentence_words = len(sentence.split())
            if current_words + sentence_words <= overlap_words:
                overlap_sentences.insert(0, sentence)
                current_words += sentence_words
            else:
                break

        return overlap_sentences

    def _create_chunk(self, text: str, metadata: Dict, chunk_index: int) -> Dict:
        """Create a chunk with metadata"""
        return {
            'chunk_id': f"{metadata.get('source', 'unknown')}_{chunk_index}",
            'text': text,
            'word_count': len(text.split()),
            'char_count': len(text),
            'source': metadata.get('source', 'unknown'),
            'title': metadata.get('title', 'Unknown'),
            'url': metadata.get('url', ''),
            'chunk_index': chunk_index,
            'created_at': datetime.now().isoformat()
        }

class GutHealthPreprocessor:
    """Main preprocessing pipeline"""

    def __init__(self):
        self.text_cleaner = MedicalTextCleaner()
        self.text_chunker = TextChunker(PROCESSING_CONFIG)

    def process_articles(self, articles: List[Dict]) -> List[Dict]:
        """Process articles into clean chunks"""
        all_chunks = []

        for article in tqdm(articles, desc="Processing articles"):
            if not article.get('content'):
                continue

            # Clean text
            cleaned_text = self.text_cleaner.clean_text(article['content'])

            if not cleaned_text:
                continue

            # Extract medical entities
            entities = self.text_cleaner.extract_medical_entities(cleaned_text)

            # Prepare metadata
            metadata = {
                'source': article.get('source', 'unknown'),
                'title': article.get('title', 'Unknown'),
                'url': article.get('url', ''),
                'scraped_at': article.get('scraped_at', ''),
                'entities': entities
            }

            # Chunk the text
            chunks = self.text_chunker.chunk_text(cleaned_text, metadata)

            # Add entity information to chunks
            for chunk in chunks:
                chunk['entities'] = entities
                chunk['entity_count'] = sum(len(entity_list) for entity_list in entities.values())
                chunk['entity_density'] = chunk['entity_count'] / chunk['word_count'] if chunk['word_count'] > 0 else 0

            all_chunks.extend(chunks)

        return all_chunks

    def filter_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Filter chunks by quality"""
        # Remove duplicates
        unique_chunks = self._remove_duplicates(chunks)

        # Filter by quality criteria
        quality_chunks = []
        for chunk in unique_chunks:
            if self._is_quality_chunk(chunk):
                quality_chunks.append(chunk)

        # Sort by medical relevance
        quality_chunks.sort(key=lambda x: x['entity_density'], reverse=True)

        return quality_chunks

    def _remove_duplicates(self, chunks: List[Dict]) -> List[Dict]:
        """Remove duplicate chunks"""
        seen_hashes = set()
        unique_chunks = []

        for chunk in chunks:
            text_hash = hashlib.md5(chunk['text'].encode()).hexdigest()
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_chunks.append(chunk)

        return unique_chunks

    def _is_quality_chunk(self, chunk: Dict) -> bool:
        """Check if chunk meets quality criteria"""
        text = chunk['text']

        # Check minimum word count
        if chunk['word_count'] < PROCESSING_CONFIG['min_chunk_size']:
            return False

        # Check maximum word count
        if chunk['word_count'] > PROCESSING_CONFIG['max_chunk_size']:
            return False

        # Check if text is mostly meaningful (not just punctuation/numbers)
        word_chars = sum(1 for char in text if char.isalnum())
        if word_chars / len(text) < 0.5:
            return False

        # Check for medical relevance
        if chunk['entity_count'] == 0:
            # Still include if it has medical keywords
            medical_keywords = ['health', 'medical', 'treatment', 'symptom', 'doctor', 'patient', 'disease', 'condition']
            if not any(keyword in text.lower() for keyword in medical_keywords):
                return False

        return True

def load_latest_raw_data() -> List[Dict]:
    """Load the latest raw data file"""
    raw_files = list(RAW_DATA_DIR.glob("gut_health_raw_data_*.json"))

    if not raw_files:
        raise FileNotFoundError("No raw data files found. Please run data_collection.py first.")

    latest_file = max(raw_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Loading raw data from: {latest_file}")

    with open(latest_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_processed_data(chunks: List[Dict]) -> str:
    """Save processed chunks"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_path = PROCESSED_DATA_DIR / f"gut_health_chunks_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    # Save CSV for analysis
    csv_path = PROCESSED_DATA_DIR / f"gut_health_chunks_{timestamp}.csv"
    df = pd.DataFrame(chunks)
    df.to_csv(csv_path, index=False)

    logger.info(f"Saved {len(chunks)} chunks to {json_path}")
    return str(json_path)

def main():
    """Main preprocessing function"""
    logger.info("Starting text preprocessing...")

    # Load raw data
    try:
        articles = load_latest_raw_data()
        logger.info(f"Loaded {len(articles)} articles")
    except FileNotFoundError as e:
        logger.error(str(e))
        return None

    # Initialize preprocessor
    preprocessor = GutHealthPreprocessor()

    # Process articles
    chunks = preprocessor.process_articles(articles)
    logger.info(f"Generated {len(chunks)} initial chunks")

    # Filter chunks
    quality_chunks = preprocessor.filter_chunks(chunks)
    logger.info(f"Filtered to {len(quality_chunks)} quality chunks")

    # Save results
    output_path = save_processed_data(quality_chunks)

    # Print summary
    print("\nPreprocessing Summary:")
    print(f"Input articles: {len(articles)}")
    print(f"Generated chunks: {len(chunks)}")
    print(f"Quality chunks: {len(quality_chunks)}")
    print(f"Average chunk size: {sum(chunk['word_count'] for chunk in quality_chunks) / len(quality_chunks):.1f} words")

    # Source distribution
    sources = {}
    for chunk in quality_chunks:
        source = chunk['source']
        sources[source] = sources.get(source, 0) + 1

    print("\nChunks by source:")
    for source, count in sources.items():
        print(f"  {source}: {count} chunks")

    # Entity statistics
    total_entities = sum(chunk['entity_count'] for chunk in quality_chunks)
    avg_entity_density = sum(chunk['entity_density'] for chunk in quality_chunks) / len(quality_chunks)

    print(f"\nMedical entities found: {total_entities}")
    print(f"Average entity density: {avg_entity_density:.3f}")

    return output_path

if __name__ == "__main__":
    main()
