"""
PubMedBERT embedding generation for gut health medical content
Optimized for medical domain with batch processing and validation
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
from datetime import datetime
import logging
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings('ignore')

# Import configuration
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import *

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PubMedBERTEmbedder:
    """PubMedBERT embedding generator"""

    def __init__(self, config: Dict):
        self.config = config
        self.model_name = config["model_name"]
        self.cache_dir = config["model_cache_dir"]
        self.batch_size = config["batch_size"]
        self.max_seq_length = config["max_seq_length"]
        self.normalize_embeddings = config["normalize_embeddings"]

        # Initialize model
        self.model = self._load_model()
        self.embedding_dim = self._get_embedding_dimension()

        logger.info(f"Model loaded: {self.model_name}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")

    def _load_model(self):
        """Load PubMedBERT model"""
        try:
            from sentence_transformers import SentenceTransformer

            # Load model with caching
            model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_dir
            )

            # Set max sequence length
            model.max_seq_length = self.max_seq_length

            # Use GPU if available
            try:
                import torch
                if torch.cuda.is_available():
                    device = torch.device('cuda')
                    model = model.to(device)
                    logger.info("Using GPU for embeddings")
                else:
                    logger.info("Using CPU for embeddings")
            except ImportError:
                logger.info("PyTorch not available, using CPU")

            return model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        try:
            # Test with a small text
            test_embedding = self.model.encode(["test"])
            return test_embedding.shape[1]
        except:
            return 768  # Default PubMedBERT dimension

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        if not texts:
            return np.array([])

        logger.info(f"Encoding {len(texts)} texts...")

        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts]

        # Generate embeddings
        embeddings = self.model.encode(
            processed_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings
        )

        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for embedding"""
        if not text:
            return ""

        # Truncate if too long
        words = text.split()
        max_words = self.max_seq_length - 10  # Leave room for special tokens

        if len(words) > max_words:
            # Try to end at sentence boundary
            truncated = ' '.join(words[:max_words])
            last_period = truncated.rfind('.')
            if last_period > len(truncated) * 0.8:
                return truncated[:last_period + 1]
            return truncated

        return text

class EmbeddingValidator:
    """Validate embedding quality"""

    def __init__(self, expected_dim: int = 768):
        self.expected_dim = expected_dim

    def validate_embeddings(self, embeddings: np.ndarray, texts: List[str]) -> Dict:
        """Comprehensive embedding validation"""
        results = {
            'total_embeddings': len(embeddings),
            'embedding_dimension': embeddings.shape[1] if len(embeddings) > 0 else 0,
            'dimension_correct': embeddings.shape[1] == self.expected_dim if len(embeddings) > 0 else False,
            'has_nan': bool(np.isnan(embeddings).any()) if len(embeddings) > 0 else False,
            'has_inf': bool(np.isinf(embeddings).any()) if len(embeddings) > 0 else False,
            'all_zeros': bool(np.all(embeddings == 0)) if len(embeddings) > 0 else False,
            'statistics': self._calculate_statistics(embeddings),
            'quality_score': 0.0
        }

        # Calculate quality score
        results['quality_score'] = self._calculate_quality_score(results, texts)

        return results

    def _calculate_statistics(self, embeddings: np.ndarray) -> Dict:
        """Calculate embedding statistics"""
        if len(embeddings) == 0:
            return {}

        return {
            'mean': float(np.mean(embeddings)),
            'std': float(np.std(embeddings)),
            'min': float(np.min(embeddings)),
            'max': float(np.max(embeddings)),
            'median': float(np.median(embeddings))
        }

    def _calculate_quality_score(self, results: Dict, texts: List[str]) -> float:
        """Calculate overall quality score (0-1)"""
        score = 1.0

        # Dimension check
        if not results['dimension_correct']:
            score -= 0.3

        # NaN/Inf check
        if results['has_nan'] or results['has_inf']:
            score -= 0.4

        # All zeros check
        if results['all_zeros']:
            score -= 0.3

        # Text quality check
        empty_texts = sum(1 for text in texts if not text.strip())
        if empty_texts > 0:
            score -= 0.1 * (empty_texts / len(texts))

        return max(0.0, score)

class SimilarityAnalyzer:
    """Analyze embedding similarities"""

    def calculate_similarity_matrix(self, embeddings: np.ndarray, sample_size: int = 100) -> Dict:
        """Calculate similarity statistics"""
        if len(embeddings) == 0:
            return {}

        # Sample for efficiency
        if len(embeddings) > sample_size:
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            sample_embeddings = embeddings[indices]
        else:
            sample_embeddings = embeddings

        # Calculate cosine similarity matrix
        similarity_matrix = np.dot(sample_embeddings, sample_embeddings.T)

        # Remove diagonal (self-similarity)
        np.fill_diagonal(similarity_matrix, 0)

        # Calculate statistics
        stats = {
            'mean_similarity': float(np.mean(similarity_matrix)),
            'std_similarity': float(np.std(similarity_matrix)),
            'min_similarity': float(np.min(similarity_matrix)),
            'max_similarity': float(np.max(similarity_matrix)),
            'median_similarity': float(np.median(similarity_matrix)),
            'sample_size': len(sample_embeddings)
        }

        return stats

class MedicalEmbeddingProcessor:
    """Process medical text embeddings"""

    def __init__(self):
        self.embedder = PubMedBERTEmbedder(EMBEDDING_CONFIG)
        self.validator = EmbeddingValidator(self.embedder.embedding_dim)
        self.analyzer = SimilarityAnalyzer()

    def process_chunks(self, chunks: List[Dict]) -> Tuple[np.ndarray, List[Dict], Dict]:
        """Process chunks and generate embeddings"""
        if not chunks:
            return np.array([]), [], {}

        logger.info(f"Processing {len(chunks)} chunks...")

        # Extract texts and metadata
        texts = []
        metadata = []

        for chunk in chunks:
            text = chunk.get('text', '')
            if text.strip():
                texts.append(text)
                metadata.append(chunk)

        if not texts:
            logger.warning("No valid texts found!")
            return np.array([]), [], {}

        # Generate embeddings
        embeddings = self.embedder.encode_texts(texts)

        # Validate embeddings
        validation_results = self.validator.validate_embeddings(embeddings, texts)

        # Calculate similarity statistics
        similarity_stats = self.analyzer.calculate_similarity_matrix(embeddings)

        # Combine validation and similarity results
        quality_report = {
            'validation': validation_results,
            'similarity': similarity_stats,
            'processing_timestamp': datetime.now().isoformat()
        }

        logger.info(f"Quality score: {validation_results['quality_score']:.3f}")
        logger.info(f"Mean similarity: {similarity_stats.get('mean_similarity', 0):.3f}")

        return embeddings, metadata, quality_report

def load_latest_processed_data() -> List[Dict]:
    """Load the latest processed chunks"""
    processed_files = list(PROCESSED_DATA_DIR.glob("gut_health_chunks_*.json"))

    if not processed_files:
        raise FileNotFoundError("No processed data files found. Please run preprocessing.py first.")

    latest_file = max(processed_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Loading processed data from: {latest_file}")

    with open(latest_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_embeddings(embeddings: np.ndarray, metadata: List[Dict], quality_report: Dict) -> Tuple[str, str]:
    """Save embeddings and metadata"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save embeddings
    embeddings_path = EMBEDDINGS_DIR / f"embeddings_{timestamp}.npy"
    np.save(embeddings_path, embeddings)

    # Prepare metadata with quality info
    metadata_package = {
        'info': {
            'model_name': EMBEDDING_CONFIG['model_name'],
            'embedding_dimension': embeddings.shape[1],
            'total_embeddings': len(embeddings),
            'created_at': datetime.now().isoformat(),
            'quality_report': quality_report
        },
        'chunks': metadata
    }

    # Save metadata
    metadata_path = EMBEDDINGS_DIR / f"metadata_{timestamp}.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_package, f, indent=2, ensure_ascii=False)

    # Save summary CSV
    summary_data = []
    for i, chunk in enumerate(metadata):
        summary_data.append({
            'embedding_id': i,
            'chunk_id': chunk.get('chunk_id', ''),
            'source': chunk.get('source', ''),
            'title': chunk.get('title', ''),
            'word_count': chunk.get('word_count', 0),
            'entity_count': chunk.get('entity_count', 0),
            'entity_density': chunk.get('entity_density', 0),
            'preview': chunk.get('text', '')[:100] + '...' if len(chunk.get('text', '')) > 100 else chunk.get('text', '')
        })

    summary_df = pd.DataFrame(summary_data)
    summary_path = EMBEDDINGS_DIR / f"embeddings_summary_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)

    logger.info(f"Embeddings saved to: {embeddings_path}")
    logger.info(f"Metadata saved to: {metadata_path}")
    logger.info(f"Summary saved to: {summary_path}")

    return str(embeddings_path), str(metadata_path)

def main():
    """Main embedding generation function"""
    logger.info("Starting embedding generation...")

    # Load processed chunks
    try:
        chunks = load_latest_processed_data()
        logger.info(f"Loaded {len(chunks)} chunks")
    except FileNotFoundError as e:
        logger.error(str(e))
        return None

    # Initialize processor
    processor = MedicalEmbeddingProcessor()

    # Process chunks
    embeddings, metadata, quality_report = processor.process_chunks(chunks)

    if len(embeddings) == 0:
        logger.error("No embeddings generated!")
        return None

    # Save results
    embeddings_path, metadata_path = save_embeddings(embeddings, metadata, quality_report)

    # Print summary
    print("\nEmbedding Generation Summary:")
    print(f"Input chunks: {len(chunks)}")
    print(f"Generated embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Model: {EMBEDDING_CONFIG['model_name']}")
    print(f"Quality score: {quality_report['validation']['quality_score']:.3f}")

    # Quality details
    validation = quality_report['validation']
    if validation['has_nan']:
        print("⚠️  Warning: NaN values detected")
    if validation['has_inf']:
        print("⚠️  Warning: Infinite values detected")
    if validation['all_zeros']:
        print("⚠️  Warning: All-zero embeddings detected")

    if validation['quality_score'] >= 0.9:
        print("✅ High quality embeddings generated")
    elif validation['quality_score'] >= 0.7:
        print("⚠️  Medium quality embeddings generated")
    else:
        print("❌ Low quality embeddings - review data")

    # Similarity analysis
    similarity = quality_report['similarity']
    if similarity:
        print(f"\nSimilarity Analysis:")
        print(f"Mean similarity: {similarity['mean_similarity']:.3f}")
        print(f"Std similarity: {similarity['std_similarity']:.3f}")

    # Source distribution
    sources = {}
    for chunk in metadata:
        source = chunk.get('source', 'unknown')
        sources[source] = sources.get(source, 0) + 1

    print(f"\nEmbeddings by source:")
    for source, count in sources.items():
        print(f"  {source}: {count} embeddings")

    return embeddings_path, metadata_path

if __name__ == "__main__":
    main()
