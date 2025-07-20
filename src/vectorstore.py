"""
FAISS vector database management for gut health knowledge base
"""
import faiss
import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import *

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VectorSearchResult:
    """Structured result from vector search"""
    content: str
    metadata: Dict[str, Any]
    score: float
    source: str
    chunk_id: str

class GutHealthVectorStore:
    """
    Specialized FAISS vector store for gut health medical content
    """
    
    def __init__(self, embedding_model_name: str = "NeuML/pubmedbert-base-embeddings"):
        """
        Initialize the vector store with PubMedBERT embeddings
        
        Args:
            embedding_model_name: Name of the embedding model to use
        """
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        self.vector_store = None
        self.embeddings = None
        self.metadata = None
        
        # Initialize embedding model
        self._initialize_embedding_model()
        
    def _initialize_embedding_model(self):
        """Initialize the PubMedBERT embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query using PubMedBERT
        
        Args:
            query: Query text to embed
            
        Returns:
            List of embedding values
        """
        try:
            embedding = self.embedding_model.encode(query, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise
    
    def create_from_embeddings(self, embeddings_file: str, metadata_file: str) -> None:
        """
        Create FAISS vector store from pre-computed embeddings
        
        Args:
            embeddings_file: Path to numpy embeddings file
            metadata_file: Path to metadata JSON file
        """
        try:
            logger.info("Creating FAISS vector store from embeddings...")
            
            # Load embeddings and metadata
            self.embeddings = np.load(embeddings_file)
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            
            # Create documents from metadata
            documents = []
            for i, chunk_info in enumerate(self.metadata['chunks']):
                doc = Document(
                    page_content=chunk_info.get('content', chunk_info.get('text')),
                    metadata={
                        'source': chunk_info['source'],
                        'chunk_id': chunk_info['chunk_id'],
                        'medical_entities': chunk_info.get('medical_entities', []),
                        'chunk_index': i
                    }
                )
                documents.append(doc)
            
            # Create FAISS index
            dimension = self.embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(self.embeddings.astype(np.float32))
            
            # Create docstore
            docstore = InMemoryDocstore()
            index_to_docstore_id = {}
            
            for i, doc in enumerate(documents):
                doc_id = str(i)
                docstore.add({doc_id: doc})
                index_to_docstore_id[i] = doc_id
            
            # Create FAISS vector store
            self.vector_store = FAISS(
                embedding_function=self.embed_query,
                index=index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id
            )
            
            logger.info(f"FAISS vector store created with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise
    
    def similarity_search_with_relevance_scores(
        self, 
        query: str, 
        k: int = 3,
        score_threshold: float = 0.0
    ) -> List[VectorSearchResult]:
        """
        Perform similarity search with relevance scoring
        
        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum relevance score
            
        Returns:
            List of VectorSearchResult objects
        """
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized")
            
            # Perform similarity search with scores
            docs_with_scores = self.vector_store.similarity_search_with_relevance_scores(
                query, k=k, score_threshold=score_threshold
            )
            
            # Convert to structured results
            results = []
            for doc, score in docs_with_scores:
                result = VectorSearchResult(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=score,
                    source=doc.metadata.get('source', 'unknown'),
                    chunk_id=doc.metadata.get('chunk_id', 'unknown')
                )
                results.append(result)
            
            logger.info(f"Found {len(results)} relevant documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def get_context_for_query(self, query: str, max_context_length: int = 2000) -> str:
        """
        Get formatted context for RAG pipeline
        
        Args:
            query: User query
            max_context_length: Maximum context length
            
        Returns:
            Formatted context string
        """
        try:
            # Search for relevant documents
            search_results = self.similarity_search_with_relevance_scores(
                query, k=3, score_threshold=0.0
            )
            
            if not search_results:
                return "No relevant medical information found in knowledge base."
            
            # Format context
            context_parts = []
            current_length = 0
            
            for i, result in enumerate(search_results):
                source_info = f"Source: {result.source} (Relevance: {result.score:.2f})"
                content_with_source = f"{source_info}\n{result.content}\n"
                
                if current_length + len(content_with_source) > max_context_length:
                    break
                
                context_parts.append(content_with_source)
                current_length += len(content_with_source)
            
            return "\n---\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Failed to get context: {e}")
            return "Error retrieving medical information."
    
    def save_vector_store(self, save_path: str) -> None:
        """
        Save FAISS vector store to disk
        
        Args:
            save_path: Directory path to save the vector store
        """
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized")
            
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            self.vector_store.save_local(str(save_path))
            
            # Save additional metadata
            metadata_path = save_path / "vector_store_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    'created_at': datetime.now().isoformat(),
                    'embedding_model': self.embedding_model_name,
                    'num_documents': len(self.vector_store.docstore._dict),
                    'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else None
                }, f, indent=2)
            
            logger.info(f"✅ Vector store saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            raise
    
    def load_vector_store(self, load_path: str) -> None:
        """
        Load FAISS vector store from disk
        
        Args:
            load_path: Directory path to load the vector store from
        """
        try:
            load_path = Path(load_path)
            
            if not load_path.exists():
                raise FileNotFoundError(f"Vector store not found at {load_path}")
            
            self.vector_store = FAISS.load_local(
                str(load_path), 
                embeddings=self.embed_query,
                allow_dangerous_deserialization=True
            )
            
            logger.info(f"✅ Vector store loaded from {load_path}")
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise

def main():
    """
    Main function to create and test the vector store
    """
    try:
        # Initialize vector store
        vector_store = GutHealthVectorStore()
        
        # Find latest embeddings and metadata files
        embeddings_dir = EMBEDDINGS_DIR
        latest_embeddings = max(embeddings_dir.glob("embeddings_*.npy"))
        latest_metadata = max(embeddings_dir.glob("metadata_*.json"))
        
        logger.info(f"Using embeddings: {latest_embeddings}")
        logger.info(f"Using metadata: {latest_metadata}")
        
        # Create vector store from embeddings
        vector_store.create_from_embeddings(
            str(latest_embeddings),
            str(latest_metadata)
        )
        
        # Save vector store
        vector_store.save_vector_store(str(FAISS_INDEX_DIR))
        
        # Test search functionality
        test_queries = [
            "What causes bloating after eating?",
            "How to improve gut health naturally?",
            "What are symptoms of SIBO?",
            "Foods to avoid for IBS",
            "How does stress affect digestion?"
        ]
        
        logger.info("\n🔍 Testing vector search functionality:")
        for query in test_queries:
            results = vector_store.similarity_search_with_relevance_scores(query, k=2)
            logger.info(f"\nQuery: {query}")
            logger.info(f"Results: {len(results)}")
            
            for i, result in enumerate(results):
                logger.info(f"  {i+1}. Source: {result.source} (Score: {result.score:.3f})")
                logger.info(f"     Content: {result.content[:100]}...")
        
        logger.info("\n✅ Vector store setup and testing completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Vector store setup failed: {e}")
        raise

if __name__ == "__main__":
    main()
