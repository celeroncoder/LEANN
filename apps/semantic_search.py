"""
Enhanced semantic search capabilities using sentence transformers
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class SemanticSearchEngine:
    """Semantic search engine using sentence transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.embeddings = {}
        self.texts = {}
        
    def load_model(self):
        """Load the sentence transformer model"""
        if self.model is None:
            try:
                logger.info(f"Loading semantic search model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                logger.info("Semantic search model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load semantic search model: {e}")
                self.model = None
    
    def add_document(self, doc_id: str, text: str, metadata: Dict[str, Any]):
        """Add a document to the semantic search index"""
        if self.model is None:
            self.load_model()
        
        if self.model is not None:
            try:
                # Generate embedding for the text
                embedding = self.model.encode(text, convert_to_tensor=False)
                self.embeddings[doc_id] = embedding
                self.texts[doc_id] = {"text": text, "metadata": metadata}
                logger.debug(f"Added document to semantic index: {doc_id}")
            except Exception as e:
                logger.error(f"Failed to add document to semantic index: {e}")
    
    def remove_document(self, doc_id: str):
        """Remove a document from the semantic search index"""
        if doc_id in self.embeddings:
            del self.embeddings[doc_id]
        if doc_id in self.texts:
            del self.texts[doc_id]
        logger.debug(f"Removed document from semantic index: {doc_id}")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str, Dict[str, Any]]]:
        """Perform semantic search"""
        if self.model is None or len(self.embeddings) == 0:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.model.encode(query, convert_to_tensor=False)
            
            # Calculate similarities
            similarities = []
            for doc_id, doc_embedding in self.embeddings.items():
                # Cosine similarity
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                similarities.append((doc_id, similarity))
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for doc_id, score in similarities[:top_k]:
                if doc_id in self.texts:
                    text_data = self.texts[doc_id]
                    results.append((doc_id, score, text_data["text"], text_data["metadata"]))
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def clear_index(self):
        """Clear the semantic search index"""
        self.embeddings.clear()
        self.texts.clear()
        logger.info("Semantic search index cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get semantic search statistics"""
        return {
            "model_name": self.model_name,
            "model_loaded": self.model is not None,
            "indexed_documents": len(self.embeddings),
            "total_embeddings_size_mb": sum(
                embedding.nbytes for embedding in self.embeddings.values()
            ) / (1024 * 1024) if self.embeddings else 0.0
        }