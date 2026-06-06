import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
import faiss

class EmbeddingModel:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the EmbeddingModel. 
        We use 'all-MiniLM-L6-v2' as it is highly efficient, runs perfectly on CPUs, 
        and produces solid semantic embeddings for our hackathon constraints.
        """
        print(f"Loading SentenceTransformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generates dense vector embeddings for a list of strings.
        Returns a numpy array of shape (len(texts), embedding_dim).
        """
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        return embeddings

class CandidateFAISS:
    def __init__(self, embedding_dim: int = 384):
        """
        Initializes the FAISS index for L2 distance (or cosine similarity).
        For all-MiniLM-L6-v2, the embedding dimension is 384.
        """
        # Using L2 distance. (Note: sentence-transformers are normalized if instructed, 
        # but L2 on normalized vectors is equivalent to cosine similarity).
        self.index = faiss.IndexFlatL2(embedding_dim)
        
    def add_embeddings(self, embeddings: np.ndarray):
        """
        Adds candidate embeddings to the FAISS index.
        """
        self.index.add(embeddings)
        print(f"Added {len(embeddings)} vectors to FAISS index. Total: {self.index.ntotal}")
        
    def search(self, query_embedding: np.ndarray, k: int = 5):
        """
        Searches the FAISS index for the top k closest vectors to the query.
        Returns distances and indices of the top k results.
        """
        # FAISS expects 2D arrays, so we ensure the query is reshaped if necessary
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        distances, indices = self.index.search(query_embedding, k)
        return distances[0], indices[0]
