import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class EmbeddingModel:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts, batch_size=32):
        print(f"Generating embeddings for {len(texts)} texts...")
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)


class CandidateFAISS:
    def __init__(self, embedding_dim=384):
        self.index = faiss.IndexFlatL2(embedding_dim)

    def add_embeddings(self, embeddings):
        self.index.add(embeddings)
        print(f"Added {len(embeddings)} vectors to FAISS index.")

    def search(self, query_embedding, k=5):
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        return distances[0], indices[0]
