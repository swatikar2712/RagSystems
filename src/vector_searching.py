from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from document_preprocessing import DocumentPreprocessing


class VectorDBSearching(DocumentPreprocessing):

    def __init__(self):
        
        super().__init__()
        self.components = self.RecursiveSplitter()
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.texts = self.components[0]
        self.embeddings = self.model.encode(self.texts)
        self.faiss_embeddings = np.array(self.embeddings).astype("float32")

    def searching(self, query):
        self.index = faiss.read_index("C:\\Users\\ARYAN SURI\\Desktop\\Amlgo Labs Assignment\\vectordb\\faissIndex.faiss")
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")
        D, I = self.index.search(query_embedding, 5)
        results = []
        for i, idx in enumerate(I[0]):
            result = {
                        "chunk": self.texts[idx],
                        "distance": float(D[0][i])
                    }
        results.append(result)
        context = []
        for i, res in enumerate(results, 1):
            context.append(res['chunk'])
        return context

__all__ = ["VectorDBSearching"]    
        