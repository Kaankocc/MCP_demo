# app/embedding.py
from sentence_transformers import SentenceTransformer

class CustomEmbeddings:
    def __init__(self):
        model = SentenceTransformer('avsolatorio/GIST-large-Embedding-v0')
        self.model = model

    def embed_query(self, text):
        return self.model.encode(text).tolist()

    def embed_documents(self, texts):
        return [self.model.encode(text).tolist() for text in texts]