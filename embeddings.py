
import os
import numpy as np
from sentence_transformers import SentenceTransformer
MODEL = SentenceTransformer("all-MiniLM-L6-v2")
def make_chunks(text, size=800, overlap=150):
    text = text.strip()
    step = size - overlap
    return [
        text[i:i + size].strip()
        for i in range(0, len(text), step)
        if text[i:i + size].strip()
    ]
def get_embeddings(texts, batch_size=10):
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
    result = MODEL.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return np.array(result, dtype=np.float32)
