
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from embeddings import get_embeddings

def search(question, chunks, embeddings, top_k=3):
    if not chunks or len(embeddings) == 0:
        return []

    question_embedding = get_embeddings([question])
    scores = cosine_similarity(question_embedding, embeddings)[0]
    indices = np.argsort(scores)[::-1][:top_k]

    return [chunks[i] for i in indices]
