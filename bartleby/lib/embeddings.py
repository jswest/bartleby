import numpy as np

from sentence_transformers import SentenceTransformer


def embed_chunk(embedding_model: SentenceTransformer, body: str) -> np.ndarray:
    if not body or not body.strip():
        return np.zeros(embedding_model.get_sentence_embedding_dimension(), dtype=np.float32)

    tokens = embedding_model.tokenizer.tokenize(body)
    if len(tokens) > embedding_model.max_seq_length:
        raise ValueError(f"Body has too many tokens: {body}")

    embedding = embedding_model.encode(
        body,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return embedding.astype(np.float32, copy=False)