from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from typing import Literal


class EmbeddingsModel:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._paraphrase_model = None
        self._e5_model = None

    @property
    def paraphrase_model(self):
        if self._paraphrase_model is None:
            self._paraphrase_model = SentenceTransformer(
                'paraphrase-multilingual-MiniLM-L12-v2',
                device=self.device
            )
        return self._paraphrase_model

    @property
    def e5_model(self):
        if self._e5_model is None:
            self._e5_model = SentenceTransformer(
                'intfloat/e5-base-v2',
                device=self.device
            )
        return self._e5_model

    def get_embedding(
        self,
        text: str,
        model_type: Literal['paraphrase', 'e5']
    ) -> np.ndarray:
        if model_type == 'paraphrase':
            model = self.paraphrase_model
        else:
            model = self.e5_model
            text = f"query: {text}"

        embedding = model.encode(text, convert_to_numpy=True)
        return embedding

    def compute_similarity(
        self,
        text1: str,
        text2: str,
        model_type: Literal['paraphrase', 'e5']
    ) -> float:
        emb1 = self.get_embedding(text1, model_type)
        emb2 = self.get_embedding(text2, model_type)

        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        similarity = dot_product / (norm1 * norm2)
        return float(similarity)