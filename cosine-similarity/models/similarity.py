from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from .embeddings import EmbeddingsModel
from utils.lemmatizer import Lemmatizer


SIMILARITY_THRESHOLD = 0.70


class SimilarityChecker:
    def __init__(self):
        self.embeddings = EmbeddingsModel()
        self.lemmatizer = Lemmatizer()
        self.tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))

    def tfidf_similarity(self, text1: str, text2: str) -> float:
        try:
            tfidf_matrix = self.tfidf.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception:
            return 0.0

    def paraphrase_similarity(self, text1: str, text2: str) -> float:
        return self.embeddings.compute_similarity(text1, text2, 'paraphrase')

    def e5_similarity(self, text1: str, text2: str) -> float:
        return self.embeddings.compute_similarity(text1, text2, 'e5')

    def check(self, original: str, paraphrase: str) -> dict:
        paraphrase_score = self.paraphrase_similarity(original, paraphrase)
        e5_score = self.e5_similarity(original, paraphrase)
        tfidf_score = self.tfidf_similarity(original, paraphrase)

        average = (paraphrase_score + e5_score + tfidf_score) / 3

        used_words = self.lemmatizer.get_used_words(original, paraphrase)
        word_check_passed = len(used_words) == 0

        similarity_scores = {
            'paraphrase-multilingual': round(paraphrase_score, 2),
            'e5-multilingual': round(e5_score, 2),
            'tfidf': round(tfidf_score, 2),
            'average': round(average, 2)
        }

        passed_threshold = (
            average >= SIMILARITY_THRESHOLD and
            word_check_passed
        )

        return {
            'similarity': similarity_scores,
            'passed_threshold': passed_threshold,
            'word_check': {
                'passed': word_check_passed,
                'used_words': used_words
            },
            'lemmas': self.lemmatizer.get_lemmas_info(original, paraphrase),
            'threshold': SIMILARITY_THRESHOLD
        }