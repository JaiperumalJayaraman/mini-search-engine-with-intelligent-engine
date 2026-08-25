import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MiniSearchEngine:
    """Small TF-IDF search engine with lightweight query expansion and ranking."""

    def __init__(self, data_path="data/documents.json"):
        self.data_path = Path(data_path)
        self.documents = self._load_documents()
        self.stop_words = "english"
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=self.stop_words,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(
            [self._document_text(doc) for doc in self.documents]
        )

    def _load_documents(self):
        with self.data_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _document_text(doc):
        return " ".join(
            [doc.get("title", ""), doc.get("category", ""), doc.get("content", ""), *doc.get("keywords", [])]
        )

    @staticmethod
    def _normalize(text):
        return re.sub(r"[^a-z0-9\s]", " ", text.lower()).strip()

    def _expand_query(self, query):
        """Add simple synonyms so related words can retrieve useful documents."""
        synonym_map = {
            "ai": ["artificial intelligence", "machine learning"],
            "ml": ["machine learning", "artificial intelligence"],
            "python": ["programming", "software"],
            "sql": ["database", "query"],
            "database": ["sql", "data"],
            "cloud": ["aws", "azure", "computing"],
            "web": ["website", "application", "frontend", "backend"],
            "career": ["job", "skills", "resume"],
        }
        normalized = self._normalize(query)
        additions = []
        for word, synonyms in synonym_map.items():
            if re.search(rf"\b{re.escape(word)}\b", normalized):
                additions.extend(synonyms)
        return query + " " + " ".join(additions)

    def search(self, query, top_k=5):
        query = query.strip()
        if not query:
            return []

        expanded_query = self._expand_query(query)
        query_vector = self.vectorizer.transform([expanded_query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()

        query_terms = set(self._normalize(query).split())
        results = []
        for index, score in enumerate(scores):
            doc = self.documents[index]
            title_terms = set(self._normalize(doc.get("title", "")).split())
            keyword_terms = set(self._normalize(" ".join(doc.get("keywords", []))).split())
            exact_bonus = 0.08 if query_terms & title_terms else 0
            keyword_bonus = 0.04 if query_terms & keyword_terms else 0
            final_score = float(score) + exact_bonus + keyword_bonus
            if final_score > 0:
                results.append({**doc, "score": final_score})

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]


if __name__ == "__main__":
    engine = MiniSearchEngine()
    for result in engine.search("machine learning"):
        print(f"{result['score']:.3f} - {result['title']}")
