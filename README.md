# Mini Search Engine with Intelligent Engine

## Problem Statement
Basic keyword search can miss useful documents when the user's wording is slightly different from the wording used in a document. This project builds a small search engine that ranks documents by relevance and adds simple intelligence to improve related-query matching.

## Approach
1. Store a small collection of documents in JSON.
2. Combine each document's title, category, content and keywords into searchable text.
3. Convert documents into TF-IDF vectors using unigrams and bigrams.
4. Convert the user's query into the same vector space and calculate cosine similarity.
5. Expand a few common terms with simple synonyms, such as `ai` → `artificial intelligence` and `sql` → `database`.
6. Add small title and keyword relevance bonuses.
7. Sort the results by the final relevance score and show the best matches in a Streamlit interface.

## Key Insights
- TF-IDF is a simple and effective baseline for document search.
- Cosine similarity makes it possible to rank documents rather than only return exact keyword matches.
- Query expansion helps related terms retrieve useful results.
- Title and keyword matches can provide extra ranking signals.
- The project is intentionally small so the complete search pipeline is easy to understand and extend.

## Tech Stack
- Python
- Streamlit
- scikit-learn
- NumPy
- JSON
- TF-IDF
- Cosine Similarity

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/JaiperumalJayaraman/mini-search-engine-with-intelligent-engine.git
cd mini-search-engine-with-intelligent-engine
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

### 4. Open the local URL

Streamlit will display a local address, normally:

```text
http://localhost:8501
```

Try searches such as:
- `machine learning`
- `AI`
- `Python`
- `SQL database`
- `cloud computing`
- `software career`

## Project Structure

```text
mini-search-engine-with-intelligent-engine/
├── app.py
├── search_engine.py
├── requirements.txt
├── README.md
└── data/
    └── documents.json
```

## Limitations
This is a learning-focused mini search engine. It uses TF-IDF rather than a large language model or production-scale vector database, and the synonym dictionary is intentionally small.

## Future Improvements
- Add more documents.
- Add fuzzy matching.
- Replace the simple synonym map with embeddings.
- Add filters by category.
- Add search history and analytics.
- Add a larger dataset and evaluation metrics such as Precision@K.
