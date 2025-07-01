import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse, os, sys

def load_products(csv_path: str):
    return pd.read_csv(csv_path)

def build_model(descriptions):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    return vectorizer, tfidf_matrix

def recommend_products(df, vectorizer, tfidf_matrix, query: str, top_n: int = 3):
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx = sims.argsort()[::-1][:top_n]
    return df.iloc[top_idx][['product_id', 'name', 'category', 'description',]]

def cli():
    parser = argparse.ArgumentParser(description="Cisco Product Recommendation Engine")
    parser.add_argument('query', help="Search terms describing what you need (e.g., 'secure branch router')")
    parser.add_argument('--data', default=os.path.join(os.path.dirname(__file__), '..', 'products.csv'),
                        help="Path to CSV file with product data")
    parser.add_argument('--top', type=int, default=3, help="Number of recommendations to return")
    args = parser.parse_args()

    df = load_products(args.data)
    vectorizer, tfidf_matrix = build_model(df['description'])
    recs = recommend_products(df, vectorizer, tfidf_matrix, args.query, args.top)
    print("\nTop Recommendations:")
    for _, row in recs.iterrows():
        print(f"- {row['name']} ({row['category']}) -> {row['description']}")

if __name__ == '__main__':
    cli()