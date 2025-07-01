# Cisco Product Recommendation Engine

This lightweight, **content‑based** recommendation engine suggests Cisco products that best match a user’s needs based on natural‑language descriptions.

## How it works
1. A small seed dataset (`products.csv`) holds basic info on sample Cisco hardware & software.
2. `src/main.py` vectorizes each product **description** with TF‑IDF.
3. When you type a query (e.g., `"secure branch router with voice"`), the script:
   * Embeds your query into the same TF‑IDF space
   * Computes cosine similarity against every product
   * Returns the top *N* most relevant items

> **Why TF‑IDF?**  
> It’s fast, interpretable, and perfect for a proof‑of‑concept. You can swap in more advanced
> models (e.g., sentence transformers) later without changing the interface.

## Quickstart

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Run a search
python src/main.py "high‑density wifi access point" --top 3
```

The console will print the 3 best‑matching Cisco products.

## Project Layout
```
cisco_recommender/
├── products.csv         # sample dataset
├── requirements.txt     # Python dependencies
├── README.md            # this file
└── src/
    └── main.py          # recommendation engine
```

## Extending the Engine
* **Add products:** Append rows to `products.csv`.
* **Better text models:** Replace TF‑IDF with embeddings from `sentence-transformers`.
* **API / UI:** Wrap `recommend_products` in a Flask/FastAPI endpoint or React front‑end.

## Reference
Inspired by the modular project structure in
[`semantic-kernel-nltosql`](https://github.com/zeyduz/semantic-kernel-nltosql), but simplified
for a standalone CLI demo.