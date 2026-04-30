# Experiments
- **Baseline:** Return false for all prior exams (Accuracy:~50%)
- **What worked:** Using `sentence-transformers` and cosine similarity (threshold 0.75). Caching embeddings to prevent timeouts.
- **What failed:** Using external LLMs (timeout).
- **Improvements:** Fine-tune on the provided public JSON.
