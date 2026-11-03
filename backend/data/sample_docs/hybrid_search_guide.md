# Modern Hybrid Search Architecture

Hybrid search integrates dense vector semantic retrieval with sparse inverted-index BM25 retrieval. 

### Why Hybrid Search is Superior
1. **Out-of-Vocabulary (OOV) Terms**: Dense vectors struggle with rare product SKUs, exact hashes, and brand identifiers. BM25 easily indexes and ranks these tokens.
2. **Semantic Generalization**: BM25 fails when users describe a concept using synonyms or non-identical vocabulary. Dense vector embeddings map queries into shared continuous vector space.
3. **Weight Calibration**: The alpha weighting parameter determines the balance between dense and sparse scores. An alpha of 1.0 corresponds to pure dense vector search, while an alpha of 0.0 corresponds to pure BM25 search. In production benchmarks, an alpha between 0.6 and 0.7 yields peak Mean Reciprocal Rank (MRR).
