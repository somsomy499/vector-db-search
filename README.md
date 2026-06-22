# Vector DB Search 🔎

GPU-accelerated vector database with HNSW indexing and distributed clustering.

## Performance

| Dataset | QPS (GPU) | QPS (CPU) | Recall@10 | Memory |
|---------|-----------|-----------|-----------|--------|
| SIFT-1M | 45,000 | 2,100 | 0.98 | 2.1GB |
| GIST-1M | 28,000 | 1,400 | 0.97 | 4.8GB |
| OpenAI-10M | 12,000 | 600 | 0.96 | 15GB |

## Quick Start

```python
from vector_db import VectorDB

db = VectorDB(dim=1536, index_type="hnsw")
db.insert(embeddings, metadata=metadata)
results = db.search(query_embedding, top_k=10)
```

## License

MIT