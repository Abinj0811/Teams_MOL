# Enrich Chunks with Footers

This script processes all chunks in your Cosmos DB source container and enriches them with footer notes, then saves them to a new container.

## What it does

1. **Reads all chunks** from the source container
2. **Groups by document** (`source_doc_id`)
3. **Extracts footers** from chunks containing `### Footers` sections
4. **Attaches footers to chunks**:
   - If a chunk contains subscripts like `(26)`, `(27)`, it attaches only those specific footers
   - If a chunk has no subscripts, it attaches ALL footers from that document
5. **Saves enriched chunks** to a new container (default: `enriched_chunks`)

## Setup

1. Make sure your `.env` file has:
   ```
   COSMOS_ENDPOINT=...
   COSMOS_KEY=...
   COSMOS_DATABASE=...
   COSMOS_CONTAINER=...  # source container
   ENRICHED_CONTAINER=enriched_chunks  # optional, defaults to this
   ```

2. Run the script:
   ```bash
   python utils/enrich_chunks_with_footers.py
   ```

## Output

The script creates a new container with the same structure as your source container, but each chunk now has a `footers` field in its metadata:

```json
{
  "id": "...",
  "text": "...",
  "source_doc_id": "...",
  "chunk_index": 0,
  "metadata": {...},
  "vector_embedding": [...],
  "footers": {
    "26": "MM report for amount USD500k or more",
    "27": "Email report for amount less than USD500k"
  }
}
```

## Usage in RAG

After running this script, you can update your `search_cosmos_documents` to read from the enriched container instead of the source container. The footers will already be attached, so you can directly use `chunk.get("footers", {})` without needing to extract them from text.

