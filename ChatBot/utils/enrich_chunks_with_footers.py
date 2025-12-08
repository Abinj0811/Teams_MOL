"""
Standalone script to enrich Cosmos DB chunks with footer notes.

This script:
1. Reads all chunks from the source container
2. Groups chunks by source_doc_id
3. Extracts footer notes from chunks containing "### Footers"
4. Attaches footers to chunks that reference them via subscripts (e.g., (26), (27))
5. If no subscripts found, attaches all footers to all chunks in the document
6. Saves enriched chunks to a new container
"""

import os
import re
import json
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from dotenv import load_dotenv
from azure.cosmos import CosmosClient, PartitionKey, exceptions

load_dotenv()

# Footer line pattern: "26 MM report for amount USD500k or more"
FOOTER_LINE_RE = re.compile(r"^\s*(\d+)\s+(.*\S)\s*$")

# Subscript pattern: (26), (27), etc.
SUBSCRIPT_RE = re.compile(r"\((\d+)\)")


def extract_footers_from_text(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Extract footer lines from text containing "### Footers" section.
    Returns: (cleaned_text_without_footers, footer_dict)
    """
    lines = text.splitlines()
    cleaned_lines = []
    footer_lines: List[str] = []
    in_footer = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("### Footers"):
            in_footer = True
            continue

        if in_footer:
            # End footer block when hitting another section header
            if stripped.startswith("### ") or stripped.startswith("## "):
                in_footer = False
                cleaned_lines.append(line)
                continue

            if stripped:
                footer_lines.append(stripped)
            continue

        cleaned_lines.append(line)

    # Parse footer lines into dict
    footer_map: Dict[str, str] = {}
    for fl in footer_lines:
        m = FOOTER_LINE_RE.match(fl)
        if m:
            num, txt = m.group(1), m.group(2).strip()
            footer_map[num] = txt

    cleaned_text = "\n".join(cleaned_lines)
    return cleaned_text, footer_map


def find_subscripts_in_text(text: str) -> Set[str]:
    """Find all subscript numbers like (26), (27) in the text."""
    matches = SUBSCRIPT_RE.findall(text)
    return set(matches)


def enrich_chunks_with_footers(
    source_container,
    target_container_name: str,
    target_db,
    partition_key_path: str = "/id"
):
    """
    Main function to process all chunks and enrich them with footers.
    
    Args:
        source_container: Source Cosmos container to read from
        target_container_name: Name of new container to write enriched chunks
        target_db: Cosmos database object
        partition_key_path: Partition key path for target container
    """
    print("=" * 60)
    print("Starting chunk enrichment with footers...")
    print("=" * 60)

    # Create target container if it doesn't exist
    try:
        target_container = target_db.create_container_if_not_exists(
            id=target_container_name,
            partition_key=PartitionKey(path=partition_key_path)
        )
        print(f"✅ Target container '{target_container_name}' ready.")
    except Exception as e:
        print(f"❌ Error creating target container: {e}")
        return

    # Step 1: Read all chunks from source container
    print("\n📖 Reading all chunks from source container...")
    sql = "SELECT c.id, c.text, c.source_doc_id, c.chunk_index, c.metadata, c.vector_embedding FROM c"
    
    all_chunks = list(source_container.query_items(
        query=sql,
        enable_cross_partition_query=True
    ))
    
    print(f"✅ Found {len(all_chunks)} total chunks.")

    # Step 2: Group chunks by source_doc_id
    print("\n📦 Grouping chunks by document...")
    chunks_by_doc: Dict[str, List[Dict]] = defaultdict(list)
    for chunk in all_chunks:
        doc_id = chunk.get("source_doc_id", "unknown")
        chunks_by_doc[doc_id].append(chunk)

    print(f"✅ Found {len(chunks_by_doc)} unique documents.")

    # Step 3: Process each document
    total_enriched = 0
    total_skipped = 0

    for doc_id, chunks in chunks_by_doc.items():
        print(f"\n📄 Processing document: {doc_id} ({len(chunks)} chunks)")

        # Sort chunks by chunk_index
        chunks_sorted = sorted(chunks, key=lambda x: x.get("chunk_index", 0))

        # Step 3a: Extract footers from all chunks in this document
        all_footers: Dict[str, str] = {}
        footer_chunk_indices: Set[int] = set()

        for idx, chunk in enumerate(chunks_sorted):
            text = chunk.get("text", "")
            if "### Footers" in text:
                footer_chunk_indices.add(chunk.get("chunk_index", idx))
                _, footer_map = extract_footers_from_text(text)
                all_footers.update(footer_map)

        if all_footers:
            print(f"  ✅ Found {len(all_footers)} footer definitions: {sorted(all_footers.keys())}")
        else:
            print(f"  ⚠️  No footers found in this document.")

        # Step 3b: For each chunk, check for subscripts and attach footers
        for chunk in chunks_sorted:
            chunk_text = chunk.get("text", "")
            chunk_index = chunk.get("chunk_index", 0)

            # Find subscripts in this chunk
            subscripts = find_subscripts_in_text(chunk_text)

            footers_to_attach = {}

            if subscripts:
                missing = []
                for sub in subscripts:
                    if sub in all_footers:
                        footers_to_attach[sub] = all_footers[sub]
                    else:
                        missing.append(sub)

                if missing:
                    # ❗ Debugging: subscripts found but footer missing
                    print(f"⚠️  Chunk {chunk_index}: Subscripts found {missing} but no footer definitions exist for them.")
                    snippet = chunk_text[:180].replace("\n", " ")
                    print(f"    Text snippet: {snippet}...")
                print(
                    f"    Chunk {chunk_index}: Found subscripts {sorted(subscripts)} → attaching {len(footers_to_attach)} footers"
                )

            else:
                print(f"    Chunk {chunk_index}: No subscripts found → attaching 0 footers")

            # --- Build enriched metadata ---
            # --- Build enriched metadata ---
            metadata = chunk.get("metadata", {}).copy()
            metadata["footers"] = footers_to_attach if footers_to_attach else {}

            # --- Build enriched chunk ---
            enriched_chunk = { **chunk }     # clone entire original document
            enriched_chunk["metadata"] = metadata
            # Enriched chunk

            target_container.upsert_item(enriched_chunk)



def main():
    """Main entry point."""
    # Get Cosmos DB credentials from environment
    cosmos_endpoint = os.getenv("COSMOS_ENDPOINT")
    cosmos_key = os.getenv("COSMOS_KEY")
    db_name = os.getenv("COSMOS_DATABASE")
    source_container_name = os.getenv("COSMOS_CONTAINER")
    target_container_name = os.getenv("ENRICHED_CONTAINER", "enriched_chunks")

    if not all([cosmos_endpoint, cosmos_key, db_name, source_container_name]):
        print("❌ Missing required environment variables:")
        print("   - COSMOS_ENDPOINT")
        print("   - COSMOS_KEY")
        print("   - COSMOS_DATABASE")
        print("   - COSMOS_CONTAINER")
        return

    try:
        # Connect to Cosmos DB
        print("🔌 Connecting to Cosmos DB...")
        client = CosmosClient(url=cosmos_endpoint, credential=cosmos_key)
        db = client.get_database_client(db_name)
        source_container = db.get_container_client(source_container_name)

        print(f"✅ Connected to database '{db_name}'")
        print(f"✅ Source container: '{source_container_name}'")

        # Run enrichment
        enrich_chunks_with_footers(
            source_container=source_container,
            target_container_name=target_container_name,
            target_db=db,
            partition_key_path="/id"
        )

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

