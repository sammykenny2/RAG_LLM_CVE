import sys
from pathlib import Path
# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz
from tqdm.auto import tqdm
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import argparse
import pickle
import chromadb
import json
from datetime import datetime

# Import configuration
from config import (
    DEFAULT_SPEED,
    DEFAULT_EMBEDDING_FORMAT,
    CHUNK_SIZE,
    CHUNK_OVERLAP_RATIO,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_PRECISION,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_PATH,
    CVE_V5_PATH,
    CVE_V4_PATH,
    DEFAULT_SCHEMA
)

# Import CVE lookup utilities
from core.cve_lookup import extract_cve_fields

def get_available_years(base_path):
    """Scan directory for available year folders."""
    import os
    if not os.path.exists(base_path):
        return []

    years = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and item.isdigit() and len(item) == 4:
            years.append(int(item))
    return sorted(years)

def read_pdf(pdf_path):
    """Extract text from each page of the PDF and return as a list of dictionaries."""
    pdf = fitz.open(pdf_path)
    texts = []

    for page in tqdm(pdf):
        text = page.get_text().replace("\n", " ").strip()
        texts.append({"text": text})
    
    return texts

def split_sentences(text):
    """
    Split text into sentences using both English and Chinese punctuation.

    Supports:
    - Chinese punctuation: 。！？
    - English punctuation: . ! ?
    - Mixed content: handles both languages seamlessly

    Args:
        text: Input text to split

    Returns:
        list: List of sentences
    """
    # Pattern matches Chinese and English sentence-ending punctuation
    pattern = r'[。！？.!?]+[\s\n]*'
    sentences = re.split(pattern, text)
    # Filter out empty strings and strip whitespace
    return [s.strip() for s in sentences if s.strip()]

def split(input_list, chunk_size):
    """
    Split a list into chunks of specified size (no overlap).

    Used for CVE data where each description is an atomic unit.

    Args:
        input_list: List to split
        chunk_size: Size of each chunk

    Returns:
        list: List of chunks (non-overlapping)
    """
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]

def split_with_overlap(input_list, chunk_size, overlap_ratio=0.3):
    """
    Split a list into chunks with configurable overlap between consecutive chunks.

    Used for PDF documents to avoid splitting important context across boundaries.

    Args:
        input_list: List to split (e.g., sentences)
        chunk_size: Number of items per chunk (e.g., 10 sentences)
        overlap_ratio: Ratio of overlap between chunks (0.0 = no overlap, 0.3 = 30% overlap)

    Returns:
        list: List of overlapping chunks

    Example:
        chunk_size=10, overlap_ratio=0.3
        - Each chunk has 10 sentences
        - Step size = 10 * (1 - 0.3) = 7 sentences
        - Chunk 0: sentences 0-9
        - Chunk 1: sentences 7-16 (3-sentence overlap with Chunk 0)
        - Chunk 2: sentences 14-23 (3-sentence overlap with Chunk 1)
    """
    if overlap_ratio <= 0:
        # No overlap, use standard split
        return split(input_list, chunk_size)

    # Calculate step size (how far to move forward for each chunk)
    step_size = max(1, int(chunk_size * (1 - overlap_ratio)))

    chunks = []
    for i in range(0, len(input_list), step_size):
        chunk = input_list[i:i + chunk_size]
        if len(chunk) > 0:
            chunks.append(chunk)
        # Stop if we've covered the entire list
        if i + chunk_size >= len(input_list):
            break

    return chunks

def process_cve_file(cve_file_path, keyword_filter=None):
    """Read CVE data from text or JSONL file.

    Args:
        cve_file_path: Path to TXT or JSONL file
        keyword_filter: Optional keyword to filter CVE descriptions (case-insensitive)

    Returns:
        list: List of CVE data dictionaries with sentence_chunk and metadata
    """
    print(f"\n{'='*60}")
    print(f"Processing CVE File")
    print(f"  File: {cve_file_path}")
    if keyword_filter:
        print(f"  Filter: '{keyword_filter}' (case-insensitive)")
    print(f"{'='*60}\n")

    cve_data = []
    file_ext = Path(cve_file_path).suffix.lower()

    if not Path(cve_file_path).exists():
        print(f"❌ File not found: {cve_file_path}")
        return []

    if file_ext == '.jsonl':
        # JSONL format (lossless)
        print("Format: JSONL (lossless, preserves all metadata)")
        with open(cve_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f.readlines(), desc="Reading JSONL"), 1):
                try:
                    record = json.loads(line.strip())
                    cve_text = (
                        f"CVE Number: {record['cve_id']}, "
                        f"Vendor: {record['vendor']}, "
                        f"Product: {record['product']}, "
                        f"Description: {record['description']}"
                    )

                    # Apply keyword filter if specified
                    if keyword_filter and keyword_filter.lower() not in cve_text.lower():
                        continue

                    cve_data.append({
                        "sentence_chunk": cve_text,
                        "source_type": "cve",
                        "source_name": f"CVE_{record['year']}_{record['schema']}",
                        "cve_id": record['cve_id']
                    })
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"⚠️ Skipping line {line_num}: {e}")
                    continue

    elif file_ext == '.txt':
        # TXT format (lossy, need to parse and infer)
        print("Format: TXT (lossy, metadata inferred from content)")
        with open(cve_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split by entries (each starts with "- CVE Number:")
        entries = [e.strip() for e in content.split('- CVE Number:') if e.strip()]

        for entry in tqdm(entries, desc="Parsing TXT"):
            try:
                # Parse: "CVE-YYYY-XXXXX, Vendor: ..., Product: ..., Description: ..."
                parts = entry.split(', ', 3)  # Split into 4 parts max
                if len(parts) < 4:
                    continue

                cve_id = parts[0].strip()
                vendor = parts[1].replace('Vendor:', '').strip()
                product = parts[2].replace('Product:', '').strip()
                description = parts[3].replace('Description:', '').strip()

                # Infer year from CVE ID
                year = cve_id.split('-')[1] if '-' in cve_id else 'unknown'

                cve_text = (
                    f"CVE Number: {cve_id}, "
                    f"Vendor: {vendor}, "
                    f"Product: {product}, "
                    f"Description: {description}"
                )

                # Apply keyword filter if specified
                if keyword_filter and keyword_filter.lower() not in cve_text.lower():
                    continue

                cve_data.append({
                    "sentence_chunk": cve_text,
                    "source_type": "cve",
                    "source_name": f"CVE_{year}_file",  # Schema unknown
                    "cve_id": cve_id
                })

            except Exception as e:
                print(f"⚠️ Error parsing entry: {e}")
                continue

    else:
        print(f"❌ Unsupported file format: {file_ext}. Use .txt or .jsonl")
        return []

    if keyword_filter:
        print(f"\n✅ Extracted {len(cve_data)} CVE descriptions (filtered by '{keyword_filter}')")
    else:
        print(f"\nExtracted {len(cve_data)} CVE descriptions")
    return cve_data

def process_cve_data(years, schema, batch_size, precision, keyword_filter=None):
    """Extract CVE data and return texts with metadata.

    Args:
        years: List of years to process
        schema: 'v5', 'v4', or 'all'
        batch_size: Batch size for processing
        precision: 'float32' or 'float16'
        keyword_filter: Optional keyword filter

    Returns:
        list: CVE data dictionaries
    """
    print(f"\n{'='*60}")
    print(f"Processing CVE Data")
    print(f"  Years: {years}")
    print(f"  Schema: {schema}")
    if keyword_filter:
        print(f"  Filter: '{keyword_filter}' (case-insensitive)")
    print(f"{'='*60}\n")

    # Collect all CVE descriptions from all years
    all_cve_data = []

    for year in years:
        print(f"\n{'='*60}")
        print(f"Processing Year: {year}")
        print(f"{'='*60}")

        # Determine paths based on schema
        paths_to_check = []
        if schema in ['v5', 'all']:
            v5_year_path = CVE_V5_PATH / str(year)
            if v5_year_path.exists():
                paths_to_check.append(('v5', v5_year_path))

        if schema in ['v4', 'all']:
            v4_year_path = CVE_V4_PATH / str(year)
            if v4_year_path.exists():
                paths_to_check.append(('v4', v4_year_path))

        if not paths_to_check:
            print(f"⚠️ No CVE data found for year {year} with schema {schema}")
            continue

        year_cve_count = 0

        for schema_type, year_path in paths_to_check:
            print(f"Scanning {schema_type.upper()} directory: {year_path}")

            for subdir in tqdm(list(year_path.iterdir()), desc=f"Processing {schema_type} ({year})"):
                if not subdir.is_dir():
                    continue

                for json_file in subdir.glob("*.json"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        cve_id, vendor, product, description = extract_cve_fields(data, json_file.stem)

                        cve_text = (
                            f"CVE Number: {cve_id}, "
                            f"Vendor: {vendor}, "
                            f"Product: {product}, "
                            f"Description: {description}"
                        )

                        # Apply keyword filter if specified
                        if keyword_filter and keyword_filter.lower() not in cve_text.lower():
                            continue

                        all_cve_data.append({
                            "sentence_chunk": cve_text,
                            "source_type": "cve",
                            "source_name": f"CVE_{year}_{schema_type}",
                            "cve_id": cve_id
                        })
                        year_cve_count += 1

                    except Exception as e:
                        continue

        print(f"Year {year}: Extracted {year_cve_count} CVE descriptions")

    if keyword_filter:
        print(f"\n{'='*60}")
        print(f"✅ Total: Extracted {len(all_cve_data)} CVE descriptions (filtered by '{keyword_filter}')")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"Total: Extracted {len(all_cve_data)} CVE descriptions")
        print(f"{'='*60}")

    return all_cve_data

def process_pdf(pdf_path, sentence_size, output_path, batch_size, precision, extension):
    """Process PDF to extract text, split into chunks, generate embeddings, and save to file."""
    # Calculate overlap info
    overlap_sentences = int(sentence_size * CHUNK_OVERLAP_RATIO)

    print(f"\n{'='*60}")
    print(f"Configuration:")
    print(f"  Chunk size: {sentence_size} sentences")
    print(f"  Chunk overlap: {CHUNK_OVERLAP_RATIO:.1%} ({overlap_sentences} sentences)")
    print(f"  Batch size: {batch_size}")
    print(f"  Precision: {precision}")
    print(f"  Output format: {extension}")
    print(f"{'='*60}\n")

    texts = read_pdf(pdf_path)

    # Split text into sentences (supports both English and Chinese)
    print("Splitting text into sentences...")
    for item in tqdm(texts):
        item["sentences"] = split_sentences(item["text"])

    # Split sentences into chunks with overlap (for PDF documents)
    print(f"Splitting sentences into chunks (with {CHUNK_OVERLAP_RATIO:.1%} overlap)...")
    for item in tqdm(texts):
        item["sentence_chunks"] = split_with_overlap(
            item["sentences"],
            sentence_size,
            overlap_ratio=CHUNK_OVERLAP_RATIO
        )


    chunks = []
    for item in tqdm(texts):
        for sentence_chunk in item["sentence_chunks"]:
            join_chunk = "".join(sentence_chunk).strip()
            chunks.append({
                "sentence_chunk": join_chunk
            })

    # Convert to DataFrame and filter
    df = pd.DataFrame(chunks)
    dic_chunks = df.to_dict(orient="records")

    # Initialize SentenceTransformer on correct device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing embedding model on {device}...")
    embedding_model = SentenceTransformer(model_name_or_path=EMBEDDING_MODEL_NAME, device=device)

    # Batch encode all chunks (MUCH faster than one-by-one)
    print(f"Generating embeddings for {len(dic_chunks)} chunks...")
    sentence_list = [item["sentence_chunk"] for item in dic_chunks]

    # Encode all at once
    embeddings = embedding_model.encode(
        sentence_list,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Convert to desired precision if needed
    if precision == 'float16':
        embeddings = embeddings.astype(np.float16)
        print(f"  └─ Converted embeddings to float16 (50% memory reduction)")

    # Assign embeddings back to dict
    for idx, item in enumerate(dic_chunks):
        item["embedding"] = embeddings[idx]

    # Save based on extension
    print(f"\nSaving embeddings to {output_path}...")
    if extension == 'csv':
        pd.DataFrame(dic_chunks).to_csv(output_path, index=False)
    elif extension == 'pkl':
        with open(output_path, 'wb') as f:
            pickle.dump(dic_chunks, f)
    elif extension == 'parquet':
        df_output = pd.DataFrame(dic_chunks)
        df_output.to_parquet(output_path, compression='snappy')
    elif extension == 'chroma':
        # Chroma vector database (persistent, no server required)
        client = chromadb.PersistentClient(path=output_path)

        # Delete existing collection if it exists (to allow re-creation)
        try:
            client.delete_collection("cve_embeddings")
        except:
            pass

        # Create new collection
        collection = client.create_collection(
            name="cve_embeddings",
            metadata={"description": "CVE embeddings for RAG system"}
        )

        # Prepare data for Chroma
        ids = [f"chunk_{i}" for i in range(len(dic_chunks))]
        embeddings_list = [item["embedding"].tolist() for item in dic_chunks]
        documents = [item["sentence_chunk"] for item in dic_chunks]

        # Prepare metadata for PDF
        pdf_name = Path(pdf_path).name
        metadatas = [
            {
                "source_type": "pdf",
                "source_name": pdf_name,
                "added_date": datetime.now().isoformat(),
                "chunk_index": i,
                "precision": precision
            }
            for i in range(len(dic_chunks))
        ]

        # Add to collection in batches (Chroma has batch size limits)
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            collection.add(
                ids=ids[i:i+batch_size],
                embeddings=embeddings_list[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )

        print(f"  └─ Stored {len(dic_chunks)} embeddings in Chroma database")

    print(f"✅ Generated: {output_path}")
    print(f"💡 To use this with theRag.py, run:")
    print(f"   python theRag.py --extension={extension}")


# Main execution
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Generate embeddings from PDF for RAG system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_embeddings.py                                    # Default (fast, pkl)
  python build_embeddings.py --speed=normal --extension=csv     # High quality, CSV
  python build_embeddings.py --speed=fastest --extension=parquet # Maximum speed, smallest file
  python build_embeddings.py --filter=snmp                      # Only include CVEs containing 'snmp'
  python build_embeddings.py --filter=Windows --extension=chroma # Filter for Windows CVEs in Chroma DB
        """
    )
    parser.add_argument(
        '--speed',
        type=str,
        choices=['normal', 'fast', 'fastest'],
        default=DEFAULT_SPEED,
        help=f'Processing speed: normal (float32), fast (float16, recommended), fastest (float16 + larger chunks) (default: {DEFAULT_SPEED})'
    )
    parser.add_argument(
        '--extension',
        type=str,
        choices=['csv', 'pkl', 'parquet', 'chroma'],
        default=DEFAULT_EMBEDDING_FORMAT,
        help=f'Output format: csv (text), pkl (pickle), parquet (optimal), chroma (vector database) (default: {DEFAULT_EMBEDDING_FORMAT})'
    )
    parser.add_argument(
        '--filter',
        type=str,
        default=None,
        help='Filter CVEs by keyword (case-insensitive). Only CVEs containing this keyword will be included. Example: --filter=snmp'
    )
    args = parser.parse_args()

    # Configure based on speed level
    if args.speed == 'normal':
        SENTENCE_SIZE = CHUNK_SIZE
        BATCH_SIZE = 32
        PRECISION = 'float32'
        speed_desc = "NORMAL (baseline quality, float32)"
    elif args.speed == 'fast':
        SENTENCE_SIZE = CHUNK_SIZE
        BATCH_SIZE = EMBEDDING_BATCH_SIZE
        PRECISION = EMBEDDING_PRECISION
        speed_desc = f"FAST (recommended, float16)"
    else:  # fastest
        SENTENCE_SIZE = CHUNK_SIZE * 2
        BATCH_SIZE = EMBEDDING_BATCH_SIZE * 2
        PRECISION = EMBEDDING_PRECISION
        speed_desc = f"FASTEST (aggressive optimization)"

    # Print header
    print(f"\n{'='*60}")
    print(f"Build Embeddings - Create New Knowledge Base")
    print(f"{'='*60}\n")

    # Ask user what to process
    print("What would you like to build embeddings from?")
    print("1. PDF file")
    print("2. CVE data by year (from JSON feeds)")
    print("3. CVE text/JSONL file (from extract_cve.py)")
    choice = input("\nEnter your choice (1-3): ").strip()

    # Construct output path
    output_path = f"{EMBEDDING_PATH}.{args.extension}"

    # Get data based on choice
    if choice == '1':
        # PDF mode
        source_type = 'pdf'
        pdf_path = input("\nEnter PDF file path: ").strip()
        year = None
        schema = None
        cve_file = None

    elif choice == '2':
        # CVE JSON mode
        source_type = 'cve'
        pdf_path = None
        cve_file = None
        year_input = input("\nEnter year(s) for CVE data (e.g., 2025, 2023-2025, or 'all'): ").strip()

        # Determine schema first (needed for 'all' year detection)
        print("\nSelect CVE schema:")
        print("1. v5 only (fastest)")
        print("2. v4 only")
        print("3. v5 with v4 fallback (default)")
        schema_choice = input("Enter your choice (1-3, default=3): ").strip() or '3'
        schema_map = {'1': 'v5', '2': 'v4', '3': 'all'}
        schema = schema_map.get(schema_choice, 'all')

        # Parse year input
        if year_input.lower() == 'all':
            # Scan directories based on schema selection
            v5_years = get_available_years(str(CVE_V5_PATH)) if schema in ['v5', 'all'] else []
            v4_years = get_available_years(str(CVE_V4_PATH)) if schema in ['v4', 'all'] else []
            years = sorted(set(v5_years + v4_years))
            if not years:
                print(f"❌ No year directories found in CVE feeds for schema '{schema}'")
                sys.exit(1)
            print(f"Processing all available years: {years}")
        elif '-' in year_input:
            # Range format: 2023-2025
            try:
                start_year, end_year = year_input.split('-')
                start_year = int(start_year.strip())
                end_year = int(end_year.strip())
                if start_year > end_year:
                    print(f"❌ Invalid range: start year {start_year} > end year {end_year}")
                    sys.exit(1)
                years = list(range(start_year, end_year + 1))
                print(f"Processing years {start_year}-{end_year}: {years}")
            except ValueError:
                print(f"❌ Invalid year range format: {year_input}. Use format: 2023-2025")
                sys.exit(1)
        elif ',' in year_input:
            # Comma-separated format: 2023,2024,2025
            try:
                years = [int(y.strip()) for y in year_input.split(',')]
                print(f"Processing years: {years}")
            except ValueError:
                print(f"❌ Invalid year format: {year_input}. Use single year (2025), range (2023-2025), comma-separated (2023,2024,2025), or 'all'")
                sys.exit(1)
        else:
            # Single year
            try:
                years = [int(year_input)]
            except ValueError:
                print(f"❌ Invalid year format: {year_input}. Use single year (2025), range (2023-2025), comma-separated (2023,2024,2025), or 'all'")
                sys.exit(1)

        # Ask for filter keyword
        filter_input = input("\nFilter CVEs by keyword (leave empty for all): ").strip()
        if filter_input:
            args.filter = filter_input
        # else keep args.filter from command line or default (None)

    elif choice == '3':
        # CVE file mode (TXT or JSONL)
        source_type = 'cve_file'
        pdf_path = None
        year = None
        schema = None
        cve_file = input("\nEnter CVE file path (.txt or .jsonl): ").strip()

    else:
        print(f"❌ Invalid choice: {choice}")
        sys.exit(1)

    # Display configuration
    print(f"\n{'='*60}")
    print(f"Configuration:")
    print(f"{'='*60}")
    print(f"Source:      {source_type}")
    if source_type == 'pdf':
        print(f"File:        {pdf_path}")
    elif source_type == 'cve':
        print(f"Years:       {years}")
        print(f"Schema:      {schema}")
        if args.filter:
            print(f"Filter:      '{args.filter}' (case-insensitive)")
    elif source_type == 'cve_file':
        print(f"File:        {cve_file}")
        if args.filter:
            print(f"Filter:      '{args.filter}' (case-insensitive)")
    print(f"Speed:       {speed_desc}")
    print(f"Chunk size:  {SENTENCE_SIZE} sentences")
    print(f"Batch size:  {BATCH_SIZE}")
    print(f"Precision:   {PRECISION}")
    print(f"Format:      {args.extension}")
    print(f"Output:      {output_path}")
    print(f"{'='*60}\n")

    # Process based on choice
    if source_type == 'pdf':
        process_pdf(pdf_path, SENTENCE_SIZE, output_path, BATCH_SIZE, PRECISION, args.extension)

    elif source_type == 'cve_file':
        # Process CVE from file
        dic_chunks = process_cve_file(cve_file, keyword_filter=args.filter)

        if not dic_chunks:
            print("❌ No CVE data found to process")
            sys.exit(1)

        # Initialize SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nInitializing embedding model on {device}...")
        embedding_model = SentenceTransformer(model_name_or_path=EMBEDDING_MODEL_NAME, device=device)

        # Generate embeddings
        print(f"Generating embeddings for {len(dic_chunks)} CVE descriptions...")
        sentence_list = [item["sentence_chunk"] for item in dic_chunks]

        embeddings = embedding_model.encode(
            sentence_list,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        if PRECISION == 'float16':
            embeddings = embeddings.astype(np.float16)
            print(f"  └─ Converted embeddings to float16 (50% memory reduction)")

        # Assign embeddings
        for idx, item in enumerate(dic_chunks):
            item["embedding"] = embeddings[idx]

        # Save based on extension
        print(f"\nSaving embeddings to {output_path}...")
        if args.extension == 'csv':
            pd.DataFrame(dic_chunks).to_csv(output_path, index=False)
        elif args.extension == 'pkl':
            with open(output_path, 'wb') as f:
                pickle.dump(dic_chunks, f)
        elif args.extension == 'parquet':
            df_output = pd.DataFrame(dic_chunks)
            df_output.to_parquet(output_path, compression='snappy')
        elif args.extension == 'chroma':
            client = chromadb.PersistentClient(path=output_path)
            try:
                client.delete_collection("cve_embeddings")
            except:
                pass
            collection = client.create_collection(
                name="cve_embeddings",
                metadata={"description": "CVE embeddings for RAG system"}
            )
            ids = [f"chunk_{i}" for i in range(len(dic_chunks))]
            embeddings_list = [item["embedding"].tolist() for item in dic_chunks]
            documents = [item["sentence_chunk"] for item in dic_chunks]

            # Add metadata
            metadatas = [
                {
                    "source_type": item.get("source_type", "cve"),
                    "source_name": item.get("source_name", "CVE_file"),
                    "cve_id": item.get("cve_id", ""),
                }
                for item in dic_chunks
            ]

            batch_size = 5000
            for i in range(0, len(ids), batch_size):
                collection.add(
                    ids=ids[i:i+batch_size],
                    embeddings=embeddings_list[i:i+batch_size],
                    documents=documents[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size]
                )
            print(f"  └─ Stored {len(dic_chunks)} embeddings in Chroma database")

        print(f"✅ Generated: {output_path}")
        print(f"\n💡 To use this with validate_report.py, run:")
        print(f"   python validate_report.py --extension={args.extension}")

    else:  # CVE JSON mode
        # Process CVE data
        dic_chunks = process_cve_data(years, schema, BATCH_SIZE, PRECISION, keyword_filter=args.filter)

        if not dic_chunks:
            print("❌ No CVE data found to process")
            sys.exit(1)

        # Initialize SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nInitializing embedding model on {device}...")
        embedding_model = SentenceTransformer(model_name_or_path=EMBEDDING_MODEL_NAME, device=device)

        # Generate embeddings
        print(f"Generating embeddings for {len(dic_chunks)} CVE descriptions...")
        sentence_list = [item["sentence_chunk"] for item in dic_chunks]

        embeddings = embedding_model.encode(
            sentence_list,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        if PRECISION == 'float16':
            embeddings = embeddings.astype(np.float16)
            print(f"  └─ Converted embeddings to float16 (50% memory reduction)")

        # Assign embeddings
        for idx, item in enumerate(dic_chunks):
            item["embedding"] = embeddings[idx]

        # Save based on extension
        print(f"\nSaving embeddings to {output_path}...")
        if args.extension == 'csv':
            pd.DataFrame(dic_chunks).to_csv(output_path, index=False)
        elif args.extension == 'pkl':
            with open(output_path, 'wb') as f:
                pickle.dump(dic_chunks, f)
        elif args.extension == 'parquet':
            df_output = pd.DataFrame(dic_chunks)
            df_output.to_parquet(output_path, compression='snappy')
        elif args.extension == 'chroma':
            client = chromadb.PersistentClient(path=output_path)
            try:
                client.delete_collection("cve_embeddings")
            except:
                pass
            collection = client.create_collection(
                name="cve_embeddings",
                metadata={"description": "CVE embeddings for RAG system"}
            )
            ids = [f"chunk_{i}" for i in range(len(dic_chunks))]
            embeddings_list = [item["embedding"].tolist() for item in dic_chunks]
            documents = [item["sentence_chunk"] for item in dic_chunks]

            # Add metadata
            metadatas = [
                {
                    "source_type": item.get("source_type", "cve"),
                    "source_name": item.get("source_name", "CVE_unknown"),
                    "cve_id": item.get("cve_id", ""),
                }
                for item in dic_chunks
            ]

            batch_size = 5000
            for i in range(0, len(ids), batch_size):
                collection.add(
                    ids=ids[i:i+batch_size],
                    embeddings=embeddings_list[i:i+batch_size],
                    documents=documents[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size]
                )
            print(f"  └─ Stored {len(dic_chunks)} embeddings in Chroma database")

        print(f"✅ Generated: {output_path}")
        print(f"\n💡 To use this with validate_report.py, run:")
        print(f"   python validate_report.py --extension={args.extension}")
