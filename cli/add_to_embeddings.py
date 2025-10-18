"""
Incremental embedding addition tool.
Add PDFs and CVE data to existing Chroma database.

Interactive mode - just run the script and follow prompts:
    python add_to_embeddings.py
"""

import sys
from pathlib import Path
# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from tqdm.auto import tqdm
import pandas as pd
import re
import numpy as np

# Import core modules
from core.embeddings import EmbeddingModel
from core.chroma_manager import ChromaManager
from core.pdf_processor import PDFProcessor
from core.cve_lookup import extract_cve_numbers, format_second_set, extract_cve_fields

# Import configuration
from config import (
    EMBEDDING_PATH,
    CVE_V5_PATH,
    CVE_V4_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP_RATIO,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_PRECISION,
    DEFAULT_SCHEMA
)

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

def split_list(input_list, chunk_size):
    """
    Split a list into chunks of specified size (no overlap).

    Used for CVE data where each description is an atomic unit.
    """
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]

def split_list_with_overlap(input_list, chunk_size, overlap_ratio=0.3):
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
        return split_list(input_list, chunk_size)

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

def process_cve_file(
    cve_file_path: str,
    chroma_manager: ChromaManager,
    embedding_model: EmbeddingModel,
    batch_size: int,
    precision: str,
    keyword_filter: str = None
):
    """
    Read CVE data from text or JSONL file and add to Chroma database.

    Args:
        cve_file_path: Path to TXT or JSONL file
        chroma_manager: ChromaManager instance
        embedding_model: EmbeddingModel instance
        batch_size: Batch size for embedding generation
        precision: 'float32' or 'float16'
        keyword_filter: Optional keyword to filter CVE descriptions (case-insensitive)

    Returns:
        int: Number of CVEs added
    """
    import json

    print(f"\n{'='*60}")
    print(f"Processing CVE File")
    print(f"  File: {cve_file_path}")
    if keyword_filter:
        print(f"  Filter: '{keyword_filter}' (case-insensitive)")
    print(f"{'='*60}\n")

    cve_file_path = Path(cve_file_path)
    file_ext = cve_file_path.suffix.lower()

    if not cve_file_path.exists():
        print(f"❌ File not found: {cve_file_path}")
        return 0

    cve_texts = []
    cve_metadata = []

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

                    cve_texts.append(cve_text)
                    cve_metadata.append({
                        'source_type': 'cve',
                        'source_name': f"CVE_{record['year']}_{record['schema']}",
                        'cve_id': record['cve_id'],
                        'added_date': datetime.now().isoformat(),
                        'chunk_index': len(cve_texts) - 1,
                        'precision': precision
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

                cve_texts.append(cve_text)
                cve_metadata.append({
                    'source_type': 'cve',
                    'source_name': f"CVE_{year}_file",  # Schema unknown
                    'cve_id': cve_id,
                    'added_date': datetime.now().isoformat(),
                    'chunk_index': len(cve_texts) - 1,
                    'precision': precision
                })

            except Exception as e:
                print(f"⚠️ Error parsing entry: {e}")
                continue

    else:
        print(f"❌ Unsupported file format: {file_ext}. Use .txt or .jsonl")
        return 0

    if not cve_texts:
        print(f"⚠️ No CVE descriptions extracted")
        return 0

    print(f"\nExtracted {len(cve_texts)} CVE descriptions")

    # Generate embeddings
    print(f"Generating embeddings (batch_size={batch_size})...")
    embeddings = embedding_model.encode(
        cve_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        precision=precision
    )

    # Convert to list format for Chroma
    embeddings_list = [emb.tolist() for emb in embeddings]

    # Add to Chroma
    print(f"Adding {len(cve_texts)} CVE descriptions to Chroma database...")
    n_added = chroma_manager.add_documents(
        texts=cve_texts,
        embeddings=embeddings_list,
        metadata=cve_metadata
    )

    print(f"✅ Added {n_added} CVE descriptions")
    return n_added

def process_pdf_files(
    pdf_files: list,
    chroma_manager: ChromaManager,
    embedding_model: EmbeddingModel,
    chunk_size: int,
    batch_size: int,
    precision: str
):
    """
    Process PDF files and add to Chroma database.

    Args:
        pdf_files: List of PDF file paths
        chroma_manager: ChromaManager instance
        embedding_model: EmbeddingModel instance
        chunk_size: Number of sentences per chunk
        batch_size: Batch size for embedding generation
        precision: 'float32' or 'float16'
    """
    # Initialize PDF processor
    pdf_processor = PDFProcessor(periodic_gc=True)

    total_added = 0

    for pdf_path in pdf_files:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            print(f"⚠️ File not found: {pdf_path}, skipping...")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {pdf_path.name}")
        print(f"{'='*60}")

        # Extract text
        print("Extracting text from PDF...")
        all_text = pdf_processor.extract_text(pdf_path)

        # Split into sentences (supports both English and Chinese)
        print("Splitting into sentences...")
        sentences = split_sentences(all_text)

        # Create chunks with overlap (for PDF documents)
        overlap_sentences = int(chunk_size * CHUNK_OVERLAP_RATIO)
        print(f"Creating chunks (size={chunk_size}, overlap={CHUNK_OVERLAP_RATIO:.1%} or {overlap_sentences} sentences)...")
        sentence_chunks = split_list_with_overlap(
            sentences,
            chunk_size,
            overlap_ratio=CHUNK_OVERLAP_RATIO
        )

        # Prepare text chunks
        text_chunks = []
        for chunk in sentence_chunks:
            joined_chunk = "".join(chunk).strip()
            if joined_chunk:  # Skip empty chunks
                text_chunks.append(joined_chunk)

        print(f"Generated {len(text_chunks)} chunks")

        # Generate embeddings
        print(f"Generating embeddings (batch_size={batch_size})...")
        embeddings = embedding_model.encode(
            text_chunks,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            precision=precision
        )

        # Convert to list format for Chroma
        embeddings_list = [emb.tolist() for emb in embeddings]

        # Prepare metadata
        metadata = [
            {
                'source_type': 'pdf',
                'source_name': pdf_path.name,
                'added_date': datetime.now().isoformat(),
                'chunk_index': i,
                'precision': precision
            }
            for i in range(len(text_chunks))
        ]

        # Add to Chroma
        print(f"Adding {len(text_chunks)} chunks to Chroma database...")
        n_added = chroma_manager.add_documents(
            texts=text_chunks,
            embeddings=embeddings_list,
            metadata=metadata
        )

        total_added += n_added
        print(f"✅ Added {n_added} chunks from {pdf_path.name}")

    return total_added

def process_cve_data(
    year: int,
    month_range: tuple,
    chroma_manager: ChromaManager,
    embedding_model: EmbeddingModel,
    batch_size: int,
    precision: str,
    schema: str,
    keyword_filter: str = None
):
    """
    Process CVE data and add to Chroma database.

    Args:
        year: Year to process (e.g., 2025)
        month_range: Tuple of (start_month, end_month) or None for all
        chroma_manager: ChromaManager instance
        embedding_model: EmbeddingModel instance
        batch_size: Batch size for embedding generation
        precision: 'float32' or 'float16'
        schema: 'v5', 'v4', or 'all'
        keyword_filter: Optional keyword to filter CVE descriptions (case-insensitive)
    """
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
        print(f"❌ No CVE data found for year {year} with schema {schema}")
        return 0

    # Collect all CVE text descriptions
    cve_texts = []
    cve_metadata = []

    print(f"\n{'='*60}")
    print(f"Processing CVE data for year {year}")
    if month_range:
        print(f"Month range: {month_range[0]}-{month_range[1]}")
    print(f"Schema: {schema}")
    if keyword_filter:
        print(f"Filter: '{keyword_filter}' (case-insensitive)")
    print(f"{'='*60}\n")

    for schema_type, year_path in paths_to_check:
        print(f"Scanning {schema_type.upper()} directory: {year_path}")

        # Iterate through all subdirectories
        for subdir in tqdm(list(year_path.iterdir()), desc=f"Processing {schema_type}"):
            if not subdir.is_dir():
                continue

            # Process JSON files in subdirectory
            for json_file in subdir.glob("*.json"):
                try:
                    import json
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Extract CVE information
                    cve_id, vendor, product, description = extract_cve_fields(data, json_file.stem)

                    # Apply month filter if specified
                    if month_range:
                        # Extract month from CVE metadata (if available)
                        # For simplicity, we'll skip month filtering for now
                        # (CVE JSON doesn't always have consistent date fields)
                        pass

                    # Format CVE description
                    cve_text = (
                        f"CVE Number: {cve_id}, "
                        f"Vendor: {vendor}, "
                        f"Product: {product}, "
                        f"Description: {description}"
                    )

                    # Apply keyword filter if specified
                    if keyword_filter and keyword_filter.lower() not in cve_text.lower():
                        continue

                    cve_texts.append(cve_text)
                    cve_metadata.append({
                        'source_type': 'cve',
                        'source_name': f"CVE_{year}_{schema_type}",
                        'cve_id': cve_id,
                        'added_date': datetime.now().isoformat(),
                        'chunk_index': len(cve_texts) - 1,
                        'precision': precision
                    })

                except Exception as e:
                    if '--verbose' in sys.argv:
                        print(f"⚠️ Error processing {json_file.name}: {e}")
                    continue

    if not cve_texts:
        print(f"⚠️ No CVE descriptions extracted")
        return 0

    print(f"\nExtracted {len(cve_texts)} CVE descriptions")

    # Generate embeddings
    print(f"Generating embeddings (batch_size={batch_size})...")
    embeddings = embedding_model.encode(
        cve_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        precision=precision
    )

    # Convert to list format for Chroma
    embeddings_list = [emb.tolist() for emb in embeddings]

    # Add to Chroma
    print(f"Adding {len(cve_texts)} CVE descriptions to Chroma database...")
    n_added = chroma_manager.add_documents(
        texts=cve_texts,
        embeddings=embeddings_list,
        metadata=cve_metadata
    )

    print(f"✅ Added {n_added} CVE descriptions")
    return n_added

def main():
    print(f"\n{'='*60}")
    print(f"Add to Embeddings - Update Existing Knowledge Base")
    print(f"{'='*60}\n")

    # Interactive mode: ask user what to add
    print("What would you like to add to the knowledge base?")
    print("1. PDF file(s)")
    print("2. CVE data by year (from JSON feeds)")
    print("3. CVE text/JSONL file (from extract_cve.py)")
    choice = input("\nEnter your choice (1-3): ").strip()

    # Determine source type
    if choice == '1':
        source_type = 'pdf'
        # Ask for PDF file path(s)
        pdf_input = input("\nEnter PDF file path(s) (comma-separated for multiple): ").strip()
        pdf_files = [f.strip() for f in pdf_input.split(',')]
        year = None
        month_range = None
        schema = DEFAULT_SCHEMA
        cve_file = None
        filter_keyword = None
        replace_existing = False  # Not applicable for PDFs
    elif choice == '2':
        source_type = 'cve'
        # Ask for year
        year_input = input("\nEnter year for CVE data (e.g., 2025): ").strip()
        try:
            year = int(year_input)
        except ValueError:
            print(f"❌ Invalid year: {year_input}")
            sys.exit(1)

        # Ask for schema
        print("\nSelect CVE schema:")
        print("1. v5 only (fastest)")
        print("2. v4 only")
        print("3. v5 with v4 fallback (default)")
        schema_choice = input("Enter your choice (1-3, default=3): ").strip() or '3'
        schema_map = {'1': 'v5', '2': 'v4', '3': 'all'}
        schema = schema_map.get(schema_choice, 'all')

        # Ask if should replace existing data
        replace_input = input("\nReplace existing year data? (Y/n): ").strip().lower()
        replace_existing = replace_input != 'n'

        # Ask for filter keyword
        filter_keyword = input("\nFilter CVEs by keyword (leave empty for all): ").strip() or None

        pdf_files = None
        month_range = None
        cve_file = None
    elif choice == '3':
        source_type = 'cve_file'
        # Ask for CVE file path
        cve_file = input("\nEnter CVE file path (.txt or .jsonl): ").strip()
        pdf_files = None
        year = None
        month_range = None
        schema = None
        filter_keyword = None  # File should be pre-filtered by extract_cve.py
        replace_existing = False  # Not applicable for CVE file import
    else:
        print(f"❌ Invalid choice: {choice}")
        sys.exit(1)

    # Use default settings
    chunk_size = CHUNK_SIZE
    batch_size = EMBEDDING_BATCH_SIZE
    precision = EMBEDDING_PRECISION

    # Display configuration
    print(f"\n{'='*60}")
    print(f"Configuration:")
    print(f"{'='*60}")
    print(f"Source:      {source_type}")
    if source_type == 'pdf':
        print(f"Files:       {', '.join(pdf_files)}")
    elif source_type == 'cve':
        print(f"Year:        {year}")
        print(f"Schema:      {schema}")
        print(f"Mode:        {'Replace' if replace_existing else 'Add only'}")
        if filter_keyword:
            print(f"Filter:      '{filter_keyword}' (case-insensitive)")
    elif source_type == 'cve_file':
        print(f"File:        {cve_file}")
    print(f"Chunk size:  {chunk_size} sentences")
    print(f"Batch size:  {batch_size}")
    print(f"Precision:   {precision}")
    print(f"Database:    {EMBEDDING_PATH}")
    print(f"{'='*60}\n")

    # Initialize components
    print(f"\nInitializing components...")
    chroma_manager = ChromaManager()
    chroma_manager.initialize(create_if_not_exists=True)

    embedding_model = EmbeddingModel()
    embedding_model.initialize()

    # Process based on source type
    try:
        if source_type == 'pdf':
            total_added = process_pdf_files(
                pdf_files,
                chroma_manager,
                embedding_model,
                chunk_size,
                batch_size,
                precision
            )

        elif source_type == 'cve':
            # If replace mode, delete existing year data first
            if replace_existing:
                print(f"\n{'='*60}")
                print(f"🗑️  Deleting existing data for year {year}")
                print(f"{'='*60}")
                deleted_count = chroma_manager.delete_by_year(year, schema)
                if deleted_count > 0:
                    print(f"✅ Cleared {deleted_count} existing documents")
                print()

            # Now add new data
            total_added = process_cve_data(
                year,
                month_range,
                chroma_manager,
                embedding_model,
                batch_size,
                precision,
                schema,
                filter_keyword
            )

        elif source_type == 'cve_file':
            total_added = process_cve_file(
                cve_file,
                chroma_manager,
                embedding_model,
                batch_size,
                precision,
                filter_keyword
            )

        # Show final statistics
        print(f"\n{'='*60}")
        print(f"Summary:")
        print(f"{'='*60}")
        print(f"Documents added:      {total_added}")

        stats = chroma_manager.get_stats()
        print(f"Total in database:    {stats['total_docs']}")
        print(f"By source type:       {stats['by_source_type']}")
        print(f"{'='*60}")

        print(f"\n✅ Operation completed successfully!")
        print(f"\n💡 Knowledge base updated at: {EMBEDDING_PATH}")

    except KeyboardInterrupt:
        print(f"\n⚠️ Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
