"""
Incremental embedding addition tool.
Add PDFs and CVE data to existing Chroma database.

Usage:
    # Add PDF files
    python addToEmbeddings.py --source=pdf --files=report1.pdf,report2.pdf

    # Add CVE data by year
    python addToEmbeddings.py --source=cve --year=2024

    # Add CVE data by year and month range
    python addToEmbeddings.py --source=cve --year=2024 --month=1-6

    # Custom chunk size and batch size
    python addToEmbeddings.py --source=pdf --files=report.pdf --chunk-size=20 --batch-size=128
"""

import argparse
import sys
from pathlib import Path
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
    CHROMA_DB_PATH,
    CVE_V5_PATH,
    CVE_V4_PATH,
    CHUNK_SIZE,
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
    """Split a list into chunks of specified size."""
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]

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

        # Create chunks
        print(f"Creating chunks (size={chunk_size})...")
        sentence_chunks = split_list(sentences, chunk_size)

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
    schema: str
):
    """
    Process CVE data and add to Chroma database.

    Args:
        year: Year to process (e.g., 2024)
        month_range: Tuple of (start_month, end_month) or None for all
        chroma_manager: ChromaManager instance
        embedding_model: EmbeddingModel instance
        batch_size: Batch size for embedding generation
        precision: 'float32' or 'float16'
        schema: 'v5', 'v4', or 'all'
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
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Add PDFs and CVE data to Chroma embeddings database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add single PDF
  python addToEmbeddings.py --source=pdf --files=report.pdf

  # Add multiple PDFs
  python addToEmbeddings.py --source=pdf --files=report1.pdf,report2.pdf

  # Add CVE data for 2024
  python addToEmbeddings.py --source=cve --year=2024

  # Add CVE data with month filter (placeholder, not fully implemented)
  python addToEmbeddings.py --source=cve --year=2024 --month=1-6

  # Custom settings
  python addToEmbeddings.py --source=pdf --files=report.pdf --chunk-size=20 --batch-size=128
        """
    )

    parser.add_argument(
        '--source',
        type=str,
        required=True,
        choices=['pdf', 'cve'],
        help='Source type: pdf or cve'
    )

    parser.add_argument(
        '--files',
        type=str,
        help='Comma-separated PDF files (for --source=pdf)'
    )

    parser.add_argument(
        '--year',
        type=int,
        help='Year for CVE data (for --source=cve)'
    )

    parser.add_argument(
        '--month',
        type=str,
        help='Month range (e.g., 1-6) for CVE data (for --source=cve)'
    )

    parser.add_argument(
        '--chunk-size',
        type=int,
        default=CHUNK_SIZE,
        help=f'Number of sentences per chunk (default: {CHUNK_SIZE})'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=EMBEDDING_BATCH_SIZE,
        help=f'Batch size for embedding generation (default: {EMBEDDING_BATCH_SIZE})'
    )

    parser.add_argument(
        '--precision',
        type=str,
        choices=['float32', 'float16'],
        default=EMBEDDING_PRECISION,
        help=f'Embedding precision (default: {EMBEDDING_PRECISION})'
    )

    parser.add_argument(
        '--schema',
        type=str,
        choices=['v5', 'v4', 'all'],
        default=DEFAULT_SCHEMA,
        help=f'CVE schema to use (default: {DEFAULT_SCHEMA})'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.source == 'pdf' and not args.files:
        parser.error("--files is required when --source=pdf")

    if args.source == 'cve' and not args.year:
        parser.error("--year is required when --source=cve")

    # Parse month range
    month_range = None
    if args.month:
        try:
            start, end = args.month.split('-')
            month_range = (int(start), int(end))
        except:
            parser.error("--month must be in format START-END (e.g., 1-6)")

    print(f"\n{'='*60}")
    print(f"Incremental Embedding Addition Tool")
    print(f"{'='*60}")
    print(f"Source: {args.source}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Precision: {args.precision}")
    print(f"Database: {CHROMA_DB_PATH}")

    # Initialize components
    print(f"\nInitializing components...")
    chroma_manager = ChromaManager()
    chroma_manager.initialize(create_if_not_exists=True)

    embedding_model = EmbeddingModel()
    embedding_model.initialize()

    # Process based on source type
    try:
        if args.source == 'pdf':
            pdf_files = [f.strip() for f in args.files.split(',')]
            total_added = process_pdf_files(
                pdf_files,
                chroma_manager,
                embedding_model,
                args.chunk_size,
                args.batch_size,
                args.precision
            )

        elif args.source == 'cve':
            total_added = process_cve_data(
                args.year,
                month_range,
                chroma_manager,
                embedding_model,
                args.batch_size,
                args.precision,
                args.schema
            )

        # Show final statistics
        print(f"\n{'='*60}")
        print(f"Summary")
        print(f"{'='*60}")
        print(f"Total documents added: {total_added}")

        stats = chroma_manager.get_stats()
        print(f"Total documents in database: {stats['total_docs']}")
        print(f"By source type: {stats['by_source_type']}")

        print(f"\n✅ Operation completed successfully!")

    except KeyboardInterrupt:
        print(f"\n⚠️ Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
