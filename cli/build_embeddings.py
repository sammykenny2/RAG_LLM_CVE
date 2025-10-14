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

# Import configuration
from config import (
    DEFAULT_SPEED,
    DEFAULT_EMBEDDING_FORMAT,
    CHUNK_SIZE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_PRECISION,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_PATH
)

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
    """Split a list into chunks of specified size."""
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]

def process_pdf(pdf_path, sentence_size, output_path, batch_size, precision, extension):
    """Process PDF to extract text, split into chunks, generate embeddings, and save to file."""
    print(f"\n{'='*60}")
    print(f"Configuration:")
    print(f"  Chunk size: {sentence_size} sentences")
    print(f"  Batch size: {batch_size}")
    print(f"  Precision: {precision}")
    print(f"  Output format: {extension}")
    print(f"{'='*60}\n")

    texts = read_pdf(pdf_path)

    # Split text into sentences (supports both English and Chinese)
    print("Splitting text into sentences...")
    for item in tqdm(texts):
        item["sentences"] = split_sentences(item["text"])

    # Split sentences into chunks
    for item in tqdm(texts):
        item["sentence_chunks"] = split(item["sentences"], sentence_size)


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

        # Add to collection in batches (Chroma has batch size limits)
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            collection.add(
                ids=ids[i:i+batch_size],
                embeddings=embeddings_list[i:i+batch_size],
                documents=documents[i:i+batch_size]
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
  python localEmbedding.py                              # Default (fast, pkl)
  python localEmbedding.py --speed=normal --extension=csv   # High quality, CSV
  python localEmbedding.py --speed=fastest --extension=parquet  # Maximum speed, smallest file
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
    args = parser.parse_args()

    # Configure based on speed level
    if args.speed == 'normal':
        SENTENCE_SIZE = CHUNK_SIZE
        BATCH_SIZE = 32
        PRECISION = 'float32'
        print("Running in NORMAL mode (baseline quality)")
    elif args.speed == 'fast':
        SENTENCE_SIZE = CHUNK_SIZE
        BATCH_SIZE = EMBEDDING_BATCH_SIZE
        PRECISION = EMBEDDING_PRECISION
        print(f"Running in FAST mode (recommended) - chunk_size={CHUNK_SIZE}, batch_size={EMBEDDING_BATCH_SIZE}, precision={EMBEDDING_PRECISION}")
    else:  # fastest
        SENTENCE_SIZE = CHUNK_SIZE * 2
        BATCH_SIZE = EMBEDDING_BATCH_SIZE * 2
        PRECISION = EMBEDDING_PRECISION
        print(f"Running in FASTEST mode (aggressive optimization) - chunk_size={CHUNK_SIZE * 2}, batch_size={EMBEDDING_BATCH_SIZE * 2}, precision={EMBEDDING_PRECISION}")

    # Get user input for PDF path
    pdf_path = input("Please enter the PDF file path (with .pdf extension): ")

    # Construct output path from EMBEDDING_PATH with extension
    output_path = f"{EMBEDDING_PATH}.{args.extension}"

    print(f"\nOutput will be saved to: {output_path}")
    print(f"(Configure via EMBEDDING_PATH in .env to change location)\n")

    # Process PDF
    process_pdf(pdf_path, SENTENCE_SIZE, output_path, BATCH_SIZE, PRECISION, args.extension)
