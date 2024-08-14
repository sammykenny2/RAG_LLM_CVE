
import fitz
from tqdm.auto import tqdm
import pandas as pd
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer
import re

## Code References https://github.com/mrdbourke/simple-local-rag

def text_formatter(text: str) -> str:
    return text.replace("\n", " ").strip()

def open_and_read_pdf(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc), desc="Processing pages"):
        text = text_formatter(page.get_text())
        pages_and_texts.append({
            "page_number": page_number,
            "text": text
        })
    return pages_and_texts

def split_list(input_list: list[str], slice_size: int) -> list[list[str]]:
    """Split list into chunks of specified size."""
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

def process_pdf(pdf_path: str, num_sentence_chunk_size: int, min_token_length: int, output_csv_path: str):
    """Process PDF to extract text, split into chunks, generate embeddings, and save to CSV."""
    pages_and_texts = open_and_read_pdf(pdf_path)

    # Initialize spaCy
    nlp = English()
    nlp.add_pipe("sentencizer")

    # Add sentences to each page
    for item in tqdm(pages_and_texts, desc="Processing sentences"):
        item["sentences"] = [str(sentence) for sentence in nlp(item["text"]).sents]
        item["page_sentence_count_spacy"] = len(item["sentences"])

    # Split sentences into chunks
    for item in tqdm(pages_and_texts, desc="Creating sentence chunks"):
        item["sentence_chunks"] = split_list(item["sentences"], num_sentence_chunk_size)
        item["num_chunks"] = len(item["sentence_chunks"])

    # Create list of chunks with stats
    pages_and_chunks = []
    for item in tqdm(pages_and_texts, desc="Processing chunks"):
        for sentence_chunk in item["sentence_chunks"]:
            joined_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            formated_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_chunk)
            pages_and_chunks.append({
                "page_number": item["page_number"],
                "sentence_chunk": formated_chunk
            })

    # Convert to DataFrame and filter
    df = pd.DataFrame(pages_and_chunks)
    filtered_chunks = df.to_dict(orient="records")

    # Initialize and use SentenceTransformer
    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cpu")
    embedding_model.to("cuda")


    for item in tqdm(filtered_chunks, desc="Generating embeddings"):
        item["embedding"] = embedding_model.encode(item["sentence_chunk"])

    # Save embeddings to CSV
    pd.DataFrame(filtered_chunks).to_csv(output_csv_path, index=False)

# Main execution
pdf_path = input("Please enter the PDF file path with the .pdf extension: ")
num_sentence_chunk_size = 10
min_token_length = 30
output_csv_path = input("Please enter the CSV file name to save embeddings (with .csv extension): ")
process_pdf(pdf_path, num_sentence_chunk_size, min_token_length, output_csv_path)
