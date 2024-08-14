import fitz
from tqdm.auto import tqdm
import pandas as pd
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer
import re

def read_pdf(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages_and_texts = []

    for page in tqdm(doc):
        text = page.get_text().replace("\n", " ").strip()
        pages_and_texts.append({
            "text": text 
        })
    return pages_and_texts


def split_list(input_list: list[str], slice_size: int) -> list[list[str]]:
    """Split list into chunks of specified size."""
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

def process_pdf(pdf_path: str, sentence_size: int, output_csv_path: str):
    """Process PDF to extract text, split into chunks, generate embeddings, and save to CSV."""
    pages_and_texts = read_pdf(pdf_path)

    # Initialize spaCy
    nlp = English()
    nlp.add_pipe("sentencizer")

    # Add sentences to each page
    for item in tqdm(pages_and_texts):
        item["sentences"] = [str(sentence) for sentence in nlp(item["text"]).sents]

    # Split sentences into chunks
    for item in tqdm(pages_and_texts):
        item["sentence_chunks"] = split_list(item["sentences"], sentence_size)

    # Create list of chunks without page number
    pages_and_chunks = []
    for item in tqdm(pages_and_texts):
        for sentence_chunk in item["sentence_chunks"]:
            join_chunk = "".join(sentence_chunk).strip()
            formated_chunk = re.sub(r'\.([A-Z])', r'. \1', join_chunk)
            pages_and_chunks.append({
                "sentence_chunk": formated_chunk 
            })

    # Convert to DataFrame and filter
    df = pd.DataFrame(pages_and_chunks)
    dic_chunks = df.to_dict(orient="records")

    # Initialize and use SentenceTransformer
    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cpu")
    embedding_model.to("cuda")

    for item in tqdm(dic_chunks):
        item["embedding"] = embedding_model.encode(item["sentence_chunk"])

    # Save embeddings to CSV
    pd.DataFrame(dic_chunks).to_csv(output_csv_path, index=False)

# Main execution
pdf_path = input("Please enter the PDF file path with the .pdf extension: ")
num_sentence_chunk_size = 10
output_csv_path = input("Please enter the CSV file name to save embeddings (with .csv extension): ")
process_pdf(pdf_path, num_sentence_chunk_size, output_csv_path)
