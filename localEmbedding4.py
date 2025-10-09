import fitz
from tqdm.auto import tqdm
import pandas as pd
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer
import torch

def read_pdf(pdf_path):
    """Extract text from each page of the PDF and return as a list of dictionaries."""
    pdf = fitz.open(pdf_path)
    texts = []

    for page in tqdm(pdf):
        text = page.get_text().replace("\n", " ").strip()
        texts.append({"text": text})
    
    return texts

def split(input_list, chunk_size):
    """Split a list into chunks of specified size."""
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]

def process_pdf(pdf_path, sentence_size, output_csv_path):
    """Process PDF to extract text, split into chunks, generate embeddings, and save to CSV."""
    texts = read_pdf(pdf_path)

    # Initialize spaCy
    nlp = English()
    nlp.add_pipe("sentencizer")

    # Add sentences to each page
    for item in tqdm(texts):
        item["sentences"] = [str(sentence) for sentence in nlp(item["text"]).sents]

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

    # Initialize and use SentenceTransformer
    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cpu")
    if torch.cuda.is_available():
        embedding_model.to("cuda")

    for item in tqdm(dic_chunks):
        item["embedding"] = embedding_model.encode(item["sentence_chunk"])

    # Save embeddings to CSV
    pd.DataFrame(dic_chunks).to_csv(output_csv_path, index=False)


# Main execution
if __name__ == "__main__":
    pdf_path = input("Please enter the PDF file path with the .pdf extension: ")
    sentence_size = 10
    output_csv_path = input("Please enter the CSV file name to save embeddings (with .csv extension): ")
    process_pdf(pdf_path, sentence_size, output_csv_path)
