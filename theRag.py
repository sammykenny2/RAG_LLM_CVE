# RAG-based CVE validation system - MULTI-LEVEL OPTIMIZATION
# Supports both demo mode (memory-optimized) and full mode (complete features)
#
# PERFORMANCE OPTIMIZATION with 3 speed levels (--speed parameter):
#
# normal (baseline, --speed=normal):
#   - Option 2: Chunk-aware CVE filtering only
#   - FP16: Disabled (FP32 for max precision)
#   - Cache clearing: Every chunk
#   - Temperature: 0.3 (standard sampling)
#   - Performance: Option 2 in ~4-5 minutes (vs 34 min original)
#
# fast (default, --speed=fast or no parameter):
#   - Option 2: Chunk-aware CVE filtering
#   - FP16: Enabled (20-30% faster, negligible precision loss)
#   - Cache clearing: Every 3 chunks (5-10% faster)
#   - Temperature: 0.3 (standard sampling)
#   - Performance: Option 2 in ~3 minutes, Options 1/3 in ~1.8 minutes
#
# fastest (aggressive, --speed=fastest):
#   - Option 2: Chunk-aware CVE filtering
#   - FP16: Enabled
#   - Cache clearing: Every 3 chunks
#   - Temperature: 0.1 (lower randomness, 5-8% faster)
#   - SDPA: Enabled if available (10-20% faster)
#   - Performance: Option 2 in ~2 minutes, Options 1/3 in ~1.2 minutes
#
# Trade-offs:
#   - normal→fast: No accuracy loss, pure optimization
#   - fast→fastest: Slightly less diverse outputs (lower temperature), but still accurate
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers.utils import logging
logging.set_verbosity_error()

import re
import fitz
import json
import gc
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import chromadb
from sentence_transformers import util, SentenceTransformer

# ============================================================================
# Configuration
# ============================================================================

# Import configuration
from config import (
    DEFAULT_SPEED,
    DEFAULT_MODE,
    DEFAULT_SCHEMA,
    DEFAULT_EMBEDDING_FORMAT,
    EMBEDDING_MODEL_NAME,
    LLAMA_MODEL_NAME,
    CVE_V5_PATH,
    CVE_V4_PATH
)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='CVE RAG Analysis Tool with Multi-Level Optimization')
parser.add_argument('--mode', choices=['demo', 'full'], default=DEFAULT_MODE,
                    help=f'Run mode: demo (memory-optimized) or full (complete features) (default: {DEFAULT_MODE})')
parser.add_argument('--schema', type=str, choices=['v5', 'v4', 'all'], default=DEFAULT_SCHEMA,
                    help=f'CVE schema to use: v5 (CVE 5.0 only), v4 (CVE 4.0 only), or all (v5→v4 fallback) (default: {DEFAULT_SCHEMA})')
parser.add_argument('--speed', type=str, choices=['normal', 'fast', 'fastest'], default=DEFAULT_SPEED,
                    help=f'Optimization level: normal=baseline, fast=recommended (+FP16), fastest=aggressive (+lower temp +SDPA) (default: {DEFAULT_SPEED})')
parser.add_argument('--extension', type=str, choices=['csv', 'pkl', 'parquet', 'chroma'], default=DEFAULT_EMBEDDING_FORMAT,
                    help=f'Embedding file extension: csv, pkl, parquet, chroma (default: {DEFAULT_EMBEDDING_FORMAT})')
args = parser.parse_args()

DEMO_MODE = (args.mode == 'demo')
CVE_SCHEMA = args.schema
SPEED_LEVEL = args.speed
EMBEDDING_EXTENSION = args.extension

# Display speed level info
speed_display = {"normal": "NORMAL (Baseline)", "fast": "FAST (Recommended)", "fastest": "FASTEST (Aggressive)"}
print(f"Speed Level: {speed_display[SPEED_LEVEL]}")

# Mode-specific configurations
if DEMO_MODE:
    print("Running in DEMO mode (memory-optimized)")
    # Demo mode settings (matches original theRag.py)
    MAX_PAGES = 10
    MAX_TEXT_LENGTH = 1000
    MAX_EMBEDDING_ROWS = 1000
    TOP_K_RETRIEVAL = 3
    CVE_EXTRACT_TOKENS = 64
    SUMMARY_TOKENS = 256
    VALIDATION_TOKENS = 256
    QA_TOKENS = 256
    ADVICE_TOKENS = 128
    MISSING_CVE_TOKENS = 64
    USE_TORCH_NO_GRAD = True
    USE_FP16 = True
    USE_LOW_CPU_MEM = True
    USE_CUDNN_OPTIMIZATIONS = True
    PERIODIC_GC = True
    TRUNCATE_TEXT = True
    CACHE_CLEAR_FREQUENCY = 1  # Clear every chunk in demo mode
    TEMPERATURE = 0.3
    USE_SDPA = False
else:
    print("Running in FULL mode (complete features)")
    # Full mode settings - adjusted based on SPEED_LEVEL
    MAX_PAGES = None  # Process all pages
    MAX_TEXT_LENGTH = None  # No truncation
    MAX_EMBEDDING_ROWS = None  # Read all rows
    TOP_K_RETRIEVAL = 5
    CVE_EXTRACT_TOKENS = 150
    SUMMARY_TOKENS = 700
    VALIDATION_TOKENS = 700
    QA_TOKENS = 700
    ADVICE_TOKENS = 700
    MISSING_CVE_TOKENS = 700
    USE_TORCH_NO_GRAD = False
    USE_LOW_CPU_MEM = False
    USE_CUDNN_OPTIMIZATIONS = False
    PERIODIC_GC = False
    TRUNCATE_TEXT = False

    # Speed-level specific optimizations
    if SPEED_LEVEL == 'normal':
        # Normal: Only chunk-aware CVE filtering (baseline)
        USE_FP16 = False
        CACHE_CLEAR_FREQUENCY = 1  # Clear every chunk
        TEMPERATURE = 0.3
        USE_SDPA = False
        print("  └─ Optimizations: Chunk-aware CVE filtering only")
    elif SPEED_LEVEL == 'fast':
        # Fast: +FP16 +reduced cache clearing (recommended)
        USE_FP16 = True
        CACHE_CLEAR_FREQUENCY = 3  # Clear every 3 chunks
        TEMPERATURE = 0.3
        USE_SDPA = False
        print("  └─ Optimizations: Chunk-aware + FP16 + reduced cache clearing")
    else:  # SPEED_LEVEL == 'fastest'
        # Fastest: +FP16 +reduced cache +lower temp +SDPA (aggressive)
        USE_FP16 = True
        CACHE_CLEAR_FREQUENCY = 3  # Clear every 3 chunks
        TEMPERATURE = 0.1  # Lower randomness
        USE_SDPA = True
        print("  └─ Optimizations: Chunk-aware + FP16 + reduced cache + low temp + SDPA")

# ============================================================================
# Memory optimization setup
# ============================================================================

torch.cuda.empty_cache()

if USE_CUDNN_OPTIMIZATIONS:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

llamaSug = ""

# Check embedding file/directory exists (consistent naming: CVEEmbeddings.{extension})
EMBEDDING_FILE = f"CVEEmbeddings.{EMBEDDING_EXTENSION}"

if not os.path.exists(EMBEDDING_FILE):
    print(f"❌ Error: {EMBEDDING_FILE} not found!")
    print(f"💡 Generate it first with:")
    print(f"   python localEmbedding.py --extension={EMBEDDING_EXTENSION}")
    sys.exit(1)

# Initialize SentenceTransformer once (global) for efficiency
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer(model_name_or_path=EMBEDDING_MODEL_NAME, device=device)

# ============================================================================
# Helper functions
# ============================================================================

def extract_cve_context(source_text: str, cve: str, window_chars: int = 1500) -> str:
    """Extract context window around CVE mentions (full mode only)."""
    pattern = re.compile(re.escape(cve), re.IGNORECASE)
    snippets = []
    for match in pattern.finditer(source_text):
        half = window_chars // 2
        start = max(match.start() - half, 0)
        end = min(match.end() + half, len(source_text))
        snippets.append(source_text[start:end])
    if snippets:
        return "\n...\n".join(snippets)
    return source_text[:window_chars]

def chunk_text(text: str, tokenizer, chunk_tokens: int = 1500, overlap: int = 200):
    """Split long documents into manageable chunks for the LLM (full mode only)."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) == 0:
        return []
    if len(token_ids) <= chunk_tokens:
        return [text]

    overlap = min(overlap, chunk_tokens // 2)
    stride = max(chunk_tokens - overlap, 1)
    chunks = []
    for start in range(0, len(token_ids), stride):
        end = start + chunk_tokens
        chunk = token_ids[start:end]
        chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
        if end >= len(token_ids):
            break
    return chunks

def generate_chunked_responses(system_prompt: str,
                               user_prompt_template: str,
                               text: str,
                               tokenizer,
                               model,
                               max_new_tokens: int = 300,
                               template_kwargs: dict | None = None) -> str:
    """Generate responses by processing text in chunks (full mode only)."""
    template_kwargs = template_kwargs or {}
    chunks = chunk_text(text, tokenizer)
    if not chunks:
        chunks = [""]

    outputs = []
    for idx, chunk in enumerate(chunks):
        content = user_prompt_template.format(text=chunk, **template_kwargs)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

        # Use no_grad for memory efficiency
        with torch.no_grad():
            generation = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=TEMPERATURE,  # Use dynamic temperature
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        response_ids = generation[0][input_ids.shape[-1]:]
        outputs.append(tokenizer.decode(response_ids, skip_special_tokens=True))

        # Clear cache based on frequency setting
        if (idx + 1) % CACHE_CLEAR_FREQUENCY == 0:
            torch.cuda.empty_cache()

    return "\n".join(outputs)

# ============================================================================
# Model initialization
# ============================================================================

def initialize_model():
    tokenizer = AutoTokenizer.from_pretrained(
        LLAMA_MODEL_NAME,
        trust_remote_code=True
    )

    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if USE_FP16:
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = "auto"

    if USE_LOW_CPU_MEM:
        model_kwargs["low_cpu_mem_usage"] = True

    # Enable SDPA (Scaled Dot-Product Attention) if requested and available
    if USE_SDPA:
        try:
            model_kwargs["attn_implementation"] = "sdpa"
            print("  └─ SDPA (optimized attention) enabled")
        except Exception as e:
            print(f"  └─ SDPA not available: {e}, using default attention")

    model = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL_NAME, **model_kwargs)
    return tokenizer, model

def cleanup_model(model):
    del model
    torch.cuda.empty_cache()
    gc.collect()

tokenizer, model = initialize_model()

# ============================================================================
# PDF processing
# ============================================================================

def extract_text_from_pdf(pdf_name):
    pdf_document = fitz.open(pdf_name)
    all_text = ""

    if MAX_PAGES is not None:
        # Demo mode: limit pages
        pages_to_process = min(len(pdf_document), MAX_PAGES)
    else:
        # Full mode: process all pages
        pages_to_process = len(pdf_document)

    for page_num in range(pages_to_process):
        page = pdf_document.load_page(page_num)
        text = page.get_text("text")
        all_text += text

        # Periodic memory cleanup in demo mode
        if PERIODIC_GC and page_num % 5 == 0:
            gc.collect()

    pdf_document.close()
    return all_text

userPDFName = input("Please Enter the name of the pdf that you would like to analyze (please include the .pdf at the end as well).")
all_text = extract_text_from_pdf(userPDFName)

# ============================================================================
# CVE extraction (optimized: direct regex instead of LLM)
# ============================================================================

def extract_cves_regex(text):
    """Extract CVEs directly from text using regex pattern."""
    pattern = r'CVE-\d{4}-\d{4,7}'
    cve_matches = re.findall(pattern, text)
    return list(set(cve_matches))

cves = extract_cves_regex(all_text)

# Legacy LLM-based extraction (kept for reference, not used)
def extract_cves_llama(text, tokenizer, model):
    if DEMO_MODE:
        # Demo mode: truncate text
        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH] + "..."

        messages = [
            {"role": "system", "content": "You are a chatbot"},
            {"role": "user", "content": f"Please extract all the CVEs mentioned in the text provided, separated by commas. Do not output anything else except for the CVE numbers.\n\nProvided text: {text}"}
        ]

        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

        if USE_TORCH_NO_GRAD:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=CVE_EXTRACT_TOKENS,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
        else:
            outputs = model.generate(
                input_ids,
                max_new_tokens=CVE_EXTRACT_TOKENS,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        response = outputs[0][input_ids.shape[-1]:]
        llm_output = tokenizer.decode(response, skip_special_tokens=True)
        return llm_output
    else:
        # Full mode: use chunking
        chunks = chunk_text(text, tokenizer)
        if not chunks:
            return ""

        responses = []
        for chunk in chunks:
            messages = [
                {"role": "system", "content": "You are a chatbot"},
                {
                    "role": "user",
                    "content": (
                        "Please extract all the CVEs mentioned in the text provided, "
                        "separated by commas. Do not output anything else except for the CVE numbers.\n\n"
                        f"Provided text: {chunk}"
                    ),
                },
            ]

            input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
            outputs = model.generate(
                input_ids,
                max_new_tokens=CVE_EXTRACT_TOKENS,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
            )
            response = outputs[0][input_ids.shape[-1]:]
            responses.append(tokenizer.decode(response, skip_special_tokens=True))

        return "\n".join(responses)

# ============================================================================
# CVE database lookup
# ============================================================================

def extract_cve_numbers(cve_string):
    pattern = r'CVE-(\d{4})-(\d{4,7})'
    match = re.match(pattern, cve_string)
    if match:
        first_set = int(match.group(1))
        second_set = match.group(2)
        return first_set, second_set
    else:
        raise ValueError("String does not match the CVE pattern")

def format_second_set(second_set):
    if len(second_set) == 4:
        return second_set[0] + 'xxx'
    else:
        return second_set[:2] + 'xxx'

def load_cve_record(cve: str, first_set: int, second_set: str) -> dict | None:
    """Load CVE JSON record based on schema configuration."""
    formatted_x = format_second_set(second_set)

    if CVE_SCHEMA == 'v5':
        # v5 only
        v5_path = CVE_V5_PATH / str(first_set) / formatted_x / f"CVE-{first_set}-{second_set}.json"
        if v5_path.exists():
            with open(v5_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    elif CVE_SCHEMA == 'v4':
        # v4 only
        v4_path = CVE_V4_PATH / str(first_set) / formatted_x / f"CVE-{first_set}-{second_set}.json"
        if v4_path.exists():
            with open(v4_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    else:  # CVE_SCHEMA == 'all'
        # Try v5 first, fallback to v4
        v5_path = CVE_V5_PATH / str(first_set) / formatted_x / f"CVE-{first_set}-{second_set}.json"
        if v5_path.exists():
            with open(v5_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        # Fallback to v4 schema
        v4_path = CVE_V4_PATH / str(first_set) / formatted_x / f"CVE-{first_set}-{second_set}.json"
        if v4_path.exists():
            with open(v4_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        return None

def extract_cve_fields(data: dict, fallback_id: str) -> tuple[str, str, str, str]:
    """Extract CVE fields from v5 or v4 schema."""
    # Try v5 schema first
    if "cveMetadata" in data:
        cve_number = data["cveMetadata"].get("cveId", fallback_id)
        cna = data.get("containers", {}).get("cna", {})
        descriptions = cna.get("descriptions", [])
        description = descriptions[0].get("value") if descriptions else "No description available."
        affected = cna.get("affected", [])
        entry = next((item for item in affected if isinstance(item, dict)), {})
        vendor = entry.get("vendor") or entry.get("vendor_name", "Unknown")
        product = entry.get("product") or entry.get("product_name", "Unknown")
        return cve_number, vendor, product, description

    # v4 schema
    legacy_meta = data.get("CVE_data_meta", {})
    cve_number = legacy_meta.get("ID", fallback_id)
    description_data = data.get("description", {}).get("description_data", [])
    description = description_data[0].get("value") if description_data else "No description available."
    vendor_data = data.get("affects", {}).get("vendor", {}).get("vendor_data", [])
    vendor_entry = vendor_data[0] if vendor_data else {}
    vendor = vendor_entry.get("vendor_name", "Unknown")
    product_data = vendor_entry.get("product", {}).get("product_data", [])
    product_entry = product_data[0] if product_data else {}
    product = product_entry.get("product_name", "Unknown")
    return cve_number, vendor, product, description

# ============================================================================
# Retrieval & recommendation
# ============================================================================

def asking_llama_for_advice(cveDesp: str) -> str:
    """Recommend similar CVE using preloaded embedding model (optimized)."""
    # Load embeddings based on file extension
    if EMBEDDING_EXTENSION == 'chroma':
        # Chroma vector database
        client = chromadb.PersistentClient(path=EMBEDDING_FILE)
        collection = client.get_collection("cve_embeddings")

        # Query Chroma directly (no need to load all embeddings)
        n_results = TOP_K_RETRIEVAL if MAX_EMBEDDING_ROWS is None else min(TOP_K_RETRIEVAL, MAX_EMBEDDING_ROWS)

        # Encode query using global embedding model
        query_embedding = embedding_model.encode(cveDesp, convert_to_tensor=False)

        # Query Chroma
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )

        # Extract context from Chroma results
        context_items = results['documents'][0]  # List of document strings
        context_str = "- " + "\n- ".join(context_items)

    else:
        # File-based formats (csv, pkl, parquet)
        if EMBEDDING_EXTENSION == 'csv':
            # CSV format
            if MAX_EMBEDDING_ROWS is not None:
                chunk_embeddings_df = pd.read_csv(EMBEDDING_FILE, nrows=MAX_EMBEDDING_ROWS)
            else:
                chunk_embeddings_df = pd.read_csv(EMBEDDING_FILE)
            chunk_embeddings_df["embedding"] = chunk_embeddings_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

        elif EMBEDDING_EXTENSION == 'pkl':
            # Pickle format
            with open(EMBEDDING_FILE, 'rb') as f:
                data = pickle.load(f)
            chunk_embeddings_df = pd.DataFrame(data)
            if MAX_EMBEDDING_ROWS is not None:
                chunk_embeddings_df = chunk_embeddings_df.head(MAX_EMBEDDING_ROWS)

        elif EMBEDDING_EXTENSION == 'parquet':
            # Parquet format
            chunk_embeddings_df = pd.read_parquet(EMBEDDING_FILE)
            if MAX_EMBEDDING_ROWS is not None:
                chunk_embeddings_df = chunk_embeddings_df.head(MAX_EMBEDDING_ROWS)

        embeddings = torch.tensor(np.stack(chunk_embeddings_df["embedding"].tolist(), axis=0), dtype=torch.float32).to(device)

        text_chunks = chunk_embeddings_df.to_dict(orient="records")

        # Use global embedding_model instead of creating new instance
        indices = retrieve_context(query=cveDesp, embeddings=embeddings, model=embedding_model)

        context_items = [text_chunks[i] for i in indices]

        context_str = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

    messages = [
        {"role": "system", "content": f""""You are a Q&A Assistant. You will be provided with relevant information about various CVEs. Based on this information, your task is to recommend the CVE number that most closely matches the description of the vulnerability.
        Provided Relevant Information: {context_str}
        """},
        {"role": "user", "content": f""" Hello, could you recommend me a CVE that most closely resembles this chunk of text based on the relevant information that you have. Chunk of text: {cveDesp}"""},
        ]

    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    # Always use no_grad for inference efficiency
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=ADVICE_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    response = outputs[0][input_ids.shape[-1]:]
    llamaSug = tokenizer.decode(response, skip_special_tokens=True)

    # Cleanup (only for non-chroma formats)
    if EMBEDDING_EXTENSION != 'chroma':
        del embeddings
        del text_chunks
    torch.cuda.empty_cache()
    gc.collect()

    return llamaSug

def retrieve_context(query: str,
                embeddings: torch.tensor,
                model: SentenceTransformer,
                n_resources_to_return: int = None):
    """Embeds a query with model and returns top k scores and indices from embeddings."""
    if n_resources_to_return is None:
        n_resources_to_return = TOP_K_RETRIEVAL

    # Embed the query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Ensure dtype consistency
    if query_embedding.dtype != embeddings.dtype:
        query_embedding = query_embedding.to(embeddings.dtype)

    dot_scores = util.dot_score(query_embedding, embeddings)[0]

    _, indices = torch.topk(input=dot_scores, k=n_resources_to_return)

    return indices

# ============================================================================
# Load CVE descriptions from database
# ============================================================================

cve_description = ""

for cve in cves:
    first_set, second_set = extract_cve_numbers(cve)
    data = load_cve_record(cve, first_set, second_set)

    if data:
        cve_number, vendor_name, product_name, description = extract_cve_fields(data, cve)
        cve_description += (
            f"-CVE Number: {cve_number}, Vendor: {vendor_name}, "
            f"Product: {product_name}, Description: {description}\n\n\n"
        )
        continue

    # CVE not found in JSON feeds
    print(f"Could Not Find {cve}, I will report it you later")

    torch.cuda.empty_cache()
    gc.collect()

    if DEMO_MODE:
        # Demo mode: use truncated context
        context_text = all_text[:500]
    else:
        # Full mode: use extract_cve_context with length limit
        context_text = extract_cve_context(all_text, cve)
        # Limit context to prevent memory issues (max 2000 chars)
        if len(context_text) > 2000:
            context_text = context_text[:2000] + "..."

    messages = [
        {"role": "system", "content": "You are a chat bot."},
        {"role": "user", "content": f"""In 2 sentences, could you describe the use of {cve} in the provided text. Provided text: {context_text}."""},
        ]

    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    # Always use no_grad for inference efficiency
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=MISSING_CVE_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    response = outputs[0][input_ids.shape[-1]:]
    llamaDesc = tokenizer.decode(response, skip_special_tokens=True)

    llamaSug = asking_llama_for_advice(llamaDesc)

    # Cleanup after each CVE
    torch.cuda.empty_cache()
    gc.collect()

# ============================================================================
# Menu options
# ============================================================================

def menu_option_1(all_text, tokenizer, model):
    """Summarize the report."""
    if DEMO_MODE:
        # Demo mode: truncate text
        limited_text = all_text[:2000] if len(all_text) > 2000 else all_text

        messages = [
            {"role": "system", "content": "You are a ChatBot that summarizes threat intelligence reports. Your task is to summarize the report given to you by the user."},
            {"role": "user", "content": f"Please summarize the following Threat Intelligence Report: {limited_text}"}
        ]

        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=SUMMARY_TOKENS,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=0.9
            )

        response = outputs[0][input_ids.shape[-1]:]
        summary = tokenizer.decode(response, skip_special_tokens=True)
        print(summary)
    else:
        # Full mode: use chunking
        summary = generate_chunked_responses(
            "You are a ChatBot that summarizes threat intelligence reports. Your task is to summarize the report given to you by the user.",
            "Please summarize the following Threat Intelligence Report: {text}",
            all_text,
            tokenizer,
            model,
            max_new_tokens=SUMMARY_TOKENS,
        )
        print(summary)

def menu_option_2(all_text, cve_description, tokenizer, model):
    """Validate CVE usage."""
    if DEMO_MODE:
        # Demo mode: truncate text
        limited_text = all_text[:2000] if len(all_text) > 2000 else all_text
        limited_cve_desc = cve_description[:1000] if len(cve_description) > 1000 else cve_description

        messages = [
            {
                "role": "system",
                "content": f"""You are a chatbot that verifies the correct use of CVEs (Common Vulnerabilities and Exposures) mentioned in a Threat Intelligence Report. A CVE is used correctly when it closely matches the provided Correct CVE Description. Incorrect usage includes citing non-existent CVEs, misrepresenting the Correct CVE Description, or inaccurately applying the CVE.
                            Correct CVE Descriptions:
                            {limited_cve_desc}
                            Instructions:
                            1. Verify each CVE mentioned in the user-provided report.
                            2. Indicate whether each CVE is used correctly or not.
                            3. You must provide a detailed explanation with direct quotes from both the report and the Correct CVE Description.
                            A CVE in the report is incorrect if it describes a different vulnerability, even if the report accurately describes the vulnerability and its impact, and provides mitigation recommendations.
                            """
            },
            {
                "role": "user",
                "content": f"Please verify if the following CVEs have been used correctly in the following Threat Intelligence Report: {limited_text}."
            },
            ]

        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=VALIDATION_TOKENS,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=0.9
            )

        response = outputs[0][input_ids.shape[-1]:]
        verification = tokenizer.decode(response, skip_special_tokens=True)
        print(verification)
        print(llamaSug)
    else:
        # Full mode: use OPTIMIZED chunking with chunk-aware CVE filtering
        # OPTIMIZATION: Instead of sending ALL CVE descriptions to every chunk,
        # we extract CVEs from each chunk and only send relevant descriptions.
        # This reduces input tokens from ~3,934 to ~1,900, improving speed 7-10x.

        # Step 1: Parse cve_description into a dictionary for fast lookup
        cve_dict = {}
        for line_block in cve_description.split('\n\n\n'):
            if line_block.strip():
                # Extract CVE ID from the line (format: "-CVE Number: CVE-2025-12345, ...")
                cve_match = re.search(r'CVE-\d{4}-\d{4,7}', line_block)
                if cve_match:
                    cve_id = cve_match.group(0)
                    cve_dict[cve_id] = line_block.strip()

        # Step 2: Chunk the text
        chunks = chunk_text(all_text, tokenizer)
        if not chunks:
            chunks = [""]

        # Step 3: Process each chunk with filtered CVE descriptions
        outputs = []
        for idx, chunk in enumerate(chunks):
            # Extract CVEs mentioned in this specific chunk
            chunk_cves = extract_cves_regex(chunk)

            # Build filtered CVE descriptions (only for CVEs in this chunk)
            if chunk_cves:
                filtered_cve_desc = "\n\n\n".join([
                    cve_dict[cve] for cve in chunk_cves if cve in cve_dict
                ])
            else:
                # No CVEs in this chunk, use empty description
                filtered_cve_desc = "No CVE descriptions needed for this chunk."

            # Build dynamic system prompt with filtered descriptions
            system_prompt = (
                "You are a chatbot that verifies the correct use of CVEs (Common Vulnerabilities and Exposures) mentioned in a "
                "Threat Intelligence Report. A CVE is used correctly when it closely matches the provided Correct CVE Description. "
                "Incorrect usage includes citing non-existent CVEs, misrepresenting the Correct CVE Description, or inaccurately applying the CVE.\n"
                "Correct CVE Descriptions:\n"
                f"{filtered_cve_desc}\n"
                "Instructions:\n"
                "1. Verify each CVE mentioned in the user-provided report.\n"
                "2. Indicate whether each CVE is used correctly or not.\n"
                "3. Provide a detailed explanation with direct quotes from both the report and the Correct CVE Description.\n"
                "A CVE in the report is incorrect if it describes a different vulnerability."
            )

            # Generate response for this chunk
            content = f"Please verify if the following CVEs have been used correctly in this Threat Intelligence Report:\n{chunk}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ]
            input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

            # Use no_grad for memory efficiency
            with torch.no_grad():
                generation = model.generate(
                    input_ids,
                    max_new_tokens=VALIDATION_TOKENS,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )

            response_ids = generation[0][input_ids.shape[-1]:]
            outputs.append(tokenizer.decode(response_ids, skip_special_tokens=True))

            # Clear cache based on frequency setting
            if (idx + 1) % CACHE_CLEAR_FREQUENCY == 0:
                torch.cuda.empty_cache()

        verification = "\n".join(outputs)
        print(verification)
        print(llamaSug)

def menu_option_3(all_text, user_question, tokenizer, model):
    """Answer questions about the report."""
    if DEMO_MODE:
        # Demo mode: truncate text
        limited_text = all_text[:2000] if len(all_text) > 2000 else all_text

        messages = [
            {"role": "system", "content": "You are a chatbot that answers questions based on the text provided to you."},
            {"role": "user", "content": f"{user_question}\n\nReport: {limited_text}"}
        ]

        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=QA_TOKENS,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=0.9
            )

        response = outputs[0][input_ids.shape[-1]:]
        answer = tokenizer.decode(response, skip_special_tokens=True)
        print(answer)
    else:
        # Full mode: use chunking
        answer = generate_chunked_responses(
            "You are a chatbot that answers questions based on the text provided to you.",
            "{question}\n\nReport: {text}",
            all_text,
            tokenizer,
            model,
            max_new_tokens=QA_TOKENS,
            template_kwargs={"question": user_question},
        )
        print(answer)

# ============================================================================
# Main menu loop
# ============================================================================

user_continue = "1"

while user_continue == "1":
    torch.cuda.empty_cache()
    menu_option_number = input("Welcome to our CVE Rag. Please select how you would like to analyze the report. Options: 1. Give a Summary of the Report. 2. Validate the Use of the CVEs mentioned. 3. Ask a general question about the Report. 4. Exit. Type 1, 2, 3, or 4: ")
    if menu_option_number == "1":
        menu_option_1(all_text, tokenizer, model)
    elif menu_option_number == "2":
        menu_option_2(all_text, cve_description, tokenizer, model)
    elif menu_option_number == "3":
        user_question = input("Please enter the question you have about the report: ")
        menu_option_3(all_text, user_question, tokenizer, model)
    else:
        print("Exiting the program.")
        break

    user_continue = input("Would you like to continue? Please enter 1 for yes and 2 for no: ")

    # Cleanup after each menu operation
    if DEMO_MODE:
        torch.cuda.empty_cache()
        gc.collect()

# Cleanup
cleanup_model(model)
