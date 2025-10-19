"""
Configuration management for RAG_LLM_CVE project.
Loads settings from .env file and provides validation.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# =============================================================================
# Path Configuration
# =============================================================================

def get_path(env_var: str, default: str) -> Path:
    """Get path from environment variable with default fallback."""
    path_str = os.getenv(env_var, default)
    return Path(path_str)

# Embedding base path (without extension)
# Usage: f"{EMBEDDING_PATH}.{extension}" where extension is csv, pkl, parquet, or chroma
EMBEDDING_PATH = get_path('EMBEDDING_PATH', './embeddings/CVEEmbeddings')

CVE_V5_PATH = get_path('CVE_V5_PATH', '../cvelistV5/cves')
CVE_V4_PATH = get_path('CVE_V4_PATH', '../cvelist')

# CVE description export path (without extension)
# Usage: f"{CVE_DESCRIPTION_PATH}.{extension}" where extension is txt or jsonl
CVE_DESCRIPTION_PATH = get_path('CVE_DESCRIPTION_PATH', './output/CVEDescription')

TEMP_UPLOAD_DIR = get_path('TEMP_UPLOAD_DIR', './temp_uploads')

# =============================================================================
# Model Configuration
# =============================================================================

LLAMA_MODEL_NAME = os.getenv('LLAMA_MODEL_NAME', 'meta-llama/Llama-3.2-1B-Instruct')
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'sentence-transformers/all-mpnet-base-v2')
HF_HOME = os.getenv('HF_HOME', None)  # None means use default ~/.cache/huggingface

# =============================================================================
# Default Parameters
# =============================================================================

DEFAULT_SPEED = os.getenv('DEFAULT_SPEED', 'fast')
DEFAULT_MODE = os.getenv('DEFAULT_MODE', 'full')
DEFAULT_SCHEMA = os.getenv('DEFAULT_SCHEMA', 'all')
DEFAULT_EMBEDDING_FORMAT = os.getenv('DEFAULT_EMBEDDING_FORMAT', 'chroma')
EMBEDDING_PRECISION = os.getenv('EMBEDDING_PRECISION', 'float16')

# CVE filter keyword (case-insensitive, empty string means no filter)
CVE_FILTER = os.getenv('CVE_FILTER', '').strip()

# Validate choices
assert DEFAULT_SPEED in ['normal', 'fast', 'fastest'], f"Invalid DEFAULT_SPEED: {DEFAULT_SPEED}"
assert DEFAULT_MODE in ['demo', 'full'], f"Invalid DEFAULT_MODE: {DEFAULT_MODE}"
assert DEFAULT_SCHEMA in ['v5', 'v4', 'all'], f"Invalid DEFAULT_SCHEMA: {DEFAULT_SCHEMA}"
assert DEFAULT_EMBEDDING_FORMAT in ['csv', 'pkl', 'parquet', 'chroma'], f"Invalid DEFAULT_EMBEDDING_FORMAT: {DEFAULT_EMBEDDING_FORMAT}"
assert EMBEDDING_PRECISION in ['float32', 'float16'], f"Invalid EMBEDDING_PRECISION: {EMBEDDING_PRECISION}"

# =============================================================================
# RAG Configuration
# =============================================================================

CONVERSATION_HISTORY_LENGTH = int(os.getenv('CONVERSATION_HISTORY_LENGTH', '10'))
RETRIEVAL_TOP_K = int(os.getenv('RETRIEVAL_TOP_K', '5'))
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '10'))
CHUNK_OVERLAP_RATIO = float(os.getenv('CHUNK_OVERLAP_RATIO', '0.3'))
EMBEDDING_BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', '64'))

# =============================================================================
# Web UI Configuration
# =============================================================================

GRADIO_SERVER_PORT = int(os.getenv('GRADIO_SERVER_PORT', '7860'))  # webUI.py (Pure Python)
GRADIO_SERVER_PORT_LANGCHAIN = int(os.getenv('GRADIO_SERVER_PORT_LANGCHAIN', '7861'))  # webUILangChain.py (LangChain)
GRADIO_SHARE = os.getenv('GRADIO_SHARE', 'False').lower() == 'true'
GRADIO_SERVER_NAME = os.getenv('GRADIO_SERVER_NAME', '127.0.0.1')
MAX_FILE_UPLOAD_SIZE_MB = int(os.getenv('MAX_FILE_UPLOAD_SIZE_MB', '50'))

# =============================================================================
# Advanced Configuration
# =============================================================================

VERBOSE_LOGGING = os.getenv('VERBOSE_LOGGING', 'False').lower() == 'true'

# CUDA device (-1 for CPU, None for auto-detection)
cuda_device_str = os.getenv('CUDA_DEVICE', '')
if cuda_device_str == '':
    CUDA_DEVICE = None  # Auto-detect
elif cuda_device_str == '-1':
    CUDA_DEVICE = -1  # Force CPU
else:
    CUDA_DEVICE = int(cuda_device_str)

# Max embedding rows (None for all)
max_rows_str = os.getenv('MAX_EMBEDDING_ROWS', '')
MAX_EMBEDDING_ROWS = int(max_rows_str) if max_rows_str else None

LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.3'))
LLM_TOP_P = float(os.getenv('LLM_TOP_P', '0.9'))

# =============================================================================
# Summary Configuration
# =============================================================================

SUMMARY_CHUNK_TOKENS = int(os.getenv('SUMMARY_CHUNK_TOKENS', '1500'))
SUMMARY_CHUNK_OVERLAP_TOKENS = int(os.getenv('SUMMARY_CHUNK_OVERLAP_TOKENS', '200'))
SUMMARY_TOKENS_PER_CHUNK = int(os.getenv('SUMMARY_TOKENS_PER_CHUNK', '150'))
SUMMARY_FINAL_TOKENS = int(os.getenv('SUMMARY_FINAL_TOKENS', '300'))
SUMMARY_CHUNK_THRESHOLD_CHARS = int(os.getenv('SUMMARY_CHUNK_THRESHOLD_CHARS', '3000'))
SUMMARY_ENABLE_SECOND_STAGE = os.getenv('SUMMARY_ENABLE_SECOND_STAGE', 'True').lower() == 'true'

# =============================================================================
# Validation Configuration
# =============================================================================

VALIDATION_CHUNK_TOKENS = int(os.getenv('VALIDATION_CHUNK_TOKENS', '1500'))
VALIDATION_CHUNK_OVERLAP_TOKENS = int(os.getenv('VALIDATION_CHUNK_OVERLAP_TOKENS', '200'))
VALIDATION_TOKENS_PER_CHUNK = int(os.getenv('VALIDATION_TOKENS_PER_CHUNK', '300'))
VALIDATION_CHUNK_THRESHOLD_CHARS = int(os.getenv('VALIDATION_CHUNK_THRESHOLD_CHARS', '3000'))
VALIDATION_ENABLE_CVE_FILTERING = os.getenv('VALIDATION_ENABLE_CVE_FILTERING', 'True').lower() == 'true'
VALIDATION_FINAL_TOKENS = int(os.getenv('VALIDATION_FINAL_TOKENS', '500'))
VALIDATION_ENABLE_SECOND_STAGE = os.getenv('VALIDATION_ENABLE_SECOND_STAGE', 'True').lower() == 'true'

# =============================================================================
# Q&A Configuration
# =============================================================================

QA_CHUNK_TOKENS = int(os.getenv('QA_CHUNK_TOKENS', '1500'))
QA_CHUNK_OVERLAP_TOKENS = int(os.getenv('QA_CHUNK_OVERLAP_TOKENS', '200'))
QA_TOKENS_PER_CHUNK = int(os.getenv('QA_TOKENS_PER_CHUNK', '200'))
QA_CHUNK_THRESHOLD_CHARS = int(os.getenv('QA_CHUNK_THRESHOLD_CHARS', '3000'))
QA_FINAL_TOKENS = int(os.getenv('QA_FINAL_TOKENS', '400'))
QA_ENABLE_SECOND_STAGE = os.getenv('QA_ENABLE_SECOND_STAGE', 'True').lower() == 'true'

# =============================================================================
# Path Validation
# =============================================================================

def validate_paths(check_cve_feeds=True, check_embeddings=False):
    """
    Validate that required paths exist.

    Args:
        check_cve_feeds: Whether to check CVE feed paths
        check_embeddings: Whether to check embedding path (disabled by default as files are generated)

    Returns:
        list: List of error messages (empty if all valid)
    """
    errors = []

    if check_cve_feeds:
        if not CVE_V5_PATH.exists():
            errors.append(f"CVE V5 path not found: {CVE_V5_PATH}")
        if not CVE_V4_PATH.exists():
            errors.append(f"CVE V4 path not found: {CVE_V4_PATH}")

    if check_embeddings:
        # Check if at least one embedding format exists
        embedding_exists = any([
            Path(f"{EMBEDDING_PATH}.chroma").exists(),
            Path(f"{EMBEDDING_PATH}.pkl").exists(),
            Path(f"{EMBEDDING_PATH}.csv").exists(),
            Path(f"{EMBEDDING_PATH}.parquet").exists()
        ])
        if not embedding_exists:
            errors.append(f"No embedding files found at: {EMBEDDING_PATH}.*")

    return errors

# =============================================================================
# Debug Utilities
# =============================================================================

def print_config():
    """Print current configuration (for debugging)."""
    print("=" * 60)
    print("RAG_LLM_CVE Configuration")
    print("=" * 60)
    print("\nPaths:")
    print(f"  EMBEDDING_PATH: {EMBEDDING_PATH}")
    print(f"  CVE_V5_PATH: {CVE_V5_PATH}")
    print(f"  CVE_V4_PATH: {CVE_V4_PATH}")
    print(f"  CVE_DESCRIPTION_PATH: {CVE_DESCRIPTION_PATH}")
    print(f"  TEMP_UPLOAD_DIR: {TEMP_UPLOAD_DIR}")

    print("\nModels:")
    print(f"  LLAMA_MODEL_NAME: {LLAMA_MODEL_NAME}")
    print(f"  EMBEDDING_MODEL_NAME: {EMBEDDING_MODEL_NAME}")
    print(f"  HF_HOME: {HF_HOME or '(default)'}")

    print("\nDefaults:")
    print(f"  DEFAULT_SPEED: {DEFAULT_SPEED}")
    print(f"  DEFAULT_MODE: {DEFAULT_MODE}")
    print(f"  DEFAULT_SCHEMA: {DEFAULT_SCHEMA}")
    print(f"  DEFAULT_EMBEDDING_FORMAT: {DEFAULT_EMBEDDING_FORMAT}")
    print(f"  EMBEDDING_PRECISION: {EMBEDDING_PRECISION}")

    print("\nRAG:")
    print(f"  CONVERSATION_HISTORY_LENGTH: {CONVERSATION_HISTORY_LENGTH}")
    print(f"  RETRIEVAL_TOP_K: {RETRIEVAL_TOP_K}")
    print(f"  CHUNK_SIZE: {CHUNK_SIZE}")
    print(f"  CHUNK_OVERLAP_RATIO: {CHUNK_OVERLAP_RATIO}")
    print(f"  EMBEDDING_BATCH_SIZE: {EMBEDDING_BATCH_SIZE}")

    print("\nWeb UI:")
    print(f"  GRADIO_SERVER_PORT: {GRADIO_SERVER_PORT} (webUI.py)")
    print(f"  GRADIO_SERVER_PORT_LANGCHAIN: {GRADIO_SERVER_PORT_LANGCHAIN} (webUILangChain.py)")
    print(f"  GRADIO_SHARE: {GRADIO_SHARE}")
    print(f"  GRADIO_SERVER_NAME: {GRADIO_SERVER_NAME}")
    print(f"  MAX_FILE_UPLOAD_SIZE_MB: {MAX_FILE_UPLOAD_SIZE_MB}")

    print("\nAdvanced:")
    print(f"  VERBOSE_LOGGING: {VERBOSE_LOGGING}")
    print(f"  CUDA_DEVICE: {CUDA_DEVICE or '(auto-detect)'}")
    print(f"  MAX_EMBEDDING_ROWS: {MAX_EMBEDDING_ROWS or '(all)'}")
    print(f"  LLM_TEMPERATURE: {LLM_TEMPERATURE}")
    print(f"  LLM_TOP_P: {LLM_TOP_P}")

    print("\nSummary:")
    print(f"  SUMMARY_CHUNK_TOKENS: {SUMMARY_CHUNK_TOKENS}")
    print(f"  SUMMARY_CHUNK_OVERLAP_TOKENS: {SUMMARY_CHUNK_OVERLAP_TOKENS}")
    print(f"  SUMMARY_TOKENS_PER_CHUNK: {SUMMARY_TOKENS_PER_CHUNK}")
    print(f"  SUMMARY_FINAL_TOKENS: {SUMMARY_FINAL_TOKENS}")
    print(f"  SUMMARY_CHUNK_THRESHOLD_CHARS: {SUMMARY_CHUNK_THRESHOLD_CHARS}")
    print(f"  SUMMARY_ENABLE_SECOND_STAGE: {SUMMARY_ENABLE_SECOND_STAGE}")

    print("\nValidation:")
    print(f"  VALIDATION_CHUNK_TOKENS: {VALIDATION_CHUNK_TOKENS}")
    print(f"  VALIDATION_CHUNK_OVERLAP_TOKENS: {VALIDATION_CHUNK_OVERLAP_TOKENS}")
    print(f"  VALIDATION_TOKENS_PER_CHUNK: {VALIDATION_TOKENS_PER_CHUNK}")
    print(f"  VALIDATION_CHUNK_THRESHOLD_CHARS: {VALIDATION_CHUNK_THRESHOLD_CHARS}")
    print(f"  VALIDATION_ENABLE_CVE_FILTERING: {VALIDATION_ENABLE_CVE_FILTERING}")
    print(f"  VALIDATION_FINAL_TOKENS: {VALIDATION_FINAL_TOKENS}")
    print(f"  VALIDATION_ENABLE_SECOND_STAGE: {VALIDATION_ENABLE_SECOND_STAGE}")

    print("\nQ&A:")
    print(f"  QA_CHUNK_TOKENS: {QA_CHUNK_TOKENS}")
    print(f"  QA_CHUNK_OVERLAP_TOKENS: {QA_CHUNK_OVERLAP_TOKENS}")
    print(f"  QA_TOKENS_PER_CHUNK: {QA_TOKENS_PER_CHUNK}")
    print(f"  QA_CHUNK_THRESHOLD_CHARS: {QA_CHUNK_THRESHOLD_CHARS}")
    print(f"  QA_FINAL_TOKENS: {QA_FINAL_TOKENS}")
    print(f"  QA_ENABLE_SECOND_STAGE: {QA_ENABLE_SECOND_STAGE}")
    print("=" * 60)

if __name__ == "__main__":
    # Test configuration
    print_config()

    # Validate paths
    errors = validate_paths(check_cve_feeds=True, check_embeddings=True)
    if errors:
        print("\n⚠️ Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\n✅ All paths validated successfully!")
