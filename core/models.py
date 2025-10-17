"""
Llama model loading and management module.
Provides unified interface for loading Meta's Llama 3.2-1B-Instruct model.
"""

import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import logging as transformers_logging

# Suppress transformers warnings
transformers_logging.set_verbosity_error()

# Import configuration
from config import (
    LLAMA_MODEL_NAME,
    CUDA_DEVICE,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    VERBOSE_LOGGING
)


class LlamaModel:
    """
    Wrapper for Llama model with automatic device selection and optimization.

    Example usage:
        llm = LlamaModel(use_fp16=True, use_sdpa=False)
        tokenizer, model = llm.get_model()
        # ... use model ...
        llm.cleanup()
    """

    def __init__(
        self,
        model_name: str = None,
        use_fp16: bool = False,
        use_low_cpu_mem: bool = False,
        use_sdpa: bool = False,
        use_4bit_quantization: bool = None
    ):
        """
        Initialize Llama model wrapper.

        Args:
            model_name: Hugging Face model ID (default from config.py)
            use_fp16: Use float16 precision for GPU speedup
            use_low_cpu_mem: Enable low CPU memory usage mode
            use_sdpa: Enable Scaled Dot-Product Attention (if available)
            use_4bit_quantization: Use 4-bit quantization (auto-enabled for 3B+ models on CUDA)
        """
        self.model_name = model_name or LLAMA_MODEL_NAME
        self.use_fp16 = use_fp16
        self.use_low_cpu_mem = use_low_cpu_mem
        self.use_sdpa = use_sdpa

        # Auto-enable 4-bit quantization for 3B+ models on CUDA
        if use_4bit_quantization is None:
            self.use_4bit_quantization = self._should_use_quantization()
        else:
            self.use_4bit_quantization = use_4bit_quantization

        self.tokenizer = None
        self.model = None
        self._initialized = False

    def _should_use_quantization(self) -> bool:
        """
        Determine if 4-bit quantization should be used based on model size and device.

        Returns:
            bool: True if quantization is recommended
        """
        # Check if CUDA is available
        if not torch.cuda.is_available():
            return False

        # Check model size from name (3B, 8B, 70B, etc.)
        model_name_lower = self.model_name.lower()
        large_model_indicators = ["3b", "8b", "13b", "70b"]

        for indicator in large_model_indicators:
            if indicator in model_name_lower:
                if VERBOSE_LOGGING:
                    print(f"🔍 Detected large model ({indicator.upper()}), enabling 4-bit quantization")
                return True

        return False

    def initialize(self):
        """Load tokenizer and model with specified optimizations."""
        if self._initialized:
            if VERBOSE_LOGGING:
                print("⚠️ Model already initialized, skipping...")
            return self.tokenizer, self.model

        if VERBOSE_LOGGING:
            print(f"Loading Llama model: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            local_files_only=True  # Use local cache only, no network required
        )

        # Build model kwargs
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "local_files_only": True,  # Use local cache only, no network required
        }

        # 4-bit quantization (takes priority over FP16)
        if self.use_4bit_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["quantization_config"] = quantization_config
            if VERBOSE_LOGGING:
                print("  └─ 4-bit quantization enabled (NF4 + double quant)")
        # Precision setting (only if not using quantization)
        elif self.use_fp16:
            model_kwargs["torch_dtype"] = torch.float16
            if VERBOSE_LOGGING:
                print("  └─ FP16 precision enabled")
        else:
            model_kwargs["torch_dtype"] = "auto"

        # Memory optimization
        if self.use_low_cpu_mem:
            model_kwargs["low_cpu_mem_usage"] = True
            if VERBOSE_LOGGING:
                print("  └─ Low CPU memory usage enabled")

        # Attention optimization (SDPA)
        if self.use_sdpa:
            try:
                model_kwargs["attn_implementation"] = "sdpa"
                if VERBOSE_LOGGING:
                    print("  └─ SDPA (optimized attention) enabled")
            except Exception as e:
                if VERBOSE_LOGGING:
                    print(f"  └─ SDPA not available: {e}, using default attention")

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )

        self._initialized = True

        if VERBOSE_LOGGING:
            device = next(self.model.parameters()).device
            print(f"✅ Model loaded on device: {device}")

        return self.tokenizer, self.model

    def get_model(self):
        """
        Get tokenizer and model (initializes if not already loaded).

        Returns:
            tuple: (tokenizer, model)
        """
        if not self._initialized:
            self.initialize()
        return self.tokenizer, self.model

    def cleanup(self):
        """Clean up model from memory and clear CUDA cache."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        torch.cuda.empty_cache()
        gc.collect()

        self._initialized = False

        if VERBOSE_LOGGING:
            print("✅ Model cleaned up from memory")

    def generate(
        self,
        messages: list,
        max_new_tokens: int = 256,
        temperature: float = None,
        top_p: float = None,
        do_sample: bool = True
    ) -> str:
        """
        Generate text from messages (chat format).

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (default from config)
            top_p: Nucleus sampling parameter (default from config)
            do_sample: Whether to use sampling

        Returns:
            str: Generated text
        """
        if not self._initialized:
            raise RuntimeError("Model not initialized. Call initialize() or get_model() first.")

        # Use config defaults if not specified
        temperature = temperature if temperature is not None else LLM_TEMPERATURE
        top_p = top_p if top_p is not None else LLM_TOP_P

        # Prepare input
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode response
        response_ids = outputs[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        return response


# =============================================================================
# Utility functions (backward compatible with theRag.py)
# =============================================================================

def initialize_model(use_fp16=False, use_low_cpu_mem=False, use_sdpa=False, use_4bit_quantization=None):
    """
    Initialize Llama model (backward compatible function).

    Args:
        use_fp16: Use float16 precision
        use_low_cpu_mem: Enable low CPU memory usage
        use_sdpa: Enable SDPA attention
        use_4bit_quantization: Use 4-bit quantization (auto-enabled for 3B+ on CUDA)

    Returns:
        tuple: (tokenizer, model)
    """
    llm = LlamaModel(
        use_fp16=use_fp16,
        use_low_cpu_mem=use_low_cpu_mem,
        use_sdpa=use_sdpa,
        use_4bit_quantization=use_4bit_quantization
    )
    return llm.initialize()


def cleanup_model(model):
    """
    Clean up model from memory (backward compatible function).

    Args:
        model: Model to clean up
    """
    del model
    torch.cuda.empty_cache()
    gc.collect()
