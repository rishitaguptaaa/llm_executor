# LLM Executor

A Python module that provides a robust fallback chain for executing LLM requests across multiple providers with automatic retry logic and seamless failover.

## Features

- **Multi-Provider Support**: Automatically routes requests through OpenRouter and Hugging Face endpoints
- **Intelligent Retry Logic**: Built-in retry mechanism with configurable wait strategies
- **Fallback Chain**: Cascades through multiple API keys and providers on failure
- **Provider-Specific Routing**: Supports routing different models to their optimal providers
- **Caching**: Caches compiled fallback chains to avoid redundant initialization
- **Zero-Configuration**: Simple `run_llm()` interface for straightforward usage
- **Built to handle real-world LLM failure modes like rate limits, provider outages, and flaky inference endpoints.**

## Installation

```bash
pip install langchain-core langchain-openai langchain-huggingface tenacity
```

## Quick Start

```python
from llm_executor import run_llm

# Execute an LLM request
response = run_llm("google/gemma-3-27b-it", "What is the capital of France?")
print(response)
```

## Configuration

### Supported Models

Currently configured models and their providers:

| Model | Providers |
|-------|-----------|
| `google/gemma-3-27b-it` | nebius, featherless-ai, scaleway |

Add new models to the `MODEL_INFERENCE_PROVIDERS` dictionary:

```python
MODEL_INFERENCE_PROVIDERS = {
    "your-model-name": ["provider1", "provider2"],
}
```

### API Keys

The module uses two sets of credentials:

**OpenRouter API Keys** (`OPENROUTER_KEYS`)
- Primary provider for LLM inference
- Supports multiple keys for load distribution and fallback
- Obtain keys from [openrouter.ai](https://openrouter.ai)

**Hugging Face Tokens** (`HUGGINGFACE_TOKENS`)
- Secondary fallback provider
- Multiple tokens allow concurrent requests
- Obtain tokens from [huggingface.co](https://huggingface.co)

### Retry Strategy

The default retry wait strategy uses a chain of fixed waits:

```python
retry_wait_strategy = wait_chain(
    wait_fixed(3),   # 3 seconds
    wait_fixed(5),   # 5 seconds
    wait_fixed(6),   # 6 seconds
)
```

Customize by modifying the `retry_wait_strategy` variable:

```python
from tenacity import wait_exponential, wait_chain

retry_wait_strategy = wait_chain(
    wait_exponential(multiplier=1, min=2, max=10),
)
```

## API Reference

### `run_llm(model_name: str, request: str | dict) -> Any`

Executes an LLM request with retries and provider fallback.

**Parameters:**
- `model_name` (str): The model identifier (e.g., `"google/gemma-3-27b-it"`)
- `request` (str | dict): The input to pass to the model (e.g., prompt text or structured request)

**Returns:**
- Raw LangChain response object from the model

**Raises:**
- `ValueError`: If the model is not in `MODEL_INFERENCE_PROVIDERS`

**Example:**

```python
# Simple string request
response = run_llm("google/gemma-3-27b-it", "Explain quantum computing")

# Structured request (dict)
response = run_llm("google/gemma-3-27b-it", {
    "messages": [
        {"role": "user", "content": "What is 2+2?"}
    ]
})
```

## How It Works

The executor builds a fallback chain in this order:

1. **OpenRouter (Primary)**
   - Tries each OpenRouter API key sequentially
   - Each key has 3 retry attempts
   - Wait strategy: 3s, 5s, 6s between retries

2. **Hugging Face (Fallback)**
   - If all OpenRouter attempts fail, cascades to Hugging Face
   - Tries each combination of HF token + provider
   - Each attempt has 3 retries with same wait strategy

The chain is cached per model, so subsequent calls reuse the compiled fallback logic.

## Example Usage

```python
from llm_executor import run_llm

# Basic usage
model = "google/gemma-3-27b-it"
prompt = "Generate a Python function that checks if a number is prime"

try:
    response = run_llm(model, prompt)
    print(f"Response: {response.content}")
except Exception as e:
    print(f"All providers failed: {e}")
```

## Architecture

```
run_llm()
    ↓
_fallback_chain_cache (check for compiled chain)
    ↓
_build_fallback_chain() (if not cached)
    ├─ OpenRouter Runnable (primary)
    │  ├─ Key 1 (retry: 3x)
    │  ├─ Key 2 (retry: 3x)
    │  └─ Key N (retry: 3x)
    │
    └─ HuggingFace Runnable (fallback)
       ├─ Token 1 + Provider 1 (retry: 3x)
       ├─ Token 1 + Provider 2 (retry: 3x)
       └─ Token N + Provider N (retry: 3x)
```

## Environment Variables

For better security, consider loading credentials from environment variables:

```python
import os

OPENROUTER_KEYS = [os.getenv("OPENROUTER_KEY_1"), ...]
HUGGINGFACE_TOKENS = [os.getenv("HF_TOKEN_1"), ...]
```

## Troubleshooting

### Model Not Supported Error
```
ValueError: Model not supported: your-model-name
```
**Solution**: Add the model to `MODEL_INFERENCE_PROVIDERS` with at least one provider.

### All Providers Failed
**Possible causes:**
- Invalid API keys or tokens
- Model name mismatch with provider
- Rate limiting from providers
- Network connectivity issues

**Solutions:**
1. Verify API keys are valid and not expired
2. Check model availability on each provider
3. Review provider rate limits
4. Check network connectivity
5. Increase retry wait times if rate limited

### Slow Responses
- Increase the initial wait time in `retry_wait_strategy`
- Add more API keys to distribute load
- Check provider status pages for outages

## Contributing

To add support for new models:

1. Add the model to `MODEL_INFERENCE_PROVIDERS`:
   ```python
   MODEL_INFERENCE_PROVIDERS = {
       "new-model": ["provider1", "provider2"],
   }
   ```

2. Test with `run_llm("new-model", "test prompt")`

3. Submit a PR with the changes

