from langchain_core.runnables import RunnableWithFallbacks
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint
from tenacity import wait_chain, wait_fixed

# --------------------------------------------------
# CREDENTIALS
# --------------------------------------------------

OPENROUTER_KEYS = [
    "Key1" ,
    "Key2" ,
    "Key3",
]

HUGGINGFACE_TOKENS = [
    "Key4",
    "Key5",
]

# --------------------------------------------------
# MODEL → INFERENCE PROVIDERS
# --------------------------------------------------

MODEL_INFERENCE_PROVIDERS = {
    "google/gemma-3-27b-it": [
        "nebius",
        "featherless-ai",
        "scaleway",
    ],
    # future models
}

# --------------------------------------------------
# RETRY STRATEGY
# --------------------------------------------------

retry_wait_strategy = wait_chain(
    wait_fixed(3),
    wait_fixed(5),
    wait_fixed(6),
)

# --------------------------------------------------
# FALLBACK CHAIN CACHE
# --------------------------------------------------

_fallback_chain_cache = {}

# --------------------------------------------------
# INTERNAL BUILDER
# --------------------------------------------------

def _build_fallback_chain(model_name: str):
    if model_name not in MODEL_INFERENCE_PROVIDERS:
        raise ValueError(f"Model not supported: {model_name}")

    runnables = []

    # ---------- OpenRouter (primary)
    for key in OPENROUTER_KEYS:
        llm = ChatOpenAI(
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
            model=model_name,
            temperature=0,
        )

        runnable = llm.with_retry(
            stop_after_attempt=3
        )

        runnables.append(runnable)

    # ---------- Hugging Face (fallback)
    providers = MODEL_INFERENCE_PROVIDERS[model_name]

    for hf_token in HUGGINGFACE_TOKENS:
        for provider in providers:
            llm = HuggingFaceEndpoint(
                huggingfacehub_api_token=hf_token,
                repo_id=model_name,
                task="text2text-generation",
                temperature=0
            )

            runnable = llm.with_retry(
                stop_after_attempt=3
            )

            runnables.append(runnable)

    return RunnableWithFallbacks(
        runnable=runnables[0],
        fallbacks=runnables[1:]
    )

# --------------------------------------------------
# PUBLIC API
# --------------------------------------------------

def run_llm(model_name: str, request: str | dict):
    """
    Executes the LLM with retries and provider fallback.
    `request` is passed directly to the model.
    Returns raw LangChain response.
    """

    if model_name not in _fallback_chain_cache:
        _fallback_chain_cache[model_name] = _build_fallback_chain(model_name)

    chain = _fallback_chain_cache[model_name]

    # Pass through exactly as provided
    return chain.invoke(request)
