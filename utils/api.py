import time
import asyncio
import json
import os
from typing import List, Optional, Dict, Any, Tuple, Callable
from openai import OpenAI, AsyncOpenAI
import openai

# For local model support
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True

    # Cache for loaded models and tokenizers
    _MODEL_CACHE = {}
    _TOKENIZER_CACHE = {}

except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Initialize both sync and async clients
_api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=_api_key)
async_client = AsyncOpenAI(api_key=_api_key)

# Pricing per 1M tokens (as of 2024)
MODEL_PRICING = {
    # Chat models
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-4.1-nano": {"input": 0.010, "output": 0.040},
    "gpt-4.1-mini": {"input": 0.040, "output": 1.60},
    "gpt-4.1": {"input": 3.00, "output": 12.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # Embedding models
    "text-embedding-3-small": {"input": 0.020, "output": 0.0},
    "text-embedding-3-large": {"input": 0.130, "output": 0.0},
    "text-embedding-ada-002": {"input": 0.100, "output": 0.0},
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    """
    Calculate the cost of an API call based on token usage.
    
    Args:
        model: The model name
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens
        
    Returns:
        Cost in USD
    """
    if model not in MODEL_PRICING:
        print(f"⚠️ Unknown model '{model}', cost calculation may be inaccurate, use GPT-5 as default")
        model = "gpt-5"
    
    pricing = MODEL_PRICING[model]
    cost = (input_tokens / 1_000_000 * pricing["input"]) + \
           (output_tokens / 1_000_000 * pricing["output"])
    return cost


# ===== Retry utilities =====
def retry_with_backoff(func, *args, max_retries=5, initial_delay=1, **kwargs):
    """
    Retry a synchronous API call with exponential backoff.
    """
    delay = initial_delay
    for _ in range(max_retries):
        try:
            return func(*args, **kwargs)
        except (openai.RateLimitError, openai.APIError, openai.APITimeoutError) as e:
            print(f"⚠️ API error ({e.__class__.__name__}): {e}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            raise
    raise RuntimeError(f"❌ Max retries ({max_retries}) exceeded for {func.__name__}")


async def async_retry_with_backoff(func, *args, max_retries=5, initial_delay=1, **kwargs):
    """
    Retry an asynchronous API call with exponential backoff.
    """
    delay = initial_delay
    for _ in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except (openai.RateLimitError, openai.APIError, openai.APITimeoutError) as e:
            print(f"⚠️ API error ({e.__class__.__name__}): {e}. Retrying in {delay}s...")
            await asyncio.sleep(delay)
            delay *= 2
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            raise
    raise RuntimeError(f"❌ Max retries ({max_retries}) exceeded for {func.__name__}")


# ===== Chat completions =====
def call_chat_completion(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs,
) -> Tuple[str, float]:
    """
    Synchronous chat completion with retry and error handling.

    Returns:
        Tuple of (response_text, cost_in_usd)
    """
    # Check if this is a local Qwen model
    if model.startswith("qwen") or "qwen" in model.lower():
        return _call_local_qwen_completion(messages, model, temperature, max_tokens, **kwargs)

    # OpenAI API models
    if "gpt-5" in model:
        response = retry_with_backoff(
            client.chat.completions.create,
            model=model,
            messages=messages,
            **kwargs,
        )
    else:
        response = retry_with_backoff(
            client.chat.completions.create,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    # Calculate cost
    usage = response.usage
    cost = calculate_cost(
        model=model,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens
    )

    content = response.choices[0].message.content or ""
    return content.strip(), cost


def call_chat_completion_with_tools(
    messages: List[Dict],
    tools: List[Dict],
    tool_runner: Callable[[str, Dict], str],
    model: str = "gpt-4.1",
    temperature: float = 0.0,
    max_iterations: int = 20,
) -> Tuple[str, float]:
    """
    Run chat completion with tool calling. When the model requests a tool call,
    tool_runner(tool_name, arguments_dict) is invoked and the result is fed back.
    Returns (final assistant text, total cost).
    """
    total_cost = 0.0
    current_messages = list(messages)

    for _ in range(max_iterations):
        response = retry_with_backoff(
            client.chat.completions.create,
            model=model,
            messages=current_messages,
            tools=tools,
            tool_choice="auto",
            temperature=temperature,
        )

        msg = response.choices[0].message
        usage = response.usage
        total_cost += calculate_cost(
            model=model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
        )

        if not getattr(msg, "tool_calls", None):
            content = msg.content or ""
            return content.strip(), total_cost

        # Append assistant message (with tool_calls) as dict for next request
        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        assistant_msg["tool_calls"] = [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]
        current_messages.append(assistant_msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            result = tool_runner(name, args)
            if not isinstance(result, str):
                result = json.dumps(result)
            current_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "", total_cost


def _call_local_qwen_completion(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs,
) -> Tuple[str, float]:
    """
    Call local Qwen model using transformers.

    Args:
        messages: Chat messages in OpenAI format
        model: Model name (e.g., "qwen2.5-7b-instruct")
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Tuple of (response_text, cost_in_usd) - cost is 0 for local models
    """
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("Transformers library not available. Install with: pip install transformers torch")

    # Map model names to HuggingFace paths
    model_mapping = {
        "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5-14b-instruct": "Qwen/Qwen2.5-14B-Instruct",
        "qwen2.5-72b-instruct": "Qwen/Qwen2.5-72B-Instruct",
        "qwen2-7b-instruct": "Qwen/Qwen2-7B-Instruct",
        "qwen2-72b-instruct": "Qwen/Qwen2-72B-Instruct",
        "qwen1.5-7b-chat": "Qwen/Qwen1.5-7B-Chat",
        "qwen1.5-14b-chat": "Qwen/Qwen1.5-14B-Chat",
    }

    hf_model_name = model_mapping.get(model, model)  # Use provided name if not in mapping

    try:
        # Load tokenizer and model (with caching)
        if hf_model_name not in _TOKENIZER_CACHE:
            print(f"Loading tokenizer for {hf_model_name}...")
            _TOKENIZER_CACHE[hf_model_name] = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
        tokenizer = _TOKENIZER_CACHE[hf_model_name]

        if hf_model_name not in _MODEL_CACHE:
            print(f"Loading model {hf_model_name}...")
            _MODEL_CACHE[hf_model_name] = AutoModelForCausalLM.from_pretrained(
                hf_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        model_instance = _MODEL_CACHE[hf_model_name]

        # Convert messages to Qwen chat format
        if len(messages) == 1:
            # System + user in one message
            prompt = messages[0]["content"]
        elif len(messages) == 2 and messages[0]["role"] == "system":
            # System + user
            system_msg = messages[0]["content"]
            user_msg = messages[1]["content"]
            prompt = f"System: {system_msg}\n\nHuman: {user_msg}\n\nAssistant:"
        else:
            # Convert to simple text format
            prompt_parts = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    prompt_parts.append(f"System: {content}")
                elif role == "user":
                    prompt_parts.append(f"Human: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")
            prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model_instance.device)

        # Generate
        with torch.no_grad():
            outputs = model_instance.generate(
                **inputs,
                max_new_tokens=max_tokens or 2048,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
                **kwargs
            )

        # Decode response
        response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        return response_text.strip(), 0.0  # No cost for local models

    except Exception as e:
        raise RuntimeError(f"Error running local Qwen model: {e}")


def clear_qwen_cache():
    """
    Clear the cached Qwen models and tokenizers to free memory.
    Call this when you want to load different models or free GPU memory.
    """
    global _MODEL_CACHE, _TOKENIZER_CACHE
    if TRANSFORMERS_AVAILABLE:
        for model in _MODEL_CACHE.values():
            del model
        for tokenizer in _TOKENIZER_CACHE.values():
            del tokenizer
        _MODEL_CACHE.clear()
        _TOKENIZER_CACHE.clear()
        torch.cuda.empty_cache()
        print("Qwen model cache cleared")


async def async_call_chat_completion(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs,
) -> Tuple[str, float]:
    """
    Asynchronous chat completion with retry and error handling.
    
    Returns:
        Tuple of (response_text, cost_in_usd)
    """
    if "gpt-5" in model:
        response = await async_retry_with_backoff(
            async_client.chat.completions.create,
            model=model,
            messages=messages,
            **kwargs,
        )
    else:
        response = await async_retry_with_backoff(
            async_client.chat.completions.create,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
    
    # Calculate cost
    usage = response.usage
    cost = calculate_cost(
        model=model,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens
    )
    
    content = response.choices[0].message.content.strip()
    # print(f"💰 Cost: ${cost:.6f} | Tokens: {usage.prompt_tokens} in + {usage.completion_tokens} out = {usage.total_tokens} total")
    
    return content, cost


# ===== Embeddings =====
def call_embedding(text: str, model: str = "text-embedding-3-small", **kwargs) -> Tuple[List[float], float]:
    """
    Synchronous embedding call with retry and error handling.
    
    Returns:
        Tuple of (embedding_vector, cost_in_usd)
    """
    response = retry_with_backoff(
        client.embeddings.create,
        model=model,
        input=text,
        **kwargs,
    )
    
    # Calculate cost
    usage = response.usage
    cost = calculate_cost(
        model=model,
        input_tokens=usage.total_tokens
    )
    
    print(f"💰 Cost: ${cost:.6f} | Tokens: {usage.total_tokens} total")
    
    return response.data[0].embedding, cost


async def async_call_embedding(text: str, model: str = "text-embedding-3-small", **kwargs) -> Tuple[List[float], float]:
    """
    Asynchronous embedding call with retry and error handling.
    
    Returns:
        Tuple of (embedding_vector, cost_in_usd)
    """
    response = await async_retry_with_backoff(
        async_client.embeddings.create,
        model=model,
        input=text,
        **kwargs,
    )
    
    # Calculate cost
    usage = response.usage
    cost = calculate_cost(
        model=model,
        input_tokens=usage.total_tokens
    )
    
    print(f"💰 Cost: ${cost:.6f} | Tokens: {usage.total_tokens} total")
    
    return response.data[0].embedding, cost


def call_batch_embedding(texts: List[str], model: str = "text-embedding-3-large", **kwargs) -> Tuple[List[List[float]], float]:
    """
    Synchronous batch embedding call with retry and error handling.
    OpenAI supports up to 2048 texts per call.
    
    Args:
        texts: List of text strings to embed
        model: Embedding model to use
        
    Returns:
        Tuple of (list_of_embedding_vectors, total_cost_in_usd)
    """
    if not texts:
        return [], 0.0
    
    response = retry_with_backoff(
        client.embeddings.create,
        model=model,
        input=texts,
        **kwargs,
    )
    
    # Calculate cost
    usage = response.usage
    cost = calculate_cost(
        model=model,
        input_tokens=usage.total_tokens
    )
    
    # Extract embeddings in order
    embeddings = [item.embedding for item in response.data]
    
    return embeddings, cost


async def async_call_batch_embedding(texts: List[str], model: str = "text-embedding-3-large", **kwargs) -> Tuple[List[List[float]], float]:
    """
    Asynchronous batch embedding call with retry and error handling.
    OpenAI supports up to 2048 texts per call.
    
    Args:
        texts: List of text strings to embed
        model: Embedding model to use
        
    Returns:
        Tuple of (list_of_embedding_vectors, total_cost_in_usd)
    """
    if not texts:
        return [], 0.0
    
    response = await async_retry_with_backoff(
        async_client.embeddings.create,
        model=model,
        input=texts,
        **kwargs,
    )
    
    # Calculate cost
    usage = response.usage
    cost = calculate_cost(
        model=model,
        input_tokens=usage.total_tokens
    )
    
    # Extract embeddings in order
    embeddings = [item.embedding for item in response.data]
    
    return embeddings, cost
