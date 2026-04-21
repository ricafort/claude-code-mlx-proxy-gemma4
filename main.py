import json
from typing import List, Dict, Any, Optional, Union, Literal
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import PreTrainedTokenizerFast
from mlx_lm import load as mlx_lm_load, generate as mlx_lm_generate, stream_generate as mlx_lm_stream_generate
try:
    from mlx_vlm import load as mlx_vlm_load, generate as mlx_vlm_generate, stream_generate as mlx_vlm_stream_generate
    has_vlm = True
except ImportError:
    has_vlm = False
from mlx_lm.utils import load_model, get_model_path
from mlx_lm.tokenizer_utils import TokenizerWrapper
from config import config

# Global variables for model and tokenizer
model = None
tokenizer = None
# Dynamic generation functions determined at load
generate_func = mlx_lm_generate
stream_generate_func = mlx_lm_stream_generate


# Content block models
class ContentBlockText(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ContentBlockImage(BaseModel):
    type: Literal["image"] = "image"
    source: Dict[str, Any]


class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Dict[str, Any]


class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]


class SystemContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ThinkingConfig(BaseModel):
    type: Literal["enabled", "disabled", "adaptive"]
    budget_tokens: Optional[int] = None


class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[
        str,
        List[
            Union[
                ContentBlockText,
                ContentBlockImage,
                ContentBlockToolUse,
                ContentBlockToolResult,
            ]
        ],
    ]


class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    original_model: Optional[str] = None


class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class MessageResponse(BaseModel):
    id: str
    type: str = "message"
    role: str = "assistant"
    content: List[ContentBlockText]
    model: str
    stop_reason: str = "end_turn"
    stop_sequence: Optional[str] = None
    usage: Usage


class MessageStreamResponse(BaseModel):
    type: str
    index: Optional[int] = None
    delta: Optional[Dict[str, Any]] = None
    usage: Optional[Usage] = None


def _load_model_with_fallback(model_name: str, tokenizer_config: dict):
    """Load model and tokenizer, falling back to PreTrainedTokenizerFast when
    the model's tokenizer uses the Transformers v5 TokenizersBackend class which
    is not available in older Transformers installations.
    """
    global generate_func, stream_generate_func
    
    if "gemma-4" in model_name.lower():
        if has_vlm:
            print(f"Detected Gemma 4, switching to mlx_vlm loaders...")
            generate_func = mlx_vlm_generate
            stream_generate_func = mlx_vlm_stream_generate
            return mlx_vlm_load(model_name, tokenizer_config_extra=tokenizer_config)
        else:
            raise ImportError("mlx_vlm is required to run Gemma 4 models!")
    
    try:
        generate_func = mlx_lm_generate
        stream_generate_func = mlx_lm_stream_generate
        return mlx_lm_load(model_name, tokenizer_config=tokenizer_config)
    except ValueError as e:
        if "TokenizersBackend" not in str(e):
            raise
        print(
            "Warning: Failed to load tokenizer via AutoTokenizer (TokenizersBackend not "
            "found). This typically means the model was saved with Transformers v5 but an "
            "older version is installed. Upgrading to 'transformers>=5.0.0' is recommended. "
            "Attempting fallback using PreTrainedTokenizerFast..."
        )

    # Use cached model files (downloaded by the failed load() call above, or already
    # present from a previous run).
    model_path, _ = get_model_path(model_name)
    mlx_model, mlx_config = load_model(model_path)

    # PreTrainedTokenizerFast.from_pretrained does not do the tokenizer-class
    # name lookup that AutoTokenizer performs, so it avoids the TokenizersBackend
    # error while still reading tokenizer.json correctly.
    hf_tokenizer = PreTrainedTokenizerFast.from_pretrained(
        str(model_path), **tokenizer_config
    )
    eos_token_id = mlx_config.get("eos_token_id")
    tokenizer = TokenizerWrapper(hf_tokenizer, eos_token_ids=eos_token_id)

    return mlx_model, tokenizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global model, tokenizer
    print(f"Loading MLX model: {config.MODEL_NAME}")

    # Prepare tokenizer config
    tokenizer_config = {}
    if config.TRUST_REMOTE_CODE:
        tokenizer_config["trust_remote_code"] = True
    if config.EOS_TOKEN:
        tokenizer_config["eos_token"] = config.EOS_TOKEN

    model, tokenizer = _load_model_with_fallback(config.MODEL_NAME, tokenizer_config)
    print("Model loaded successfully!")
    yield
    # Cleanup on shutdown
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)


def extract_text_from_content(
    content: Union[
        str,
        List[
            Union[
                ContentBlockText,
                ContentBlockImage,
                ContentBlockToolUse,
                ContentBlockToolResult,
            ]
        ],
    ],
) -> str:
    """Extract text content from Claude-style content blocks"""
    if isinstance(content, str):
        return content

    text_parts = []
    for block in content:
        if hasattr(block, "type") and block.type == "text":
            text_parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            text_parts.append(block.get("text", ""))

    return " ".join(text_parts)


def extract_system_text(
    system: Optional[Union[str, List[SystemContent]]],
) -> Optional[str]:
    """Extract system text from system parameter"""
    if isinstance(system, str):
        return system
    elif isinstance(system, list):
        return " ".join([content.text for content in system])
    return None


def format_messages_for_llama(
    messages: List[Message], system: Optional[Union[str, List[SystemContent]]] = None
) -> str:
    """Convert Claude-style messages to Llama format"""
    formatted_messages = []

    # Add system message if provided
    system_text = extract_system_text(system)
    if system_text:
        formatted_messages.append({"role": "system", "content": system_text})

    # Add user messages
    for message in messages:
        content_text = extract_text_from_content(message.content)
        formatted_messages.append({"role": message.role, "content": content_text})

    # Ensure we use the actual tokenizer from the processor if it exists
    actual_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer

    # Apply chat template if available
    if actual_tokenizer.chat_template is not None:
        try:
            result = actual_tokenizer.apply_chat_template(
                formatted_messages, add_generation_prompt=True, tokenize=False
            )
            # Ensure we return a string, not tokens
            if isinstance(result, str):
                return result
        except Exception:
            # Fall through to manual formatting if template fails
            pass

    # Fallback formatting (used if no template or template fails)
    prompt = ""
    for msg in formatted_messages:
        if msg["role"] == "system":
            prompt += f"<|system|>\n{msg['content']}\n<|end|>\n"
        elif msg["role"] == "user":
            prompt += f"<|user|>\n{msg['content']}\n<|end|>\n"
        elif msg["role"] == "assistant":
            prompt += f"<|assistant|>\n{msg['content']}\n<|end|>\n"
    prompt += "<|assistant|>\n"
    return prompt


def count_tokens(text: str) -> int:
    """Count tokens in text"""
    try:
        # MLX tokenizers often expect the text to be handled through their specific methods
        # First try the standard approach with proper string handling
        if isinstance(text, str) and text.strip():
            # Ensure we use the actual tokenizer
            actual_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
            
            # For MLX, we may need to use a different approach
            # Try to get tokens using the tokenizer's __call__ method or encode
            try:
                # Some MLX tokenizers work better with this approach
                result = actual_tokenizer(text, return_tensors=False, add_special_tokens=False)
                if isinstance(result, dict) and "input_ids" in result:
                    return len(result["input_ids"])
                elif hasattr(result, "__len__"):
                    return len(result)
            except (AttributeError, TypeError, ValueError):
                pass

            # Try direct encode without parameters
            try:
                encoded = actual_tokenizer.encode(text)
                return (
                    len(encoded) if hasattr(encoded, "__len__") else len(list(encoded))
                )
            except (AttributeError, TypeError, ValueError):
                pass

            # Try with explicit string conversion and basic parameters
            try:
                tokens = actual_tokenizer.encode(str(text), add_special_tokens=False)
                return len(tokens)
            except (AttributeError, TypeError, ValueError):
                pass

        # Final fallback: character-based estimation
        return max(1, len(str(text)) // 4)  # At least 1 token, ~4 chars per token

    except Exception as e:
        print(f"Token counting failed with error: {e}")
        return max(1, len(str(text)) // 4)  # Fallback estimation


@app.post("/v1/messages")
async def create_message(request: MessagesRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Format messages for Llama
        prompt = format_messages_for_llama(request.messages, request.system)

        # Count input tokens
        input_tokens = count_tokens(prompt)

        if request.stream:
            return StreamingResponse(
                stream_generate_response(request, prompt, input_tokens),
                media_type="text/event-stream",
            )
        else:
            return await generate_response(request, prompt, input_tokens)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/messages/count_tokens")
async def count_tokens_endpoint(request: TokenCountRequest):
    if tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Format messages for token counting
        prompt = format_messages_for_llama(request.messages, request.system)

        # Count tokens
        token_count = count_tokens(prompt)

        return {"input_tokens": token_count}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def generate_response(request: MessagesRequest, prompt: str, input_tokens: int):
    """Generate non-streaming response"""
    # Generate text
    # MLX generate function parameters
    response_text = generate_func(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=request.max_tokens,
        verbose=config.VERBOSE,
    )
    
    # Handle mlx_vlm's GenerationResult object return type
    if hasattr(response_text, "text"):
        response_text = response_text.text

    # Count output tokens
    output_tokens = count_tokens(response_text)

    # Create Claude-style response
    response = MessageResponse(
        id="msg_" + str(abs(hash(prompt)))[:8],
        content=[ContentBlockText(text=response_text)],
        model=request.model,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
    )

    return response


async def stream_generate_response(
    request: MessagesRequest, prompt: str, input_tokens: int
):
    """Generate streaming response with tool call interception"""
    response_id = "msg_" + str(abs(hash(prompt)))[:8]
    full_text = ""

    # Send message start event
    message_start = {
        "type": "message_start",
        "message": {
            "id": response_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": request.model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        },
    }
    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

    # Send content block start
    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

    state = "TEXT"
    buffer = ""
    tool_buffer = ""
    block_index = 0
    generated_tool = False
    in_text_block = True

    def get_text_delta(t: str):
        if not t: return ""
        d = {"type": "content_block_delta", "index": block_index, "delta": {"type": "text_delta", "text": t}}
        return f"event: content_block_delta\ndata: {json.dumps(d)}\n\n"

    for i, response in enumerate(stream_generate_func(model, tokenizer, prompt=prompt, max_tokens=request.max_tokens)):
        chunk = response.text
        full_text += chunk
        buffer += chunk

        while buffer:
            if state == "TEXT":
                idx = buffer.find("<")
                if idx == -1:
                    if not in_text_block:
                        block_index += 1
                        in_text_block = True
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                    yield get_text_delta(buffer)
                    buffer = ""
                else:
                    if idx > 0:
                        if not in_text_block:
                            block_index += 1
                            in_text_block = True
                            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                        yield get_text_delta(buffer[:idx])
                        buffer = buffer[idx:]
                    if len(buffer) < len("<|channel>thought"):
                        if "<|channel>thought".startswith(buffer) or "<|tool_call>".startswith(buffer):
                            break  # wait for more
                        else:
                            if not in_text_block:
                                block_index += 1
                                in_text_block = True
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                            yield get_text_delta(buffer[0])
                            buffer = buffer[1:]
                    else:
                        if buffer.startswith("<|channel>thought"):
                            buffer = buffer[len("<|channel>thought"):]
                            if not in_text_block:
                                block_index += 1
                                in_text_block = True
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                            yield get_text_delta("\n🤔 **Thinking:**\n")
                            state = "THOUGHT"
                        elif buffer.startswith("<|tool_call>"):
                            buffer = buffer[len("<|tool_call>"):]
                            state = "TOOL"
                            tool_buffer = ""
                        else:
                            if not in_text_block:
                                block_index += 1
                                in_text_block = True
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                            yield get_text_delta(buffer[0])
                            buffer = buffer[1:]
            elif state == "THOUGHT":
                idx = buffer.find("<channel|>")
                if idx == -1:
                    last_lt = buffer.rfind("<")
                    if last_lt == -1:
                        yield get_text_delta(buffer)
                        buffer = ""
                    else:
                        if "<channel|>".startswith(buffer[last_lt:]):
                            if last_lt > 0:
                                yield get_text_delta(buffer[:last_lt])
                            buffer = buffer[last_lt:]
                            break
                        else:
                            yield get_text_delta(buffer)
                            buffer = ""
                else:
                    if idx > 0:
                        yield get_text_delta(buffer[:idx])
                    buffer = buffer[idx + len("<channel|>"):]
                    yield get_text_delta("\n\n")
                    state = "TEXT"
            elif state == "TOOL":
                idx = buffer.find("<tool_call|>")
                if idx == -1:
                    last_lt = buffer.rfind("<")
                    if last_lt == -1:
                        tool_buffer += buffer
                        buffer = ""
                    else:
                        if "<tool_call|>".startswith(buffer[last_lt:]):
                            tool_buffer += buffer[:last_lt]
                            buffer = buffer[last_lt:]
                            break
                        else:
                            tool_buffer += buffer
                            buffer = ""
                else:
                    tool_buffer += buffer[:idx]
                    buffer = buffer[idx + len("<tool_call|>"):]
                    
                    import re
                    match = re.match(r'call:(\w+):?\s*(.*)', tool_buffer, re.DOTALL)
                    if match:
                        name = match.group(1)
                        args = match.group(2).strip()
                        if not args: args = "{}"
                        
                        if in_text_block:
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                            in_text_block = False
                            
                        block_index += 1
                        generated_tool = True
                        
                        import random
                        import string
                        tool_id = "toolu_01" + "".join(random.choices(string.ascii_letters + string.digits, k=16))
                        
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': name, 'input': {}}})}\n\n"
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'input_json_delta', 'partial_json': args}})}\n\n"
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                        
                    state = "TEXT"
                    tool_buffer = ""

    if buffer and state == "TEXT":
        if not in_text_block:
            block_index += 1
            in_text_block = True
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        yield get_text_delta(buffer)
    elif state == "THOUGHT" and buffer:
        yield get_text_delta(buffer)

    if in_text_block:
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"

    output_tokens = count_tokens(full_text)
    stop_reason = "tool_use" if generated_tool else "end_turn"
    
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}})}\n\n"
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/")
async def root():
    return {
        "message": "Claude Code MLX Proxy",
        "status": "running",
        "model_loaded": model is not None,
    }


if __name__ == "__main__":
    print(f"Starting Claude Code MLX Proxy on {config.HOST}:{config.PORT}")
    uvicorn.run(app, host=config.HOST, port=config.PORT)
