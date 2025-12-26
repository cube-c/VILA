import argparse
import base64
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from typing import List, Literal, Optional, Union, get_args

import requests
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image as PILImage
from pydantic import BaseModel

from fastapi import FastAPI

from llava.mm_utils import get_model_name_from_path
from llava.utils import disable_torch_init
import llava
import asyncio 
from anyio.lowlevel import RunVar
from anyio import CapacityLimiter

class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class MediaURL(BaseModel):
    url: str


class ImageContent(BaseModel):
    type: Literal["image_url"]
    image_url: MediaURL


class VideoContent(BaseModel):
    type: Literal["video_url"]
    video_url: MediaURL
    frames: Optional[int] = 8


def semaphore(value: int):
    """Decorator to limit the number of concurrent executions of an async function."""
    sem = asyncio.Semaphore(value)
    def decorator(func):
        async def wrapper(*args, **kwargs):
            async with sem:
                return await func(*args, **kwargs)
        return wrapper
    return decorator

IMAGE_CONTENT_BASE64_REGEX = re.compile(r"^data:image/(png|jpe?g);base64,(.*)$")
VIDEO_CONTENT_BASE64_REGEX = re.compile(r"^data:video/(mp4);base64,(.*)$")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Union[TextContent, ImageContent, VideoContent]]]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    # these params are not actually used by NVILA
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 0.9
    temperature: Optional[float] = 0.2
    use_cache: Optional[bool] = True
    num_beams: Optional[int] = 1
    # fastapi 
    client: Optional[dict] = None


model = None
model_name = None
tokenizer = None
image_processor = None
context_len = None

def get_timestamp():
    return int(time.time())


def load_image(image_url: str) -> PILImage:
    if image_url.startswith("http") or image_url.startswith("https"):
        response = requests.get(image_url)
        image = PILImage.open(BytesIO(response.content)).convert("RGB")
    else:
        match_results = IMAGE_CONTENT_BASE64_REGEX.match(image_url)
        if match_results is None:
            raise ValueError(f"Invalid image url: {image_url[:64]}")
        image_base64 = match_results.groups()[1]
        image = PILImage.open(BytesIO(base64.b64decode(image_base64))).convert("RGB")
    return image


def get_literal_values(cls, field_name: str):
    field_type = cls.__annotations__.get(field_name)
    if field_type is None:
        raise ValueError(f"{field_name} is not a valid field name")
    if hasattr(field_type, "__origin__") and field_type.__origin__ is Literal:
        return get_args(field_type)
    raise ValueError(f"{field_name} is not a Literal type")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_name, tokenizer, image_processor, context_len
    disable_torch_init()
    model_path = app.args.model_path
    model_name = get_model_name_from_path(model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name, None)
    model = llava.load(model_path)
    # model = None
    print(f"{model_name=} {model_path=} loaded successfully. Context length: {context_len}")
    print("start & set capacity limiter to 1")
    RunVar("_default_thread_limiter").set(CapacityLimiter(1))
    global globallock
    globallock = asyncio.Lock()
    yield


app = FastAPI(lifespan=lifespan)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the VILA API. This is for internal use only. Please use /chat/completions for chat completions."}

        
@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # print("DEBUG0")
    current_time = time.strftime("%H:%M:%S-%s", time.localtime())
    current_time_hash = uuid.uuid5(uuid.NAMESPACE_DNS, current_time)
    print("[Req recv]", current_time_hash, current_time, request.dict().keys())
    try:
        global model, tokenizer, image_processor, context_len

        if request.model != model_name:
            raise ValueError(
                f"The endpoint is configured to use the model {model_name}, "
                f"but the request model is {request.model}"
            )

        ########################################################################### 
        prompt = []
        messages = request.messages
        for message in messages:
            if isinstance(message.content, str):
                prompt.append(message.content)

            if isinstance(message.content, list):
                for content in message.content:
                    print(content.type)
                    if content.type == "text":
                        prompt.append(content.text)
                    elif content.type == "image_url":
                        image = load_image(content.image_url.url)
                        prompt.append(image)
                    else:
                        raise NotImplementedError(f"Unsupported content type: {content.type}")
        
        with torch.inference_mode():
            await globallock.acquire()
            outputs = model.generate_content(prompt)
            # outputs = "helloworld!" 
            if globallock.locked():
                globallock.release()
            print("\nAssistant: ", outputs)
            resp_content = outputs
            return {
                "id": uuid.uuid4().hex,
                "object": "chat.completion",
                "created": get_timestamp(),
                "model": request.model,
                "index": 0,
                "choices": [
                    {"message": ChatMessage(role="assistant", content=resp_content)}
                ],
            }
    except Exception as e:
        if globallock.locked():
            globallock.release()
            
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
    finally:
        pass
    
if __name__ == "__main__":
    global host, port
    host = os.getenv("VILA_HOST", "0.0.0.0")
    port = os.getenv("VILA_PORT", 8000)
    model_path = os.getenv("VILA_MODEL_PATH", "Efficient-Large-Model/NVILA-8B")
    conv_mode = os.getenv("VILA_CONV_MODE", "auto")
    workers = os.getenv("VILA_WORKERS", 1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=host)
    parser.add_argument("--port", type=int, default=port)
    parser.add_argument("--model-path", type=str, default=model_path)
    parser.add_argument("--conv-mode", type=str, default=conv_mode)
    app.args = parser.parse_args()
    port = int(app.args.port)
    uvicorn.run(app, 
        host = app.args.host, 
        port = app.args.port, 
        workers = 1,
        timeout_keep_alive = 60,
    )
