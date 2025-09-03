from typing import Callable, ParamSpec, TypeVar

from anthropic.resources import AsyncMessages

P2 = ParamSpec("P2")
R2 = TypeVar("R2")


def call_twice(func: Callable[P2, R2], *args: P2.args, **kwargs: P2.kwargs) -> tuple[R2, R2]:
    first = func(*args, **kwargs)
    second = func(*args, **kwargs)
    return first, second


# Prepare messages with system prompt and image
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "tell me whats in this image"
            },
            {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": "https://images.unsplash.com/photo-1502082553048-f009c37129b9?w=800"
                }
            }
        ]
    }
]

system_prompt = "You are a helpful assistant that describes images in detail."

result = call_twice(AsyncMessages.create,             model="claude-3-5-sonnet-20241022",
            system=system_prompt,
            messages=messages,
            max_tokens="1000")
