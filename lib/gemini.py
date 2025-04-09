import base64
import os
from google import genai
from google.genai import types

MODEL = "gemini-2.0-flash"


def generate(prompt, json=False):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        # temperature=1,
        temperature=0.7,
        top_p=0.95,
        top_k=64,
        max_output_tokens=8192,
        # max_output_tokens=65536,
        response_mime_type="application/json" if json else "text/plain",
    )
    res = ''
    for chunk in client.models.generate_content_stream(
        model=MODEL,
        contents=contents,
        config=generate_content_config,
    ):
        # print(chunk.text, end="")
        res += chunk.text or ''
    return res


__all__ = [
    "generate",
]
