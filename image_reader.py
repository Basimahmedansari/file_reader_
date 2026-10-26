
import os
from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def analyze_image(file, question):
    response = client.models.generate_content(
        model=os.getenv("GEMINI_VISION_MODEL", "gemini-3.6-flash"),
        contents=[question, types.Part.from_bytes(data=file.getvalue(), mime_type=file.type)]
    )
    return response.text
