import os
from openai import OpenAI
import os
import base64

XAI_API_KEY = "xai-LyLp6YhsAL1Q1PSbTCGNiehsMivO2HQJ1gRf7REea2Kj6xk40F3ek62Ka0vgjQ2wuYN8dFGl57NdjKon"
image_path = "/Users/nathaniel.cannon/Documents/VScodeWork/Art_AI/art_agent_team/tests/test_data/input/Emile Clause - Summer morning 1891.jpeg"

client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

# Getting the base64 string
base64_image = encode_image(image_path)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high",
                },
            },
            {
                "type": "text",
                "text": "What's in this image?",
            },
        ],
    },
]

completion = client.chat.completions.create(
    model="grok-2-vision-1212",
    messages=messages,
    temperature=0.01,
)

print(completion.choices[0].message.content)