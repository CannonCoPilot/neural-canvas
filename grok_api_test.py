from openai import OpenAI
    
client = OpenAI(
  api_key="xai-LyLp6YhsAL1Q1PSbTCGNiehsMivO2HQJ1gRf7REea2Kj6xk40F3ek62Ka0vgjQ2wuYN8dFGl57NdjKon",
  base_url="https://api.x.ai/v1",
)

completion = client.chat.completions.create(
  model="grok-3-beta",
  messages=[
    {"role": "user", "content": "What is the meaning of life?"}
  ]
)
print(completion.choices[0].message.content)