import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# --- Configuration ---
# Prioritize environment variable, fall back to direct input if needed
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("OpenRouter API key not found in environment variables.")
    # Optional: Prompt user or use a hardcoded placeholder for quick testing
    # OPENROUTER_API_KEY = input("Please enter your OpenRouter API key: ")
    OPENROUTER_API_KEY = "Sk-or-v1-b5cf81f20c945787cbd1da847bcb52058987b3c8b9bc85c124001447203f3c8f" # Replace if testing directly

# Model to test (use a free one)
# TEST_MODEL = "meta-llama/llama-4-maverick:free" # Original
TEST_MODEL = "google/gemini-2.0-flash-exp:free" # Testing Gemini Flash 2.0
# TEST_MODEL = "qwen/qwen2.5-vl-32b-instruct:free" # Uncomment to test Qwen

# Simple test prompt
TEST_PROMPT = "Who are you?"

# --- API Call ---
print(f"Attempting to call OpenRouter model: {TEST_MODEL}")
print(f"Using API Key starting with: {OPENROUTER_API_KEY[:5]}..." if OPENROUTER_API_KEY else "No API Key found!")

if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
    print("\nERROR: API Key is missing or is the placeholder value.")
    print("Please set the OPENROUTER_API_KEY environment variable or replace the placeholder in the script.")
else:
    try:
        # Initialize client
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
            # Optional: Add default headers if needed, e.g., for referrer
            # default_headers={"HTTP-Referer": "YOUR_SITE_URL", "X-Title": "YOUR_APP_NAME"},
        )

        # Make the API call
        completion = client.chat.completions.create(
            model=TEST_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": TEST_PROMPT},
            ],
            max_tokens=50,
        )

        # Print the result
        print("\n--- API Call Successful ---")
        print(f"Response from {TEST_MODEL}:")
        if completion.choices:
            print(completion.choices[0].message.content)
        else:
            print("No choices returned.")
        # print("\nFull Completion Object:")
        # print(completion)

    except openai.AuthenticationError as e:
        print("\n--- API Call FAILED: Authentication Error ---")
        print(f"Error Code: {e.status_code}")
        print(f"Error Message: {e.body.get('error', {}).get('message', 'N/A') if e.body else 'N/A'}")
        print("Please verify your OpenRouter API key and ensure it's correctly set.")
    except openai.APIConnectionError as e:
        print("\n--- API Call FAILED: Connection Error ---")
        print(f"Error: {e}")
        print("Check your network connection and the OpenRouter API status.")
    except openai.RateLimitError as e:
        print("\n--- API Call FAILED: Rate Limit Error ---")
        print(f"Error: {e}")
        print("You may have exceeded the rate limits for the free models.")
    except openai.APIStatusError as e:
        print(f"\n--- API Call FAILED: Status Error {e.status_code} ---")
        print(f"Error Response: {e.response}")
        print("Check the model name and OpenRouter API status.")
    except Exception as e:
        print("\n--- API Call FAILED: An unexpected error occurred ---")
        print(f"Error: {e}")