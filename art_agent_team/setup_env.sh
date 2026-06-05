#!/bin/bash

# Set environment variables for API keys
export GOOGLE_APPLICATION_CREDENTIALS="/Users/nathaniel.cannon/Documents/VScodeWork/Art_AI/API_keys/geministudioapi-b5da91c0cd01.json"

# Set the Gemini Studio API key that works for all services
export GOOGLE_API_KEY="AIzaSyC3eRhKZlrio2FfCjNdXOrEBKQ3EYMqGrY"

# Configure API scopes
export GOOGLE_CLOUD_API_SCOPES="https://www.googleapis.com/auth/cloud-platform"

# Verify environment variables
echo "Checking environment setup..."
echo "GOOGLE_APPLICATION_CREDENTIALS is set to: $GOOGLE_APPLICATION_CREDENTIALS"
echo "GOOGLE_API_KEY is set: $(if [ ! -z "$GOOGLE_API_KEY" ]; then echo "Yes (using GeminiStudioAPI key)"; else echo "No"; fi)"

# Check if files exist
if [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "Error: Google credentials file not found at $GOOGLE_APPLICATION_CREDENTIALS"
    exit 1
fi

# Verify Google credentials file permissions
if [ -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    chmod 600 "$GOOGLE_APPLICATION_CREDENTIALS"
    echo "Set secure permissions on Google credentials file"
fi

echo "Environment setup complete."