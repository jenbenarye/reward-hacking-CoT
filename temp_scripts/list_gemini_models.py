import litellm
import os

# Make sure GOOGLE_API_KEY is set
# os.environ["GOOGLE_API_KEY"] = "your-api-key"  # or it should already be set

# List available Gemini models
models = litellm.get_valid_models(
    custom_llm_provider="gemini",
    check_provider_endpoint=True
)

print("Available Gemini models:")
for model in models:
    print(f"  - {model}")