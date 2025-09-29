import google.generativeai as genai
import os


genai.configure(api_key="your_api")
def check_gemini_api_status():
    """
    Checks if the Gemini API is working by listing the available models.
    """
    try:
        # This is a lightweight call that doesn't generate content but checks connectivity.
        models = genai.list_models()
        model_names = [m.name for m in models]

        if model_names:
            print("Successfully connected to the Gemini API!")
            print("Available models:")
            for model_name in model_names:
                print(f"- {model_name}")
            return True
        else:
            print("Could not retrieve model list. The API might be down or there's an issue.")
            return False

    except Exception as e:
        print(f"An error occurred: {e}")
        print("This could be due to an invalid API key, network issues, or a service outage.")
        return False

if __name__ == "__main__":
    check_gemini_api_status()
