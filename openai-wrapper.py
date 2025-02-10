import os
import requests
import json
from typing import Optional, Dict, Any, List

class OpenAIWrapper:
    """
    A wrapper class for interacting with OpenAI's API
    """
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the wrapper with an API key
        Args:
            api_key: OpenAI API key. If not provided, will look for OPENAI_API_KEY environment variable
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided either directly or through OPENAI_API_KEY environment variable")
        
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create a chat completion
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: The model to use
            temperature: Controls randomness (0-1)
            max_tokens: Maximum number of tokens to generate
        Returns:
            API response as a dictionary
        """
        endpoint = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
            
        try:
            response = requests.post(endpoint, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")

    def create_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        n: int = 1,
        response_format: str = "url"
    ) -> Dict[str, Any]:
        """
        Create an image using DALL-E
        Args:
            prompt: The image description
            size: Image size (1024x1024, 512x512, or 256x256)
            n: Number of images to generate
            response_format: 'url' or 'b64_json'
        Returns:
            API response as a dictionary
        """
        endpoint = f"{self.base_url}/images/generations"
        
        payload = {
            "prompt": prompt,
            "size": size,
            "n": n,
            "response_format": response_format
        }
        
        try:
            response = requests.post(endpoint, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Initialize the wrapper
    openai = OpenAIWrapper()
    
    # Example chat completion
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    try:
        response = openai.chat_completion(messages)
        print("Chat response:", response['choices'][0]['message']['content'])
    except Exception as e:
        print(f"Error: {e}")
    
    # Example image generation
    try:
        response = openai.create_image("A beautiful sunset over Paris")
        print("Image URL:", response['data'][0]['url'])
    except Exception as e:
        print(f"Error: {e}")
