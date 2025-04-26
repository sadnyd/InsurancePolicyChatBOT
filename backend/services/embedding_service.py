import os
from google import genai
from google.genai import types
from dotenv import load_dotenv


load_dotenv()

class EmbeddingService:
    def __init__(self, model_name: str = "models/text-embedding-004"):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("❌ GEMINI_API_KEY not found in environment variables.")
        
        self.model_name = model_name
        self.client = genai.Client(api_key=self.api_key)

    def get_embeddings(self, texts: list[str], task_type: str = "SEMANTIC_SIMILARITY") -> list[list[float]]:
        embeddings = []
        for text in texts:
            try:
                response = self.client.models.embed_content(
                            contents=text,
                            model=self.model_name
                            )
                embeddings.append(response.embeddings[0].values)

            except Exception as e:
                print(f"Error embedding text: {e}")
                embeddings.append([])
        return embeddings
