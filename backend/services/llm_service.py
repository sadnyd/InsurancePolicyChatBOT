import os
import dotenv
from google import genai
from google.genai import types

class GeminiContextualQA:
    def __init__(self, top_k=5, model="gemini-2.0-flash", system_instruction=None):
        # Load API Key
        dotenv.load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")

        # Initialize Gemini Client
        self.client = genai.Client(api_key=api_key)
        self.top_k = top_k
        self.model = model
        self.system_instruction = system_instruction or "You are a AI assistant that answers Insurance Policy Qusetions."

    def build_prompt(self, context_chunks, user_query):
        context_text = "\n\n".join(context_chunks[:self.top_k])
        prompt = f"""Here is some background context:
{context_text}

User question: {user_query}

Please provide a detailed and helpful response."""
        return prompt

    def ask(self, match_results, user_query):
        context = [item['text'] if isinstance(item, dict) else item for item in match_results]
        prompt_text = self.build_prompt(context, user_query)

        response = self.client.models.generate_content(
            model=self.model,
            config=types.GenerateContentConfig(
                system_instruction=self.system_instruction
            ),
            contents=prompt_text
        )

        return response.text

