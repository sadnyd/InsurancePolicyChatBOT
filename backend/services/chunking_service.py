import nltk
import tiktoken
from typing import List

nltk.download("punkt")  
nltk.download('punkt_tab')


class ChunkingService:
    def __init__(self, model_name: str = "gpt-3.5-turbo", max_tokens: int = 2000, overlap: int = 200):
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.tokenizer = tiktoken.encoding_for_model(model_name)

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def split_text_semantically(self, text: str) -> List[str]:
        """Split text into semantically coherent chunks using sentence boundaries."""
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            # If sentence itself is too long, force split
            if sentence_tokens > self.max_tokens:
                continue

            # If adding sentence would exceed max_tokens
            if current_tokens + sentence_tokens > self.max_tokens:
                # Finish current chunk
                chunks.append(" ".join(current_chunk))

                # Apply overlap
                if self.overlap > 0:
                    overlap_tokens = 0
                    overlap_chunk = []
                    for s in reversed(current_chunk):
                        t = self._count_tokens(s)
                        if overlap_tokens + t <= self.overlap:
                            overlap_chunk.insert(0, s)
                            overlap_tokens += t
                        else:
                            break
                    current_chunk = overlap_chunk
                    current_tokens = sum(self._count_tokens(s) for s in current_chunk)
                else:
                    current_chunk = []
                    current_tokens = 0

            # Add the sentence
            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
