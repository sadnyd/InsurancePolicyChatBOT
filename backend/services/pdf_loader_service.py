import os
from typing import List
import fitz  

class PDFLoaderService:
    @staticmethod
    def load_pdf(file_path: str) -> str:
        """
        Load a single PDF and extract text.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            doc = fitz.open(file_path)
            text = ""

            for page in doc:
                page_text = page.get_text("text")
                if page_text:
                    text += page_text + "\n"

            doc.close()
            return text.strip()

        except Exception as e:
            raise RuntimeError(f"Error while reading PDF {file_path}: {str(e)}")

    @staticmethod
    def load_multiple_pdfs(file_paths: List[str]) -> List[str]:
        """
        Load multiple PDFs and extract texts as a list.
        Each entry in the list corresponds to one PDF.
        """
        texts = []
        for path in file_paths:
            pdf_text = PDFLoaderService.load_pdf(path)
            texts.append(pdf_text)
        return texts
