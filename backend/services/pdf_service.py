import os
from utils.extractor import PDFTextExtractor

class PDFExtractionService:
    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def extract_and_save(self, pdf_path):
        filename = os.path.splitext(os.path.basename(pdf_path))[0] + ".txt"
        output_path = os.path.join(self.data_dir, filename)

        extractor = PDFTextExtractor(pdf_path)
        text = extractor.extract_text(verbose=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Text saved to {output_path}")
        return output_path
