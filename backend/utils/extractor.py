import fitz  

class PDFTextExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract_text(self, verbose=False):
        text = ""
        with fitz.open(self.pdf_path) as doc:
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if verbose:
                    print(f"--- Extracting Page {page_num + 1} ---")
                text += page_text + "\n"
        return text
