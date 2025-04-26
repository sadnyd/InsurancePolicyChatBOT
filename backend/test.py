from services.pdf_loader_service import PDFLoaderService
from services.chunking_service import ChunkingService

def main():
    # 1. Load PDF text
    pdf_path = "test.pdf"  
    loader = PDFLoaderService()
    text = loader.load_pdf(pdf_path)

    if not text.strip():
        print("❌ Failed to extract text from PDF.")
        return
    else:
        print(f"✅ Extracted text of length {len(text)} characters.")

    # 2. Chunk the extracted text
    chunker = ChunkingService(max_tokens=500, overlap=50)
    chunks = chunker.split_text_semantically(text)

    if not chunks:
        print("❌ Chunking failed or produced no chunks.")
        return

    print(f"✅ Successfully split into {len(chunks)} chunks.")

    # 3. Print a preview
    for idx, chunk in enumerate(chunks[:3]):  # Show only first 3 chunks
        print(f"\n--- Chunk {idx+1} ---\n")
        print(chunk[:500])  # Show first 500 chars of each chunk
        print("...")

if __name__ == "__main__":
    main()
