from services.pdf_loader_service import PDFLoaderService
from services.chunking_service import ChunkingService
from services.embedding_service import EmbeddingService
from services.vector_store_service import VectorStoreService
from services.llm_service import GeminiContextualQA  

def main():
    pdf_path = "test.pdf"
    loader = PDFLoaderService()
    text = loader.load_pdf(pdf_path)

    if not text.strip():
        print("❌ Failed to extract text from PDF.")
        return
    else:
        print(f"✅ Extracted text of length {len(text)} characters.")

    chunker = ChunkingService(max_tokens=2000, overlap=200)
    chunks = chunker.split_text_semantically(text)

    if not chunks:
        print("❌ Chunking failed or produced no chunks.")
        return

    print(f"✅ Successfully split into {len(chunks)} chunks.")

    for idx, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {idx+1} ---\n")
        print(chunk[:500])
        print("...")

    embedding_service = EmbeddingService()
    embeddings = embedding_service.get_embeddings(chunks)

    if not embeddings or not embeddings[0]:
        print("❌ Embedding generation failed.")
        return

    print(f"✅ Successfully generated {len(embeddings)} embeddings.")
    print(f"\n--- Embedding Preview (first chunk) ---\n")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(embeddings[0][:10])

    # -----------------------------
    # 🧠 Store embeddings into Pinecone
    # -----------------------------
    print("\n🚀 Storing embeddings into Pinecone...")
    vector_store = VectorStoreService()
    vector_store.store(chunks, embeddings)
    print("✅ Stored embeddings successfully!")

    # -----------------------------
    # 🔎 Search test
    # -----------------------------
    print("\n🔎 Running a sample search...")
    sample_query = "Summary of the document"  # You can change
    sample_query_embedding = embedding_service.get_embeddings([sample_query])[0]
    matches = vector_store.search(sample_query_embedding)

    if matches:
        print("\n✅ Search results:")
        for match in matches:
            print(f"- ID: {match['id']} | Score: {match['score']:.4f}")
            print(f"  Chunk: {match['metadata']['chunk_text'][:200]}...")
    else:
        print("❌ No matches found.")
        return  # No point going ahead if no matches

    # -----------------------------
    # ✨ NEW: Run Query against Gemini!
    # -----------------------------
    print("\n🤖 Running the query with Gemini...")
    qa_service = GeminiContextualQA(top_k=5, system_instruction="You are a helpful AI assistant for summarizing documents.")
    user_query = "Can you summarize the main points of this document?"

    response = qa_service.ask(
        match_results=[{"text": match['metadata']['chunk_text']} for match in matches],
        user_query=user_query
    )

    print("\n✅ Gemini Response:")
    print(response)

if __name__ == "__main__":
    main()
