import os
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

load_dotenv()

class VectorStoreService:
    def __init__(self):
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = "policy-index"
        self.dimension = 768  
        self.cloud = "aws"
        self.region = "us-east-1"

        
        self.pc = Pinecone(api_key=self.pinecone_api_key)

        # Create index if not exists
        if not self.pc.has_index(self.index_name):
            self.pc.create_index(
                name=self.index_name,
                vector_type="dense",
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self.cloud,
                    region=self.region
                ),
                deletion_protection="disabled",
                tags={"environment": "development"}
            )

        # Connect to the index
        self.index = self.pc.Index(self.index_name)

    def store(self, chunks, embeddings):
        """
        Store your own embeddings + chunk metadata into the index.
        
        Args:
            chunks (List[str]): List of text chunks.
            embeddings (List[EmbeddingObject or List[float]]): Corresponding embeddings.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match.")

        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = f"chunk-{i}"
            metadata = {"chunk_text": chunk}

            # If embedding is an object, extract its values
            if hasattr(embedding, "values"):
                embedding = embedding.values  # Take the raw list

            vectors.append({
                "id": vector_id,
                "values": embedding,   # now it will be a list of floats
                "metadata": metadata
            })

        self.index.upsert(vectors=vectors)
        print(f"✅ Successfully upserted {len(vectors)} embeddings into the index.")

    def search(self, query_embedding, top_k=5):
        """
        Search the index using your own embedding vector.
        
        Args:
            query_embedding (List[float]): Embedding of the query text.
            top_k (int): Number of top matches to retrieve.
        
        Returns:
            List[Dict]: Matching vectors with metadata.
        """
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results["matches"]

    def delete_all(self):
        """ Delete all vectors from the index (use carefully). """
        self.index.delete(delete_all=True)
        print(" All vectors deleted from the index.")

