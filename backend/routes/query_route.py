from flask import Blueprint, request, jsonify
from services.vector_store_service import VectorStoreService
from services.embedding_service import EmbeddingService
from services.llm_service import GeminiContextualQA  

query_blueprint = Blueprint("query_route", __name__)

# Initialize Services
vector_service = VectorStoreService()
embedding_service = EmbeddingService()
qa_service = GeminiContextualQA()

@query_blueprint.route("/query", methods=["POST"])
def query_vector_store():
    try:
        data = request.json
        user_query = data.get("query")

        if not user_query:
            return jsonify({"error": "Missing 'query' field in JSON body."}), 400

        query_embedding = embedding_service.get_embeddings([user_query])[0]  # First (and only) embedding

        search_results = vector_service.search(query_embedding, top_k=qa_service.top_k)

        context_chunks = [match['metadata']['chunk_text'] for match in search_results]

        answer = qa_service.ask(context_chunks, user_query)

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
