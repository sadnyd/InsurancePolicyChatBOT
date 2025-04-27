from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os

from services.pdf_loader_service import PDFLoaderService
from services.chunking_service import ChunkingService
from services.embedding_service import EmbeddingService
from services.vector_store_service import VectorStoreService

upload_bp = Blueprint("upload_bp", __name__)

chunking_service = ChunkingService()
embedding_service = EmbeddingService()
vector_store_service = VectorStoreService()

UPLOAD_FOLDER = './uploads'  
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@upload_bp.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            # Save the file temporarily to disk
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Now process the file as a regular path
            text = PDFLoaderService.load_pdf(filepath)

            chunks = chunking_service.split_text_semantically(text)

            embeddings = embedding_service.get_embeddings(chunks)

            vector_store_service.store(chunks, embeddings)

            # Optionally, remove the file after processing
            os.remove(filepath)

            return jsonify({"message": f"Successfully processed and stored {len(chunks)} chunks."}), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Unknown error"}), 500
