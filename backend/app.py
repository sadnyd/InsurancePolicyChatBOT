from flask import Flask, jsonify, request
from flask_cors import CORS
from routes.upload_pdf_to_vectorDB import upload_bp
from routes.query_route import query_blueprint  


app = Flask(__name__)
CORS(app) 


app.register_blueprint(upload_bp)
app.register_blueprint(query_blueprint)

@app.route("/", methods=["GET"])
def home():
    api_info = {
        "api_version": "1.0",
        "description": "This API allows you to upload PDF files and query them using a machine learning model.",
        "endpoints": {
            "/api/upload": {
                "method": "POST",
                "description": "Upload a PDF file to be processed and stored in the vector database."
            },
            "/query": {
                "method": "POST",
                "description": "Query the vector store using a question. The model will return an answer based on the stored content."
            }
        }
    }
    return jsonify(api_info)

if __name__ == "__main__":
    app.run(debug=True)
