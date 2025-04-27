from flask import Flask
from routes.upload_pdf_to_vectorDB import upload_bp
from routes.query_route import query_blueprint  


app = Flask(__name__)


app.register_blueprint(upload_bp, url_prefix="/api")
app.register_blueprint(query_blueprint)

if __name__ == "__main__":
    app.run(debug=True)
