from flask import Flask
from flask_cors import CORS
from app.routes.semantic_search import bp as semantic_search_bp

def createApp():
    app = Flask(__name__)
    
    CORS(app, origins=["*"])

    app.register_blueprint(semantic_search_bp, url_prefix='/semantic-search')

    return app