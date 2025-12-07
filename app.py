import os
import numpy as np
import pandas as pd
import recommendation
from rapidfuzz import process, fuzz
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ROUTES
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    query = request.form["query"]
    mode = int(request.form["mode"])

    return jsonify(recommendation.recommend_book(query, mode))

if __name__ == "__main__":
    app.run(debug=True)
