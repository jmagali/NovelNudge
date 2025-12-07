from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import os
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import recommendation

app = Flask(__name__)

# ROUTES
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    query = request.form["query"]
    btn = int(request.form["btn"])

    if btn == 1:
        # covert returned object into JSON
        return jsonify(recommendation.recommend_from_title(query))
    else:
        return jsonify(recommendation.recommend_from_query(query))


if __name__ == "__main__":
    app.run(debug=True)
