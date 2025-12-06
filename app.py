from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import os
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

app = Flask(__name__)

# ------------------------
# LOAD DATA & MODEL
# ------------------------
csv_file = "./data/processed_data.csv"
df = pd.read_csv(csv_file)

descriptionTxt = df["description"].astype(str).tolist()
titleTxt = df["title"].astype(str).tolist()

model = SentenceTransformer("all-MiniLM-L6-v2")

authorTxt = df["authors"].apply(lambda authors: " ".join(eval(authors))).tolist()
genreTxt  = df["genre_list"].apply(lambda genres: " ".join(eval(genres))).tolist()

vector_files = {
    "description": "./data/vectors/vectorsDes.npy",
    "author": "./data/vectors/vectorsAuthor.npy",
    "genre": "./data/vectors/vectorsGenre.npy",
}

if all(os.path.exists(f) for f in vector_files.values()):
    vectorsDes = np.load(vector_files["description"])
    vectorsAuthor = np.load(vector_files["author"])
    vectorsGenre = np.load(vector_files["genre"])
else:
    vectorsDes = model.encode(descriptionTxt, batch_size=256, show_progress_bar=True)
    vectorsAuthor = model.encode(authorTxt, batch_size=256, show_progress_bar=True)
    vectorsGenre = model.encode(genreTxt, batch_size=256, show_progress_bar=True)

    np.save(vector_files["description"], vectorsDes)
    np.save(vector_files["author"], vectorsAuthor)
    np.save(vector_files["genre"], vectorsGenre)

# weights
desc_weight = 1.0
genre_weight = 0.3
author_weight = 0.6

combined_vectors = (
    vectorsDes * desc_weight +
    vectorsGenre * genre_weight +
    vectorsAuthor * author_weight
)
combined_vectors = normalize(combined_vectors)

def get_top_n_similar(vector_list, target_vector, top_n=10, exclude_index=None):
    scores = cosine_similarity([target_vector], vector_list)[0]
    if exclude_index is not None:
        scores[exclude_index] = -1
    top_indices = np.argsort(scores)[-top_n:][::-1]
    return [(i, scores[i]) for i in top_indices]


# ------------------------
# ROUTES
# ------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
    query = request.form["query"]

    # First match by title
    result = process.extractOne(query, titleTxt, scorer=fuzz.token_sort_ratio)

    if result is not None:
        matched_title, score, row_index = result
        target_vector = combined_vectors[row_index]
    else:
        qv = model.encode([query])[0]
        target_vector = normalize(qv.reshape(1, -1))[0]
        row_index = None
        matched_title, score = query, None

    top_similar = get_top_n_similar(
        combined_vectors, target_vector,
        top_n=10,
        exclude_index=row_index
    )

    recommendations = []
    for i, sim_score in top_similar:
        recommendations.append({
            "title": df["title"][i],
            "author": df["authors"][i],
            "description": df["description"][i],
            "year": df["year"][i] if "year" in df.columns else "",
            "thumbnail": df["thumbnail"][i] if "thumbnail" in df.columns else "",
            "similarity": float(sim_score)
        })

    return jsonify({
        "best_match": matched_title,
        "best_score": float(score) if score else None,
        "recommendations": recommendations
    })


if __name__ == "__main__":
    app.run(debug=True)
