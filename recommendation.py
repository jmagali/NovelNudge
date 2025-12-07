import os
import sys
import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity # Finds the angle between vectors through dot product
from sklearn.preprocessing import normalize # Finds the unit vector of a vector

# Functions
def get_top_n_similar(vector_list, target_vector, top_n=10, exclude_index=None):
    # Given a target vector, return top N most similar row indices and similarity scores.
    scores = cosine_similarity([target_vector], vector_list)[0]
    
    # If exclude_index is given, set its score very low so it won't appear
    if exclude_index is not None:
        scores[exclude_index] = -1 
    
    top_indices = np.argsort(scores)[-top_n:][::-1]  # highest first
    return [(i, scores[i]) for i in top_indices]

def recommend_book(user_input, mode):
    # PRE-COMPUTATION
    csv_file = "./data/processed_data.csv"
    description_column = "description"
    title_column = "title"

    df = pd.read_csv(csv_file)
    descriptionTxt = df[description_column].astype(str).tolist()
    titleTxt = df[title_column].astype(str).tolist()

    # Get reranker model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Combine authors into one string per book
    authorTxt = df['authors'].apply(lambda authors: " ".join(eval(authors)) if isinstance(authors, str) else "").tolist()

    # Combine genres into one string per book
    genreTxt = df['genre_list'].apply(lambda genres: " ".join(eval(genres)) if isinstance(genres, str) else "").tolist()

    vector_files = {
        "description": "./data/vectors/vectorsDes.npy",
        "author": "./data/vectors/vectorsAuthor.npy",
        "genre": "./data/vectors/vectorsGenre.npy",
        "combined": "./data/vectors/combined_vectors.npy"
    }

    # Load vectors if they exist
    if os.path.exists("./data/vectors/combined_vectors.npy"):
        print("Loading combined vectors from disk...")
        combined_vectors = np.load(vector_files["combined"])
    else:
        print("Encoding vectors, this may take some time...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Create vectors
        vectorsDes = model.encode(descriptionTxt, batch_size=256, show_progress_bar=True)
        vectorsAuthor = model.encode(authorTxt, batch_size=256, show_progress_bar=True)
        vectorsGenre = model.encode(genreTxt, batch_size=256, show_progress_bar=True)

        # Save vectors for future runs
        np.save(vector_files["description"], vectorsDes)
        np.save(vector_files["author"], vectorsAuthor)
        np.save(vector_files["genre"], vectorsGenre)
        print("Vectors saved to disk.")

        # Vector weights
        desc_weight = 1.0
        genre_weight = 0.3  # very small weight for genre
        author_weight = 0.6 # small weight for author

        # Sum the vectors
        combined_vectors = (vectorsDes * desc_weight + vectorsGenre * genre_weight + vectorsAuthor * author_weight)

        # Normalize combined vectors (turn into unit vector equivalent)
        combined_vectors = normalize(combined_vectors)
        np.save("./data/vectors/combined_vectors.npy", combined_vectors)
    
    # 1: title mode; 2: description mode
    if mode == 1:
        user_input = user_input.strip().lower()
        
        # Use RapidFuzz for title matching
        result = process.extractOne(user_input, titleTxt, scorer=fuzz.token_sort_ratio, score_cutoff=60)
        
        if result is None:
            return {"error": "No close match found."}
        
        matched_title, score, row_index = result
        target_vector = combined_vectors[row_index]
        
        # Find similar books
        top_similar = get_top_n_similar(combined_vectors, target_vector, top_n=10,
                                    exclude_index=row_index)
        
        # Build JSON payload
        recommendations = []
        for i, similarity in top_similar:
            row = df.iloc[i]

            # Fix subtitle handling
            subtitle = row["subtitle"]
            if isinstance(subtitle, float) or pd.isna(subtitle):
                subtitle = ""
            else:
                subtitle = subtitle.title()

            title = row["title"].title()
            full_title = f"{title}: {subtitle}".strip(": ")

            recommendations.append({
                "title": full_title,
                "author": ", ".join([author.title() for author in eval(row["authors"])]),
                "description": safe_val(row.get("description", "")),
                "thumbnail": safe_val(row.get("thumbnail", "")),
                "year": safe_val(row.get("published_year", "")),
                "rating": safe_val(row.get("average_rating", "")),
                "similarity": f"{(similarity * 100):.2f}%"
            })
            
        return {
            "matched_title": matched_title,
            "matched_score": f"{float(score):.2f}%",
            "matched_index": int(row_index),
            "recommendations": recommendations
        }
    else: 
        # Encode the query into a vector
        query_vector = model.encode([user_input])[0]
        
        # Normalize
        target_vector = normalize(query_vector.reshape(1, -1))[0]
        
        # Find top similar books
        top_similar = get_top_n_similar(combined_vectors, query_vector, top_n=10)
        
        # Build JSON payload
        recommendations = []
        for i, similarity in top_similar:
            row = df.iloc[i]

            # Fix subtitle handling
            subtitle = row["subtitle"]
            if isinstance(subtitle, float) or pd.isna(subtitle):
                subtitle = ""
            else:
                subtitle = subtitle.title()

            title = row["title"].title()
            full_title = f"{title}: {subtitle}".strip(": ")

            recommendations.append({
                "title": full_title,
                "author": ", ".join([author.title() for author in eval(row["authors"])]),
                "description": safe_val(row.get("description", "")),
                "thumbnail": safe_val(row.get("thumbnail", "")),
                "year": safe_val(row.get("published_year", "")),
                "rating": safe_val(row.get("average_rating", "")),
                "similarity": f"{(similarity * 100):.2f}%"
        })
        
        return {
            "query": user_input,
            "recommendations": recommendations
        }
        
def safe_val(val):
    #Converts NaN/None to empty string; ensures all JSON values are valid.

    if pd.isna(val) or val is None:
        return ""
    return str(val)