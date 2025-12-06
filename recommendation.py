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

def recommend_from_title(user_input):
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
        recommendations.append({
            "title": f"{row['title'].title()}: {row['subtitle'].title()}",
            "author": ", ".join([author.title() for author in eval(row["authors"])]),
            "description": row["description"],
            "thumbnail": row["thumbnail"],
            "year": row["published_year"],
            "rating": row["average_rating"],
            "similarity": f"{float(similarity):.2}%"
        })
        
    return {
        "matched_title": matched_title,
        "matched_score": score,
        "matched_index": int(row_index),
        "recommendations": recommendations
    }
    
def recommend_from_query(query):
    # Encode the query into vectors
    query_vectorDes = model.encode([query])[0]
    query_vectorAuthor = np.zeros_like(query_vectorDes)  # no author info
    query_vectorGenre = np.zeros_like(query_vectorDes)   # no genre info
    
    # Combine with same weights as books
    target_vector = (query_vectorDes * desc_weight +
                     query_vectorGenre * genre_weight +
                     query_vectorAuthor * author_weight)
    
    # Normalize
    target_vector = normalize(target_vector.reshape(1, -1))[0]
    
    # Find top similar books
    top_similar = get_top_n_similar(combined_vectors, target_vector, top_n=10)
    
    # Build JSON output
    recommendations = []
    for i, similarity in top_similar:
        row = df.iloc[i]

        recommendations.append({
            "title": f"{row['title'].title()}: {row['subtitle'].title()}",
            "author": ", ".join([author.title() for author in eval(row["authors"])]),
            "description": row["description"],
            "thumbnail": row["thumbnail"],
            "year": row["published_year"],
            "rating": row["average_rating"],
            "similarity": f"{float(similarity):.2}%"
        })
    
    return {
        "query": query,
        "recommendations": recommendations
    }

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
}

# Load vectors if they exist
if all(os.path.exists(f) for f in vector_files.values()):
    print("Loading saved vectors from disk...")
    vectorsDes = np.load(vector_files["description"])
    vectorsAuthor = np.load(vector_files["author"])
    vectorsGenre = np.load(vector_files["genre"])
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

# # Get user interest/book
# print("\nDo you want to pick a row from CSV or type a query?")
# print("1 = Type a book title")
# print("2 = Type a custom sentence")
# choice = input("Enter 1 or 2: ")

# if choice.strip() == "1":
#     user_input = input("Enter book name: ").strip().lower()
    
#     # Use RapidFuzz for title matching
#     result = process.extractOne(user_input, titleTxt, scorer=fuzz.token_sort_ratio, score_cutoff=60)
    
#     if result is None:
#         print("Could not find a close match.")
#         sys.exit()
    
#     matched_title, score, row_index = result
#     print(f"Closest match: {titleTxt[row_index]} (Score: {score:.2f}%)")
#     target_vector = combined_vectors[row_index]
# else:
#     query = input("Enter your sentence/query: ")
    
#     # Encode all three features for the query
#     query_vectorDes = model.encode([query])[0]
#     query_vectorAuthor = np.zeros_like(query_vectorDes)  # no author info for query
#     query_vectorGenre = np.zeros_like(query_vectorDes)   # no genre info for query
    
#     # Combine with same weights
#     target_vector = (query_vectorDes * desc_weight +
#                      query_vectorGenre * genre_weight +
#                      query_vectorAuthor * author_weight)
    
#     # Normalize
#     target_vector = normalize(target_vector.reshape(1, -1))[0]
#     print(f"\nUsing your query as target: {query}")

# # Find similar books
# top_similar = get_top_n_similar(combined_vectors, target_vector, top_n=10,
#                                 exclude_index=row_index if choice=="1" else None)

# # display similar books
# print("\nTop 10 similar rows:")
# for i, score in top_similar:
#     formatted_title = df['title'][i].title()
#     print(f"Title: {formatted_title} (Similarity: {score:.2f}%)")
    

