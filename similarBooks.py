# ================================
# Full Working Script: Text Similarity
# ================================

# Step 0: Install dependencies if you haven't already:
# pip install pandas sentence-transformers scikit-learn

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------------------------
# Step 1: Load your CSV
# ---------------------------
print("hello")
csv_file = "unprocessedData.csv"    # <-- change this to your CSV path
print("hello")
text_column = "description"  # <-- change this to the column with text

print("Loading CSV...")
df = pd.read_csv(csv_file)
texts = df[text_column].astype(str).tolist()
print(f"Loaded {len(texts)} rows from column '{text_column}'")

# ---------------------------
# Step 2: Vectorize text
# ---------------------------
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding text into vectors...")
vectors = model.encode(texts, batch_size=64, show_progress_bar=True)
print("Text encoding complete!")

# ---------------------------
# Step 3: Similarity function
# ---------------------------
def get_top_n_similar(vector_list, target_vector, top_n=10):
    """
    Given a target vector, return top N most similar row indices and similarity scores.
    """
    scores = cosine_similarity([target_vector], vector_list)[0]
    top_indices = np.argsort(scores)[-top_n:][::-1]  # highest first
    return [(i, scores[i]) for i in top_indices]

# ---------------------------
# Step 4: Choose target
# ---------------------------
print("\nDo you want to pick a row from CSV or type a query?")
print("1 = Pick row by index (0-based)")
print("2 = Type a custom sentence")
choice = input("Enter 1 or 2: ")

if choice.strip() == "1":
    row_index = int(input(f"Enter row index (0 to {len(texts)-1}): "))
    target_vector = vectors[row_index]
    print(f"\nUsing row {row_index} as target: {df[text_column][row_index]}")
else:
    query = input("Enter your sentence/query: ")
    target_vector = model.encode([query])[0]
    print(f"\nUsing your query as target: {query}")

# ---------------------------
# Step 5: Get top 10 similar rows
# ---------------------------
top_similar = get_top_n_similar(vectors, target_vector, top_n=10)

print("\nTop 10 similar rows:")
for i, score in top_similar:
    print(f"Row {i} (Similarity: {score:.4f}): {df[text_column][i]}")
