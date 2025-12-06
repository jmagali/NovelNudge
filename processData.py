import pandas as pd

# Get genres file
genres_df = pd.read_csv("./data/genre_list.csv", header=None, names=["genre"])
genres = genres_df["genre"].tolist()
genres = [genre.lower() for genre in genres]

# Fetch the unprocessed csv file
df = pd.read_csv('./data/unprocessedData.csv')

# Fill null data with empty string
df = df.fillna("")

# Remove thumbnail
df = df.drop(['thumbnail'], axis=1)

# Clean data based on ISBN13/10 
# Ensure proper data type; Remove duplicate ISBNs
df["isbn13"] = df["isbn13"].astype(str).str.replace("-", "", regex=False)
df["isbn10"] = df["isbn10"].astype(str).str.replace("-", "", regex=False)
df = df.drop_duplicates(subset=["isbn13", "isbn10"])

def remove_punctuation(text):
    if not isinstance(text, str):
        return ""
    for char in ['.', ',', ';', ':', '/', '\\', '(', ')', '[', ']', '{', '}', '"', "'"]:
        text = text.replace(char, " ")
    return " ".join(text.split()).lower()

def extract_genres(category):
    if not isinstance(category, str) or not category.strip():
        return []
    
    cell = remove_punctuation(category)
    found = []

    # Compares genre to list of valid genres
    for genre in genres:
        if genre in cell and genre not in found:
            found.append(genre)

    return found

# Processes genre
df["genre_list"] = df["categories"].apply(extract_genres)
df = df[df["genre_list"].map(len) > 0]

# Process Authors
def clean_authors(authors):
    if not isinstance(authors, str):
        return []
    
    # Split by semicolon
    author_list = [a.strip().lower() for a in authors.split(";") if a.strip()]
    
    return author_list

df["authors"] = df["authors"].apply(clean_authors)

# Convert titles and subtitles to lowercase
df["title"] = df["title"].apply(lambda title: title.lower())
df["subtitle"] = df["subtitle"].apply(lambda subtitle: subtitle.lower())

df.to_csv('./data/processed_data.csv', index=False)
