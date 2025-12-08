# NovelNudge

<i>Submission for <a href="https://hacktheridge.ca" target="_blank" rel="noopener noreferrer">Hack The Ridge 2025</a>.</i>

A smart book recommendation engine powered by vector embeddings.

Discover books tailored to your interests!
<b>NovelNudge</b> uses <b>SentenceTransformers</b> to embed book titles, descriptions, authors, and genres into vectors. 
It then finds the most similar books using cosine similarity and fuzzy title matching, delivering highly relevant recommendations in real-time.

## Features
<ul>
  <li>Search by book title or description</li>
  <li>Combines multiple factors (title, description, author, genre) into a weighted vector for better recommendations</li>
  <li>RapidFuzz integration for fuzzy matching of book titles</li>
  <li>Responsive web interface with live search results and thumbnails</li>
  <li>Easily extendable dataset and vector embeddings</li>
</ul>

## Installation
### Requirements
- **Python 3.10 — 3.12** recommended
- **pip** installed

### 1.  Clone the repo
   ```sh
   git clone https://github.com/jmagali/NovelNudge.git
   ```
### 2.  In the terminal, navigate to the directory where the repository was cloned, e.g.,
   ```sh
   C:\Users\User\NovelNudge
   ```
### 3.  Install the required Python libraries
   ```sh
   pip install -r requirements.txt # This installs the required libraries
   ```
### 4.  Run the Flask application
   ```sh
   python app.py
   ```
### 5.  Open in browser
   ```sh
   Navigate to http://127.0.0.1:5000/ to use NovelNudge.
   ```

## How It Works
- <b>Vectorization:</b> Titles, descriptions, authors, and genres are encoded into vectors using SentenceTransformer.
- <b>Combining Vectors:</b> Each book's vectors are weighted and summed to form a combined vector.
- <b>Search & Matching:</b>
  - <b>Title mode:</b> RapidFuzz fuzzy matching selects a close match first.
  - <b>Description mode:</b> Input text is embedded and compared with all combined vectors.
- <b>Recommendation:</b> Top-N similar books are returned based on cosine similarity.
- <b>Frontend Display:</b> Results are displayed with thumbnails, authors, ratings, and similarity scores.

## Director Structure
```bash
.
├── data/
│   ├── vectors/
│   │   ├── combined_vectors.npy
│   │   ├── vectorsAuthor.npy
│   │   ├── vectorsDes.npy
│   │   ├── vectorsGenre.npy
│   │   └── vectorsTitle.npy
│   ├── genre_list.csv
│   ├── processed_data.csv
│   └── unprocessed_data.csv
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── app.js
│   └── img/
│       ├── book-education-learn-svgrepo-com.svg
│       └── github-mark.svg
├── templates/
│   └── index.html
├── .gitignore
├── .gitattributes
├── app.py
├── process_data.py
├── recommendations.py
└── requirements.txt
└── README.md
```
