function sendSearch() {
  const query = document.getElementById("searchBox").value;
  const modeBtn = document.getElementById("mode");
  const bestMatch = document.getElementById("bestMatch");
  const recTitle = document.getElementById("recommendations_title");
  const recContainer = document.getElementById("recommendations");
  const hiddenItems = [bestMatch, recTitle, recContainer];
  let mode;

  if (modeBtn.classList.contains("title")) {
    mode = 1;
  } else {
    mode = 2;
  }

  // Send data to backend
  const formData = new FormData();
  formData.append("query", query);
  formData.append("mode", mode);

  fetch("/search", {
    method: "POST",
    body: formData,
  })
    .then((res) => res.json())
    .then((data) => {
      // TITLE SEARCH RESPONSE
      const bestTitle = data.matched_title || data.query || "No match";
      const bestScore = data.matched_score || "N/A";

      document.getElementById("bestMatch").innerHTML = `
            <h3>Best Match: ${bestTitle}</h3>
            <p>Similarity Score: ${bestScore}</p>
        `;

      const recDiv = document.getElementById("recommendations");
      recDiv.innerHTML = "";

      data.recommendations.forEach((rec) => {
        // Create card for each book
        const bookCard = document.createElement("div");
        bookCard.classList.add("book-card");
        recDiv.appendChild(bookCard);

        // Create book thumbnail
        const thumbnail = document.createElement("img");
        thumbnail.src = `${rec.thumbnail}`;
        thumbnail.alt = "";
        bookCard.appendChild(thumbnail);

        // Append book data
        const title = document.createElement("h3");
        title.textContent = `${rec.title}`;
        bookCard.appendChild(title);

        bookCard.appendChild(labeledText("Author", rec.author));
        bookCard.appendChild(labeledText("Synopsis", rec.description));
        bookCard.appendChild(labeledText("Year", rec.year));
        bookCard.appendChild(labeledText("Rating", `${rec.rating}/5.0`));
        bookCard.appendChild(labeledText("Similarity", rec.similarity));
      });
    })
    .catch((err) => console.error("Fetch error:", err));

  // Make items visible
  hiddenItems.forEach((element) => {
    element.classList.remove("hidden");
  });
}

function labeledText(label, value) {
  const p = document.createElement("p");
  const strong = document.createElement("strong");

  strong.textContent = label + ": ";
  p.appendChild(strong);
  p.appendChild(document.createTextNode(value));

  return p;
}

function changeMode() {
  const modeBtn = document.getElementById("mode");
  const searchBox = document.getElementById("searchBox");

  if (modeBtn.classList.contains("title")) {
    modeBtn.classList.remove("title");
    modeBtn.classList.add("query");
    modeBtn.textContent = "Book Description";
    searchBox.placeholder = "Enter a book description...";
  } else {
    modeBtn.classList.remove("query");
    modeBtn.classList.add("title");
    modeBtn.textContent = "Book Title";
    searchBox.placeholder = "Enter a book Search for a book...";
  }
}
