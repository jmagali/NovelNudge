function sendSearch() {
    const query = document.getElementById("searchBox").value;

    const formData = new FormData();
    formData.append("query", query);

    fetch("/search", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("bestMatch").innerHTML = `
            <h3>Best Match: ${data.best_match}</h3>
            <p>Similarity Score: ${data.best_score ?? "N/A"}</p>
        `;

        const recDiv = document.getElementById("recommendations");
        recDiv.innerHTML = ""; // reset

        data.recommendations.forEach(rec => {
            recDiv.innerHTML += `
                <div class="book-card">
                    <img src="${rec.thumbnail}" alt="">
                    <h3>${rec.title}</h3>
                    <p><strong>Author:</strong> ${rec.author}</p>
                    <p><strong>Year:</strong> ${rec.year}</p>
                    <p>${rec.description}</p>
                    <p><strong>Similarity:</strong> ${rec.similarity.toFixed(3)}</p>
                </div>
            `;
        });
    });
}
