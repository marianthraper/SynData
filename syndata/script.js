// Main Chatbot logic (index.html)
async function sendMessage() {
    let userInput = document.getElementById("user-input").value.trim();
    if (!userInput) return;

    let chatBox = document.getElementById("chat-box");

    let userMessage = document.createElement("p");
    userMessage.style.color = "blue";
    userMessage.innerText = "You: " + userInput;
    chatBox.appendChild(userMessage);

    let botMessage = document.createElement("p");
    botMessage.style.color = "green";
    botMessage.innerText = "Bot: Thinking...";
    chatBox.appendChild(botMessage);

    try {
        let response = await fetch("http://127.0.0.1:8000/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: userInput })
        });

        let data = await response.json();
        botMessage.innerText = "Bot: " + (data.response || "No response from server.");

        // Check if the bot's response suggests data generation
        if (data.response && data.response.toLowerCase().includes("generate data")) {
            if (data.response.toLowerCase().includes("file")) {
                window.location.href = "file-upload.html";  // Redirect to file upload page
            } else if (data.response.toLowerCase().includes("columns")) {
                window.location.href = "column-input.html";  // Redirect to column input page
            }
        }

    } catch (error) {
        botMessage.innerText = "Bot: Error generating data.";
        console.error("Error:", error);
    }

    document.getElementById("user-input").value = "";
    chatBox.scrollTop = chatBox.scrollHeight;
}

// File Upload (file-upload.html)
async function uploadFile() {
    let fileInput = document.getElementById("file-input").files[0];
    let numRows = document.getElementById("num-rows").value.trim();

    if (!fileInput || !numRows) {
        alert("Please select a file and enter the number of rows.");
        return;
    }

    let formData = new FormData();
    formData.append("file", fileInput);
    formData.append("num_samples", numRows);

    try {
        let response = await fetch("http://127.0.0.1:8000/generate", {
            method: "POST",
            body: formData
        });

        if (!response.ok) throw new Error(`Server error: ${response.status}`);

        let data = await response.json();
        if (data.file_url) {
            document.getElementById("download-link").innerHTML = 
                `<a href="${data.file_url}" download>Download Synthetic Data</a>`;
        } else {
            document.getElementById("download-link").innerText = "Error generating synthetic data.";
        }

    } catch (error) {
        console.error("Upload Error:", error);
        document.getElementById("download-link").innerText = "Error uploading file.";
    }
}

// Column Input (column-input.html)
async function generateWithoutFile() {
    let columnNames = document.getElementById("column-names").value.trim();
    let numRows = document.getElementById("num-rows-column").value.trim();

    if (!columnNames || !numRows) {
        alert("Please enter column names and the number of rows.");
        return;
    }

    let formData = new FormData();
    formData.append("column_names", columnNames);
    formData.append("num_samples", numRows);

    try {
        let response = await fetch("http://127.0.0.1:8000/generate-with-columns", {
            method: "POST",
            body: formData
        });

        if (!response.ok) throw new Error(`Server error: ${response.status}`);

        let data = await response.json();
        if (data.file_url) {
            document.getElementById("download-link").innerHTML = 
                `<a href="${data.file_url}" download>Download Synthetic Data</a>`;
        } else {
            document.getElementById("download-link").innerText = "No relevant dataset found.";
        }

    } catch (error) {
        console.error("RAG Error:", error);
        document.getElementById("download-link").innerText = "Error fetching synthetic data from RAG.";
    }
}
