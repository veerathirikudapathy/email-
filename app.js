const messageInput = document.getElementById("email-message");
const charCount = document.getElementById("char-count");
const button = document.getElementById("check-button");
const errorBox = document.getElementById("error-message");
const resultBox = document.getElementById("result");

messageInput.addEventListener("input", () => {
  charCount.textContent = `${messageInput.value.length.toLocaleString()} / 10,000`;
});

button.addEventListener("click", async () => {
  const message = messageInput.value.trim();
  errorBox.hidden = true;
  resultBox.hidden = true;

  if (!message) {
    showError("Please paste an email message before analyzing.");
    messageInput.focus();
    return;
  }

  button.disabled = true;
  button.querySelector(".button-label").textContent = "Analyzing…";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({message})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not analyze this message.");

    const isSpam = data.label === "spam";
    document.getElementById("result-title").textContent = data.display_label;
    document.getElementById("result-description").textContent = isSpam
      ? "This message contains patterns commonly associated with spam. Be cautious with links, attachments, and requests for personal information."
      : "This message looks more like a normal email based on the patterns learned by the model.";
    const badge = document.getElementById("result-badge");
    badge.textContent = isSpam ? "!" : "✓";
    badge.className = `result-badge ${isSpam ? "spam" : "ham"}`;
    document.getElementById("confidence-value").textContent = `${data.confidence.toFixed(2)}%`;
    document.getElementById("confidence-meter").style.width = `${data.confidence}%`;
    document.getElementById("spam-score").textContent = `${(data.probabilities.spam ?? 0).toFixed(2)}%`;
    document.getElementById("ham-score").textContent = `${(data.probabilities.ham ?? 0).toFixed(2)}%`;
    resultBox.hidden = false;
    resultBox.scrollIntoView({behavior: "smooth", block: "nearest"});
  } catch (error) {
    showError(error.message || "Something went wrong. Please try again.");
  } finally {
    button.disabled = false;
    button.querySelector(".button-label").textContent = "Analyze message";
  }
});

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}
