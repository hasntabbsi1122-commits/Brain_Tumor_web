import { auth, db } from "./firebase-config.js";
import { 
  collection, addDoc, serverTimestamp 
} from "https://www.gstatic.com/firebasejs/11.0.1/firebase-firestore.js";

const chatBox = document.getElementById("chatBox");
const userInput = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");
const clearBtn = document.getElementById("clearBtn");

const GEMINI_API_KEY = "AIzaSyBuJPQVBp99-KSY36cllWQLarUlhYGZiCY";

// ✅ FIX 1: Use gemini-2.0-flash (1.5-flash is outdated and often quota-blocked)
const API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=" + GEMINI_API_KEY;

// ✅ Keep conversation history for multi-turn chat
const conversationHistory = [];

// --- Send Message ---
sendBtn.addEventListener("click", sendMessage);

// ✅ FIX 2: Also allow pressing Enter key to send
userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

async function sendMessage() {
  const question = userInput.value.trim();
  if (!question) return;

  appendMessage("user", question);
  userInput.value = "";

  appendMessage("ai", "Typing...");
  const aiMessage = chatBox.lastElementChild;

  // ✅ FIX 3: Add user message to history BEFORE sending
  conversationHistory.push({
    role: "user",
    parts: [{ text: question }]
  });

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({

        // ✅ FIX 4: Add system instruction — tells Gemini to be a brain tumor assistant
        systemInstruction: {
          parts: [{
            text: `You are a helpful medical AI assistant specialized in brain tumors. 
                   You help users understand brain tumor types (Glioma, Meningioma, Pituitary), 
                   symptoms, treatments, and MRI analysis results. 
                   Always be clear, empathetic, and recommend consulting a real doctor for diagnosis.`
          }]
        },

        // ✅ FIX 5: Send full conversation history (not just one message)
        // This is what makes Gemini remember previous messages
        contents: conversationHistory,

        // ✅ FIX 6: Add generation config for better responses
        generationConfig: {
          temperature: 0.7,
          maxOutputTokens: 1024,
        }
      })
    });

    // ✅ FIX 7: Check if response is OK before parsing
    if (!response.ok) {
      const errorData = await response.json();
      console.error("Gemini API Error:", errorData);
      aiMessage.textContent = "⚠️ API Error: " + (errorData.error?.message || "Unknown error");
      return;
    }

    const data = await response.json();
    console.log("Gemini response:", data);

    let answer = "Sorry, I couldn't understand that.";

    if (data.candidates && data.candidates.length > 0) {
      // ✅ FIX 8: Also check finishReason — if SAFETY, model was blocked
      const candidate = data.candidates[0];
      if (candidate.finishReason === "SAFETY") {
        answer = "⚠️ Response blocked due to safety filters. Please rephrase your question.";
      } else {
        answer = candidate.content.parts[0].text;
      }
    }

    aiMessage.textContent = answer;

    // ✅ FIX 9: Add AI reply to history so next message has context
    conversationHistory.push({
      role: "model",
      parts: [{ text: answer }]
    });

    saveChatToFirebase(question, answer);

  } catch (err) {
    aiMessage.textContent = "⚠️ Error connecting to AI.";
    console.error(err);
  }
}

// --- Clear Chat ---
clearBtn.addEventListener("click", () => {
  chatBox.innerHTML = `<div class="message ai">👋 Hi! I'm your Brain Tumor AI assistant. Ask me anything about brain tumors, MRI results, or symptoms.</div>`;
  // ✅ Also clear conversation history when chat is cleared
  conversationHistory.length = 0;
});

// --- Append Message ---
function appendMessage(sender, text) {
  const div = document.createElement("div");
  div.classList.add("message", sender);
  div.textContent = text;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}

// --- Save Chat to Firestore ---
async function saveChatToFirebase(question, answer) {
  const user = auth.currentUser;
  if (!user) return;

  try {
    await addDoc(collection(db, "users", user.uid, "chatHistory"), {
      question,
      answer,
      timestamp: serverTimestamp()
    });
  } catch (error) {
    console.error("Error saving chat:", error);
  }
}