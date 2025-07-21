# Install required libraries (first time only)
# pip install pandas sentence-transformers faiss-cpu chainlit google-generativeai
import os
import csv
import uuid
import pandas as pd
import chainlit as cl
import faiss
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from datetime import datetime

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the Gemini API key securely


###
#
# ---------------------------
# 1. Configure Gemini API
# ---------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # <-- Replace this with your real Gemini API key
#genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# 🔍 List available models and what they support
#for m in genai.list_models():
#    print(m.name, "  supports:", m.supported_generation_methods)

# ---------------------------
# 2. Load CSV dataset
# ---------------------------
data = pd.read_csv('/Users/lakshitgupta/Library/CloudStorage/OneDrive-SeattleUniversity/Quater5/Capstone/Notebooks/costco_insta_socials_Threads.csv')

# Combine review text and metadata
texts = (
    data['product_name'].fillna('') + ' ' +
    data['main_category'].fillna('') + ' ' +
    data['subcategory'].fillna('') + ' ' +
    data['comments'].fillna('') + ' ' +
    data['comment_year'].astype(str).fillna('') + ' ' +
    data['comment_month'].astype(str).fillna('') + ' ' +
    data['source'].astype(str).fillna('')
).fillna('')

sentiments = data['sentiment'].fillna('')
categories = (data['main_category'].fillna('') + ' ' + data['subcategory'].fillna('')).fillna('')

# Embedding models
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Generating text and category embeddings...")
text_embeddings = embedder.encode(texts.tolist())
category_embeddings = embedder.encode(categories.tolist())
print("Embeddings generated.")

# Build FAISS indexes
text_index = faiss.IndexFlatL2(text_embeddings.shape[1])
text_index.add(np.array(text_embeddings))

category_index = faiss.IndexFlatL2(category_embeddings.shape[1])
category_index.add(np.array(category_embeddings))

# ---------------------------
# 4. Ask Gemini API
# ---------------------------


def ask_gemini(history, context, question):
    prompt = f"""You are a helpful assistant analyzing Costco product reviews and their sentiment labels (Positive, Negative, Neutral) from various social media platforms.

Your task is to provide concise, data-backed insights. Include:
- The number of reviews mentioned.
- A count of each sentiment type (Positive / Negative / Neutral).
- Any clear trends, recurring complaints, or praises.
- Use numbers or percentages where possible.
- I dont need the review, just the analysis

{history}
Reviews:
{context}

Question:
{question}

Answer:"""
    model = genai.GenerativeModel('models/gemini-1.5-pro')  
    response = model.generate_content(prompt)
    print("\nGemini Response:\n", response.text)
    return response.text

# ---------------------------
# 5. Log interactions to CSV
# ---------------------------
#def log_interaction(question, answer, feedback=None, session_id=None):
#    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#    with open("interaction_log.csv", "a", newline="") as file:
#        writer = csv.writer(file)
#        writer.writerow([timestamp, session_id, question, answer, feedback])


def log_interaction(question, answer, feedback=None, session_id=None):
    import os

    log_file = "interaction_log.csv"
    file_exists = os.path.isfile(log_file)

    with open(log_file, "a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "session_id", "question", "answer", "feedback"])  # Add header once
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            session_id or "unknown",
            question or "",
            answer or "",
            feedback or ""
        ])

def extract_category_or_subcategory(user_input):
    user_input_lower = user_input.lower()
    found_main = None
    found_sub = None

    for cat in main_categories:
        if cat in user_input_lower:
            found_main = cat
            break

    for sub in subcategories:
        if sub in user_input_lower:
            found_sub = sub
            break

    return found_main, found_sub

# ---------------------------
# 6. Main chatbot handler
# ---------------------------
@cl.on_message
async def main(message: cl.Message):
    try:
        user_input = message.content

        # Embed the query and search category embeddings
        query_embedding = embedder.encode([user_input])
        distances, indices = category_index.search(np.array(query_embedding), k=30)

        matched_texts = []
        for idx in indices[0]:
            review_text = texts.iloc[idx]
            sentiment_label = sentiments.iloc[idx]
            matched_texts.append(f"Review: {review_text}\nSentiment: {sentiment_label}")

        if not matched_texts:
            await cl.Message(content="❌ No relevant reviews found. Try asking in a different way.").send()
            return

        context = "\n\n".join(matched_texts)
        if len(context) > 1700:
            context = context[:1700]

        history = cl.user_session.get("history", "")
        answer = ask_gemini(history, context, user_input)

        cl.user_session.set("history", history + f"\nUser: {user_input}\nAssistant: {answer}\n")
        cl.user_session.set("last_q", user_input)
        cl.user_session.set("last_a", answer)

        await cl.Message(
            content=answer,
            author="Gemini Bot",
            actions=[
                cl.Action(name="feedback", label="👍", value="thumbs_up"),
                cl.Action(name="feedback", label="👎", value="thumbs_down"),
                cl.Action(name="reset_session", label="🔁", value="reset")
            ]
        ).send()

    except Exception as e:
        import traceback
        traceback.print_exc()
        await cl.Message(content=f"❌ Error: {str(e)}").send()

# ---------------------------
# 7. Handle thumbs feedback (Chainlit 2.x compatible)
# ---------------------------

# Feedback Handler
@cl.action_callback("feedback")
async def handle_feedback(action: cl.Action):
    q = cl.user_session.get("last_q")
    a = cl.user_session.get("last_a")
    session_id = cl.user_session.get("session_id", "anonymous")
    log_interaction(q, a, feedback=action.value, session_id=session_id)
    await cl.Message(content=f"✅ Feedback received: {action.value}").send()

# Reset Chat
@cl.action_callback("reset_session")
async def handle_reset(action: cl.Action):
    cl.user_session.set("history", "")
    cl.user_session.set("last_q", "")
    cl.user_session.set("last_a", "")
    await cl.Message(content="🔄 Chat reset. You can now ask a new question.").send()

# Session Start
@cl.on_chat_start
async def start():
    session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", session_id)

    await cl.Message(
        content="**Welcome to the Costco Review Assistant!!**\nAsk me anything about Costco products, customer sentiments, or trends",
        author="Team MSDS 25.4"
    ).send()
