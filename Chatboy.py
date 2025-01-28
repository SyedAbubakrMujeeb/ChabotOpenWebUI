from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Add CORS support
import os
import fitz  # PyMuPDF for PDF text extraction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ollama import chat, ChatResponse

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend interaction

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

pdf_texts = {}

# Load PDF data
def load_pdf_data():
    global pdf_texts
    for filename in os.listdir(UPLOAD_FOLDER):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(UPLOAD_FOLDER, filename)
            pdf_texts[filename] = extract_text_from_pdf(pdf_path)

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text += f"Page {page.number + 1}:\n" + page.get_text() + "\n"
    return text

# Check if query is relevant to the context
def is_query_in_context(query, context):
    query_keywords = set(query.lower().split())
    context_keywords = set(context.lower().split())
    return bool(query_keywords & context_keywords)

# Check if answer is based on the context
def is_answer_in_context(answer, context):
    answer_keywords = set(answer.lower().split())
    context_keywords = set(context.lower().split())
    return bool(answer_keywords & context_keywords)

# Query chatbot
def query_chatbot(user_query, model_name="llama3.2"):
    try:
        system_messages = [
            {
                "role": "system",
                "content": f"PDF Name: {filename}\n\nText:\n{pdf_text}",
            }
            for filename, pdf_text in pdf_texts.items()
        ]
        messages = system_messages + [{"role": "user", "content": user_query}]
        response: ChatResponse = chat(model=model_name, messages=messages)
        return response.message.content
    except Exception as e:
        raise RuntimeError(f"Error querying chatbot: {str(e)}")

@app.route("/")
def index():
    return render_template("Chatbot.html")

@app.route("/query", methods=["POST"])
def query_chatbot_route():
    try:
        # Get the user query
        user_query = request.json.get("query")
        if not user_query:
            return jsonify({"error": "No query provided"}), 400

        if not pdf_texts:
            return jsonify({"error": "No PDF data available to answer your query."}), 400

        # Combine and split PDF content into meaningful chunks
        context = "\n\n".join(
            [f"PDF Name: {filename}\n\nText:\n{pdf_text}" for filename, pdf_text in pdf_texts.items()]
        )
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, separators=["\n\n", ".", " "])
        context_chunks = text_splitter.split_text(context)

        # Identify relevant chunks
        relevant_chunks = [chunk for chunk in context_chunks if is_query_in_context(user_query, chunk)]
        context_to_use = "\n\n".join(relevant_chunks) if relevant_chunks else context

        # System message
        system_messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict assistant. Only answer questions using the following PDF data. "
                    "If the answer cannot be found in the PDF, respond only with: "
                    "'Sorry, I cannot help you with that.'\n\n"
                    f"{context_to_use}"
                )
            }
        ]
        messages = system_messages + [{"role": "user", "content": user_query}]

        # Query the chatbot
        response: ChatResponse = chat(model="llama3.2", messages=messages)
        bot_response = response.message.content.strip()

        # Validate response relevance
        if not is_answer_in_context(bot_response, context_to_use):
            bot_response = "Sorry, I cannot help you with that."

        return jsonify({"response": bot_response})

    except Exception as e:
        return jsonify({"error": f"Error querying chatbot: {str(e)}"}), 500

if __name__ == "__main__":
    load_pdf_data()  # Load PDF data before starting the app
    app.run(debug=True)
