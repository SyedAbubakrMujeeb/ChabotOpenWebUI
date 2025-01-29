from flask import Flask, request, jsonify, render_template
import os
import fitz  # PyMuPDF
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Initialize Flask app
app = Flask(__name__)

# Define folders
UPLOAD_FOLDER = "uploads"
INDEX_FOLDER = "faiss_index"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set HuggingFace API token (use environment variable for security)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YOUR_HF_API_TOKEN"

# Initialize global variables
faiss_index = None
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def process_pdf_files():
    """Processes all PDFs in UPLOAD_FOLDER and creates a FAISS index."""
    global faiss_index
    
    pdf_texts = []
    files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(".pdf")]

    if not files:
        print("No PDFs found in the upload folder.")
        return False

    # Extract text from all PDFs
    for filename in files:
        pdf_path = os.path.join(UPLOAD_FOLDER, filename)
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text() for page in doc])
        if text:
            pdf_texts.append(text)

    if not pdf_texts:
        print("No text extracted from PDFs.")
        return False

    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_text("".join(pdf_texts))

    if not text_chunks:
        print("Failed to split text into chunks.")
        return False

    # Create FAISS index and save it
    faiss_index = FAISS.from_texts(text_chunks, embedding)
    faiss_index.save_local(INDEX_FOLDER)
    print("FAISS index successfully created and saved.")
    return True

@app.route("/")
def index():
    """Render the main HTML page."""
    return render_template("Chatbot.html")

@app.route("/process", methods=["POST"])
def process_pdfs():
    """Processes PDFs automatically from UPLOAD_FOLDER."""
    success = process_pdf_files()
    if success:
        return jsonify({"message": "PDFs processed and FAISS index created."}), 200
    else:
        return jsonify({"error": "No valid PDFs found or text extraction failed."}), 400

@app.route("/query", methods=["POST"])
def query_index():
    """Handles user queries against the FAISS index."""
    global faiss_index

    # Load FAISS index if not already loaded
    if faiss_index is None:
        try:
            faiss_index = FAISS.load_local(INDEX_FOLDER, embedding)
        except:
            return jsonify({"error": "Index not found. Process PDFs first."}), 400

    user_query = request.json.get("query")
    if not user_query:
        return jsonify({"error": "No query provided."}), 400

    # Perform similarity search
    results = faiss_index.similarity_search(user_query, k=2)
    return jsonify({"results": [result.page_content for result in results]}), 200

if __name__ == "__main__":
    # Automatically process PDFs at startup
    print("Processing PDFs at startup...")
    process_pdf_files()
    
    app.run(debug=True)
