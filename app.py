import os
from flask import Flask, request, render_template
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import speech_recognition as sr
# from transformers import CLIPProcessor, CLIPModel
import chromadb
from PIL import Image
from openai import OpenAI
from langchain_core.documents import Document
from moviepy import VideoFileClip, AudioFileClip
import pytesseract

app = Flask(__name__)

# Initialize ChromaDB client
def initialize_chromadb(chromadb_path="./chromadb_storage"):
    if not os.path.exists(chromadb_path):
        os.makedirs(chromadb_path)
    settings = chromadb.config.Settings(persist_directory=chromadb_path)
    return chromadb.Client(settings)

# Check if a collection exists
def get_existing_collection(client, collection_name):
    collections = client.list_collections()
    for collection in collections:
        if collection.name == collection_name:
            return client.get_collection(collection_name)
    return None

# Transcribe MP3 audio
def transcribe_audio(mp3_path):
    try:
        if not os.path.exists(mp3_path):
            raise FileNotFoundError(f"File not found: {mp3_path}")

        wav_path = "temp_audio.wav"
        audio_clip = AudioFileClip(mp3_path)
        audio_clip.write_audiofile(wav_path, codec="pcm_s16le")
        audio_clip.close()

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data)
        os.remove(wav_path)
        return text

    except Exception as e:
        return f"Error transcribing audio: {e}"

# Transcribe video by extracting audio
def transcribe_video(video_path):
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"File not found: {video_path}")

        audio_path = "temp_video_audio.wav"
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, codec="pcm_s16le")
        text = transcribe_audio(audio_path)
        os.remove(audio_path)
        return text

    except Exception as e:
        return f"Error transcribing video: {e}"

# Process PDF files
def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(docs)

def extract_text_from_image(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    
    except Exception as e:
        return f"Error extracting text from image: {e}"

# Create or load ChromaDB collection
def load_or_create_chromadb_collection(client, collection_name="amit_dk", folder_path="./datasets"):
    existing_collection = get_existing_collection(client, collection_name)
    if existing_collection:
        print("ChromaDB collection found. Using existing embeddings.")
        return existing_collection

    print("Creating new ChromaDB collection.")
    documents = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # Process files based on extension
        if file_name.endswith(".pdf"):
            documents.extend(process_pdf(file_path))
        elif file_name.endswith(".mp3"):
            audio_text = transcribe_audio(file_path)
            documents.append(Document(page_content=audio_text, metadata={"source": "audio"}))
        elif file_name.endswith((".mp4", ".avi", ".mov")):
            video_text = transcribe_video(file_path)
            documents.append(Document(page_content=video_text, metadata={"source": "video"}))

        elif file_name.endswith((".jpg", ".jpeg", ".png")):
            image_description = extract_text_from_image(file_path)
            documents.append(Document(page_content=image_description, metadata={"source": "image"}))

    # Add documents to the collection
    content = [doc.page_content for doc in documents]
    metadata = [doc.metadata for doc in documents]
    ids = [str(i) for i in range(len(documents))]
    collection = client.get_or_create_collection(name=collection_name)
    collection.add(documents=content, metadatas=metadata, ids=ids)

    return collection

# Retrieve chunks from ChromaDB
def retrieve_useful_chunk(query, collection):
    results = collection.query(query_texts=[query], n_results=3)
    if not results["documents"]:
        return None
    return " ".join(doc for doc in results["documents"][0])

# Generate answers using LLM
def generate_questions_answers(question, context):
    if context is None:
        return "No relevant information found in the context. Please refine your query."

    llm_client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )

    prompt = (
            f"Context: {context}\n\n"
            f"Question: {question}\n"
            "Your task is to answer the question based strictly on the provided context. If the context does not contain sufficient "
            "or relevant information to answer the question, respond with: 'The provided context does not address this question. "
            "Please refine your query or provide more relevant details.'\n"
            "Answer:"
            )

    response = llm_client.chat.completions.create(
        model="llama3.2:1b",
        messages=[{"role": "system", "content": "You are a helpful assistant. if context does not match then return ask question related to context"},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Flask app routes
initialized = False  # Flag to track initialization

@app.before_request
def initialize_once():
    global collection, initialized
    if not initialized:
        print("Initializing ChromaDB collection...")
        collection = load_or_create_chromadb_collection(chroma_client)
        initialized = True

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    if request.method == "POST":
        question = request.form.get("question", "")
        if question:
            context = retrieve_useful_chunk(question, collection)
            answer = generate_questions_answers(question, context)


    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    chroma_client = initialize_chromadb()
    app.run(debug=True)