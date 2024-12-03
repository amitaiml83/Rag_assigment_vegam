import os
from flask import Flask, request, render_template
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import speech_recognition as sr
from transformers import CLIPProcessor, CLIPModel
import chromadb
from PIL import Image
from openai import OpenAI
from langchain_core.documents import Document
from moviepy import VideoFileClip, AudioFileClip

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

def transcribe_audio(mp3_path):
    """Transcribe MP3 audio file using moviepy and speech recognition."""
    try:
        if not os.path.exists(mp3_path):
            raise FileNotFoundError(f"File not found: {mp3_path}")

        print(f"Converting MP3 to WAV... {mp3_path}")
        wav_path = "temp_audio.wav"

        audio_clip = AudioFileClip(mp3_path)
        audio_clip.write_audiofile(wav_path, codec='pcm_s16le')
        audio_clip.close()

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            print("Loading audio for transcription...")
            audio_data = recognizer.record(source)

        print("Audio successfully loaded, starting transcription...")
        text = recognizer.recognize_google(audio_data)
        os.remove(wav_path)
        return text

    except FileNotFoundError as e:
        print(f"File error: {e}")
        return str(e)

    except sr.RequestError as e:
        print(f"Speech recognition API error: {e}")
        return f"Speech recognition API error: {e}"

    except sr.UnknownValueError:
        print("Speech recognition could not understand the audio.")
        return "Could not transcribe: Unclear audio."

    except Exception as e:
        print(f"Unexpected error: {e}")
        return f"Error processing audio: {e}"

def transcribe_video(video_path):
    """Extract audio from a video file and transcribe it."""
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"File not found: {video_path}")

        print(f"Extracting audio from video... {video_path}")
        audio_path = "temp_video_audio.wav"
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, codec="pcm_s16le")
        transcription = transcribe_audio(audio_path)
        os.remove(audio_path)
        return transcription

    except Exception as e:
        print(f"Error processing video: {e}")
        return f"Error processing video: {e}"

def process_pdf(pdf_path):
    """Load and process PDF files."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
    return split_docs

def extract_description_from_image(image_path):
    """Extract description from an image using CLIP."""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt", text=["a photo", "an image"])
    outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    description_index = logits_per_image.argmax().item()
    description = ["a photo", "an image"][description_index]
    return description

def load_or_create_chromadb_collection(client, collection_name="amit_dk", folder_path="./files"):
    """Process all files in a specified folder and create a ChromaDB collection."""
    existing_collection = get_existing_collection(client, collection_name)
    if existing_collection:
        print("ChromaDB collection found. Using existing embeddings.")
        return existing_collection

    print("ChromaDB collection not found. Creating embeddings.")
    documents = []

    # Process all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # Check for PDF files
        if file_name.endswith(".pdf"):
            print(f"Processing PDF: {file_name}")
            pdf_docs = process_pdf(file_path)
            documents.extend(pdf_docs)

        # Check for audio (MP3) files
        elif file_name.endswith(".mp3"):
            print(f"Processing MP3: {file_name}")
            audio_text = transcribe_audio(file_path)
            documents.append(Document(page_content=audio_text, metadata={"source": "audio"}))

        # Check for video files
        elif file_name.endswith((".mp4", ".avi", ".mov")):
            print(f"Processing Video: {file_name}")
            video_text = transcribe_video(file_path)
            documents.append(Document(page_content=video_text, metadata={"source": "video"}))

        # Check for image files (optional)
        elif file_name.endswith((".jpg", ".jpeg", ".png")):
            print(f"Processing Image: {file_name}")
            image_description = extract_description_from_image(file_path)
            documents.append(Document(page_content=image_description, metadata={"source": "image"}))

    # Add documents to ChromaDB collection
    docx_dict = {
        "content": [str(doc.page_content) for doc in documents],
        "metadata": [doc.metadata for doc in documents],
        "idx": [str(i) for i in range(len(documents))]
    }

    collection = client.get_or_create_collection(name=collection_name)
    collection.add(
        documents=docx_dict["content"],
        metadatas=docx_dict["metadata"],
        ids=docx_dict["idx"]
    )
    return collection

def retrieve_useful_chunk(query, collection):
    """Retrieve useful chunk from ChromaDB based on the query."""
    results = collection.query(query_texts=[query], n_results=5)
    if not results["documents"]:
        return None
    return " ".join(results["documents"][0])

def generate_questions_answers(question, context):
    """Generate answers based on context."""
    if context is None:
        return "Please ask a question related to the provided files. I couldn't find relevant information."

    else:
        llm_client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )

        prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"

        response = llm_client.chat.completions.create(
            model="llama3.2:1b",
            messages=[{"role": "system", "content": "You are a helpful assistant. Answer based on the context."},
                      {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
folder_path = "./datasets" 
chroma_client = initialize_chromadb("./chromadb_storage")
collection = load_or_create_chromadb_collection(
    chroma_client, collection_name="amit_dk", folder_path=folder_path
    )

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    if request.method == "POST":
        question = request.form.get("question", "")
        if question:
            # Load or create ChromaDB collection from the folder containing all files # Specify the folder containing PDFs, MP3s, and videos

            context = retrieve_useful_chunk(question, collection)
            answer = generate_questions_answers(question, context)

    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
