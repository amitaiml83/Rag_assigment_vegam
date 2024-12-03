# RAG Assignment Project

This project demonstrates a Retrieval-Augmented Generation (RAG) application that integrates multiple data modalities such as PDFs, audio, video, and images. It uses a combination of natural language processing (NLP), transcription, and computer vision techniques to create a searchable knowledge base using ChromaDB and generates responses based on user queries with the help of the Ollama LLM.

## Features

- Processes and indexes:
  - PDFs (using `langchain_community.document_loaders`)
  - Audio files (MP3)
  - Video files (extracts audio for transcription)
  - Images (descriptions generated using CLIP)
- Stores embeddings in a ChromaDB database.
- Uses Ollama to generate responses based on a user-provided context and question.
- Fully functional Flask-based web interface for user interaction.

---

## Setup Instructions

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed locally
- GPU support (optional for faster inference with certain models)

---

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>

python -m venv env
source env/bin/activate  # On Windows: .\env\Scripts\activate

pip install -r requirement.txt

4. Install Ollama
Follow the instructions on the Ollama website to install the CLI tool.

ollama pull llama3.2

6. Run the Application
Start the Flask server:

bash
Copy code
python app.py
The application will run on http://127.0.0.1:5000.


Code Explanation
1. Data Processing
PDFs: Splits large documents into manageable chunks using RecursiveCharacterTextSplitter.
Audio/Video: Converts MP3s and videos to text using speech_recognition.
Images: Generates simple descriptions using the CLIP model.
2. Embedding and Storage
Stores document embeddings in a local ChromaDB database.
Supports retrieval of relevant content based on user queries.
3. Query and Response
Extracts context from the knowledge base using the ChromaDB collection.
Sends the context and user query to the Ollama LLM for response generation.
4. Web Interface
Simple Flask app allows users to input questions and view responses.
Example Usage
Place your files (PDFs, MP3s, videos, and images) in the datasets directory.
Start the server (python app.py).
Open your browser at http://127.0.0.1:5000.
Enter your question and view the AI-generated response.
Future Improvements
Add support for additional file formats.
Improve error handling and edge-case management.
Extend the interface to include file upload functionality.
Feel free to contribute or suggest improvements by creating an issue or pull request!
