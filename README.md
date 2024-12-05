# RAG Assignment Project

A Retrieval-Augmented Generation (RAG) application that creates an intelligent knowledge base from multiple data sources including PDFs, audio, video, and images. The system leverages ChromaDB for vector storage and Ollama LLM for generating context-aware responses.

## Features

- Multi-modal data processing and indexing:
  - PDF documents using LangChain document loaders
  - Audio files (MP3) with speech-to-text conversion
  - Video files with audio extraction and transcription
  - Images with CLIP-based description generation
- Vector storage using ChromaDB
- Context-aware response generation with Ollama LLM
- Interactive web interface built with Flask

## Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed locally
- CUDA-compatible GPU (optional, for improved performance)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create and activate a virtual environment:
```bash
python -m venv env

# For Unix/macOS
source env/bin/activate

# For Windows
.\env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install and configure Ollama:
   - Install Ollama following the instructions at [ollama.ai](https://ollama.ai)
   - Pull the required model:
```bash
ollama pull llama3.2
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Access the web interface at `http://127.0.0.1:5000`

3. Upload your documents or enter queries through the web interface

## Architecture

### Data Processing Pipeline

1. **Document Processing**
   - PDF processing: Chunks documents using RecursiveCharacterTextSplitter
   - Audio processing: Converts speech to text using speech_recognition
   - Video processing: Extracts audio and performs transcription
   - Image processing: Generates descriptions using CLIP model

2. **Vector Storage**
   - Generates embeddings for all processed content
   - Stores vectors in ChromaDB for efficient retrieval
   - Implements similarity search for context retrieval

3. **Response Generation**
   - Retrieves relevant context from ChromaDB based on user queries
   - Uses Ollama LLM to generate contextually appropriate responses
   - Implements response filtering and post-processing

## Project Structure

```
rag-project/
├── app.py              # Flask application and preprocess
├── datasets/           # Input data directory
├── templates/         # Flask templates
└── requirements.txt   # Python dependencies
```

## Development

### Setting Up Development Environment

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Set up pre-commit hooks:
```bash
pre-commit install
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Future Improvements

- Support for additional file formats
- Enhanced error handling and logging
- Bulk file upload functionality
- API documentation
- Performance optimizations
- Docker containerization

## License

## Acknowledgments

- [Ollama](https://ollama.ai/) for the LLM implementation
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [LangChain](https://langchain.com/) for document processing
