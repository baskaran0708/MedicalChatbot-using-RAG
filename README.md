# Medical Chatbot with LLMs, LangChain, Pinecone & Flask

An intelligent medical chatbot powered by Large Language Models (LLMs), LangChain, Pinecone vector database, and Flask. This chatbot provides medical information by leveraging Retrieval-Augmented Generation (RAG) from PDF documents and web search fallback for comprehensive answers.

##  Project Overview

This medical chatbot is designed to answer medical queries using a combination of:
1. **RAG (Retrieval-Augmented Generation)**: Retrieves relevant information from medical PDF documents stored in Pinecone vector database
2. **Web Search Fallback**: Uses DuckDuckGo search when the knowledge base doesn't contain sufficient information
3. **LLM Processing**: Utilizes Mistral-7B-Instruct model via OpenRouter for intelligent response generation

The chatbot features a modern, minimalist web interface with a medical-themed design and provides accurate, contextual medical information.

## ✨ Features

- 🔍 **Intelligent Query Processing**: Handles medical questions with context-aware responses
- 📚 **PDF Knowledge Base**: Extracts and indexes medical information from PDF documents
- 🌐 **Web Search Integration**: Falls back to web search for queries not covered in the knowledge base
- 🤖 **Advanced LLM**: Uses Mistral-7B-Instruct model for natural language understanding
- 💬 **Interactive Chat Interface**: Clean, responsive web UI with chat history
- 📝 **Comprehensive Logging**: Detailed request/response logging for monitoring and debugging
- 🎨 **Modern UI Design**: Minimalist, Apple-inspired design with medical color scheme
- ⚡ **Fast Retrieval**: Vector similarity search using Pinecone for quick information retrieval

## 🏗️ Architecture

```
┌─────────────┐
│   User      │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│      Flask Web Application          │
│  ┌───────────────────────────────┐  │
│  │   Chat Interface (HTML/CSS)   │  │
│  └───────────────────────────────┘  │
└──────────────┬──────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│      LangChain RAG Pipeline          │
│  ┌────────────────────────────────┐  │
│  │  1. Query Embeddings           │  │
│  │  2. Pinecone Vector Search     │  │
│  │  3. Context Retrieval          │  │
│  │  4. LLM Response Generation    │  │
│  └────────────────────────────────┘  │
└──────────────┬───────────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌─────────────┐  ┌──────────────┐
│  Pinecone   │  │  OpenRouter  │
│  Vector DB  │  │  (Mistral)   │
└─────────────┘  └──────────────┘
       │
       ▼ (fallback)
┌─────────────────┐
│  DuckDuckGo     │
│  Web Search     │
└─────────────────┘
```

## 🛠️ Tech Stack

### Backend
- **Flask**: Web framework for serving the application
- **LangChain**: Framework for building LLM applications
- **Pinecone**: Vector database for semantic search
- **HuggingFace Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- **OpenRouter**: API gateway for accessing Mistral-7B-Instruct model
- **DuckDuckGo Search**: Web search fallback mechanism

### Frontend
- **HTML5/CSS3**: Modern, responsive UI
- **JavaScript**: Interactive chat functionality
- **Custom CSS**: Minimalist, medical-themed design

### Data Processing
- **PyPDF**: PDF document loading and parsing
- **RecursiveCharacterTextSplitter**: Intelligent text chunking

## 📦 Prerequisites

Before you begin, ensure you have the following:

- **Python 3.8+** installed
- **pip** package manager
- **Git** for cloning the repository
- API Keys for:
  - [Pinecone](https://www.pinecone.io/) (for vector database)
  - [OpenRouter](https://openrouter.ai/) (for LLM access)
  - [OpenAI](https://openai.com/) (optional, for alternative models)

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/medical-chatbot.git
cd medical-chatbot
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv medibot_venv
medibot_venv\Scripts\activate

# Linux/Mac
python3 -m venv medibot_venv
source medibot_venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install the Package

```bash
pip install -e .
```

## ⚙️ Configuration

### Step 1: Create Environment File

Create a `.env` file in the root directory:

```bash
# .env
PINECONE_API_KEY=your_pinecone_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

> **Important**: Never commit your `.env` file to version control. It's already included in `.gitignore`.

### Step 2: Add Medical PDF Documents

Place your medical PDF documents in the `data/` directory:

```bash
data/
  ├── medical_book_1.pdf
  ├── medical_book_2.pdf
  └── ...
```

### Step 3: Create Pinecone Index and Store Embeddings

Run the indexing script to process PDFs and create the vector database:

```bash
python store_index.py
```

This script will:
1. Load all PDF files from the `data/` directory
2. Split documents into chunks (500 characters with 20 character overlap)
3. Generate embeddings using HuggingFace model
4. Create a Pinecone index named "medical-bot"
5. Upload embeddings to Pinecone

> **Note**: This process may take several minutes depending on the size of your PDF documents.

## 🎮 Usage

### Running the Application

Start the Flask development server:

```bash
python app.py
```

The application will be available at: **http://localhost:8080**

### Using the Chatbot

1. Open your web browser and navigate to `http://localhost:8080`
2. You'll see the landing page with a "Start Chat" button
3. Click the button to enter the chat interface
4. Type your medical question in the input box
5. Press Enter or click Send
6. The chatbot will respond with relevant medical information

### Example Queries

- "What are the symptoms of diabetes?"
- "How is hypertension treated?"
- "What causes migraine headaches?"
- "Explain the difference between Type 1 and Type 2 diabetes"

## 📁 Project Structure

```
medical-chatbot/
│
├── app.py                      # Main Flask application
├── store_index.py              # Script to create Pinecone index
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup file
├── .env                        # Environment variables (not in repo)
├── .gitignore                  # Git ignore rules
├── Dockerfile                  # Docker configuration
│
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── helper.py               # Helper functions for PDF processing
│   └── prompt.py               # System prompts for LLM
│
├── data/                       # Medical PDF documents
│   └── *.pdf
│
├── templates/                  # HTML templates
│   ├── index.html              # Landing page
│   └── chat.html               # Chat interface
│
├── static/                     # Static assets
│   ├── css/
│   ├── js/
│   └── images/
│
├── logs/                       # Application logs
│   └── chatbot.log
│
└── research/                   # Research notebooks
    └── trials.ipynb
```

## 🔍 How It Works

### 1. Document Processing Pipeline

```python
# Load PDFs from data/ directory
documents = load_pdf_file(data='data/')

# Filter metadata to keep only source information
filtered_docs = filter_to_minimal_docs(documents)

# Split into chunks (500 chars, 20 overlap)
text_chunks = text_split(filtered_docs)

# Generate embeddings (384 dimensions)
embeddings = download_hugging_face_embeddings()

# Store in Pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name="medical-bot",
    embedding=embeddings
)
```

### 2. Query Processing Flow

1. **User Input**: User submits a medical question
2. **Greeting Detection**: Simple greetings are handled without RAG
3. **RAG Retrieval**: 
   - Query is embedded using the same HuggingFace model
   - Top 3 similar chunks are retrieved from Pinecone
   - Context is passed to the LLM with the query
4. **Response Generation**: Mistral-7B-Instruct generates an answer
5. **Cleanup**: Model artifacts (tags like `<s>`, `[INST]`) are removed
6. **Fallback**: If RAG response is insufficient:
   - DuckDuckGo web search is performed
   - Search results are summarized by the LLM
   - Response includes "(Source: Web search)" notation

### 3. Logging System

All interactions are logged with unique request IDs:

```
[20251202_211230_123456] NEW REQUEST: 'What is diabetes?'
[20251202_211230_123456] Client IP: 127.0.0.1
[20251202_211230_123456] Attempting RAG retrieval...
[20251202_211230_123456] SOURCE: RAG (PDF Knowledge Base)
[20251202_211230_123456] RESPONSE: Diabetes is a chronic condition...
[20251202_211230_123456] REQUEST COMPLETED
```

## 🌐 Deployment

### Docker Deployment

Build and run using Docker:

```bash
# Build the image
docker build -t medical-chatbot .

# Run the container
docker run -p 8080:8080 --env-file .env medical-chatbot
```

### AWS Deployment (EC2)

1. Launch an EC2 instance (Ubuntu 20.04 recommended)
2. Install Python and dependencies
3. Clone the repository
4. Set up environment variables
5. Run the application with a production WSGI server:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8080 app:app
```

### Environment Variables for Production

```bash
export FLASK_ENV=production
export FLASK_DEBUG=False
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'src'`
- **Solution**: Run `pip install -e .` to install the package in editable mode

**Issue**: Pinecone connection errors
- **Solution**: Verify your API key and ensure the index name matches in both `store_index.py` and `app.py`

**Issue**: LLM responses contain artifacts like `<s>` or `[INST]`
- **Solution**: The cleanup code in `app.py` should handle this. If issues persist, add more cleanup patterns

**Issue**: Web search not working
- **Solution**: Check your internet connection and ensure `duckduckgo-search` is properly installed

**Issue**: Slow response times
- **Solution**: 
  - Reduce `max_tokens` in the LLM configuration
  - Decrease `k` value in retriever (currently 3)
  - Consider upgrading Pinecone plan for faster queries

### Logs

Check the logs for detailed error information:

```bash
tail -f logs/chatbot.log
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide for Python code
- Add docstrings to all functions
- Update README.md for significant changes
- Test thoroughly before submitting PR

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** for the excellent LLM framework
- **Pinecone** for vector database infrastructure
- **HuggingFace** for embeddings models
- **OpenRouter** for LLM API access
- **Mistral AI** for the Mistral-7B-Instruct model

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**⚠️ Disclaimer**: This chatbot is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns.
