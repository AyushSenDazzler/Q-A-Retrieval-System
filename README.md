# 🎈 Ollama PDF-based Retrieval-Augmented Generation (RAG) with Streamlit

This **Streamlit** application enables you to upload PDF documents, process them into vector embeddings, and interact with the content through a question-answering interface powered by **Ollama's Llama 3.1 8B** language models. The system uses **Retrieval-Augmented Generation (RAG)** techniques to generate answers based on context derived from the uploaded PDF document.

With this tool, users can seamlessly:
- Upload PDFs
- Extract vector-based information
- Query the document's content in a conversational format

---

## 🚀 Features

- **Upload PDFs**: Easily upload PDF files for processing and retrieval.
- **Vector Database**: Automatically splits the PDF into chunks, embeds them, and stores them in a **Chroma** vector store for fast querying.
- **Model Selection**: Choose from available local models powered by **Ollama**.
- **Query-Based Interactions**: Ask questions based on the document's content, and get context-aware responses.
- **PDF Viewer**: View all PDF pages as images with a zoom feature.
- **Delete Vector Database**: Option to delete the vector database and reset the app.
- **Customizable Zoom**: Adjust the zoom level to view PDF content clearly.
- **Streamlit UI**: A user-friendly web interface built with **Streamlit**.

---

## 🛠️ Technologies Used

- **Streamlit**: For building the interactive user interface.
- **Ollama**: For serving **Llama 3** models locally.
- **LangChain**: For document processing, vector store management, and RAG-powered querying.
- **Chroma**: For handling vector embeddings and search queries.
- **PDFPlumber**: To extract and display PDF pages as images.
- **SQLite**: For session management (via **pysqlite3**).
- **Docker**: For running Ollama and llama3.1 model.

---

## 📥 Installation & Setup

### 1. Prerequisites

Make sure you have the following installed:

- **Python 3.10.0**
- **Docker** (optional, for Docker installation)
- **Git**

### 2. Clone the Repository

```bash
git clone https://github.com/yourusername/Q-A-Retrieval-System.git
cd Q-A-Retrieval-System
```

### 3. Install Ollama

#### Option 1: Docker Installation
To install Ollama using Docker, run:

```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

#### Option 2: Direct Installation
Download and install Ollama from [ollama.com](https://ollama.com/)

### 4. Pull Required Models

```bash
# Text Embedding Models
ollama pull nomic-embed-text
# For late Chunking
ollama pull jina/jina-embeddings-v2-base-en

# Language Models
ollama pull llama3.1
```

### 5. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 6. Running the Application

Start the Streamlit app:

```bash
streamlit run streamlit_app.py
```

## 🧑‍💻 Usage

### 1. Upload a PDF
- Click the **Upload PDF** button to upload any PDF document.
- The app will process the document, split it into chunks, and store them in a vector database.

### 2. Select a Model
- After uploading, choose from available **Ollama models**.
- The application will use the selected model to process queries and generate responses.

### 3. Ask Questions
- Type your question into the chat input box.
- The system retrieves relevant document chunks and provides an answer based on the PDF content.

### 4. View PDF Pages
- Browse uploaded PDF pages as images.
- Use the zoom slider to adjust the view.

### 5. Delete Vector Database
- Click the **Delete Collection** button to clear the database or reset the application.

---

## 📊 How It Works

### **Step 1: PDF Processing**
- Extracts text from the PDF and splits it into manageable chunks.
- Embeds chunks using **OllamaEmbeddings**.

### **Step 2: Vector Database**
- Uses **Chroma** vector store to store document chunk embeddings.
- Enables fast retrieval of relevant chunks.

### **Step 3: Multi-query Retriever**
- Generates alternative question phrasings for more accurate retrieval.
- Overcomes limitations of traditional similarity searches.

### **Step 4: Generating Answers**
- Passes retrieved chunks to the language model.
- Generates answers based on relevant document context.

### **Step 5: RAG (Retrieval-Augmented Generation)**
- Combines document retrieval and generation.
- Provides contextualized responses grounded in document content.

---

## 🔗 Useful Links

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Ollama Documentation](https://ollama.com/docs)
- [LangChain Documentation](https://langchain.com/docs/)
- [Chroma Documentation](https://www.trychroma.com/docs)

---

