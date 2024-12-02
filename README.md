# 🎈 Ollama PDF-based Retrieval-Augmented Generation (RAG) with Streamlit

This **Streamlit** application enables you to upload PDF documents, process them into vector embeddings, and interact with the content through a question-answering interface powered by **Ollama’s Llama 3** language models. The system uses **Retrieval-Augmented Generation (RAG)** techniques to generate answers based on context derived from the uploaded PDF document.

With this tool, users can seamlessly:
- Upload PDFs
- Extract vector-based information
- Query the document’s content in a conversational format

---

## 🚀 Features

- **Upload PDFs**: Easily upload PDF files for processing and retrieval.
- **Vector Database**: Automatically splits the PDF into chunks, embeds them, and stores them in a **Chroma** vector store for fast querying.
- **Model Selection**: Choose from available local models powered by **Ollama**.
- **Query-Based Interactions**: Ask questions based on the document’s content, and get context-aware responses.
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

---

## 📥 Installation & Setup

### 1. Prerequisites

Make sure you have the following installed:

- **Python 3.8+**
- **Ollama** running locally (required for the Llama model)

### 2. Install Dependencies

Clone this repository and install the required dependencies via `pip`:

```bash
git clone https://github.com/yourusername/ollama-pdf-rag-streamlit.git
cd ollama-pdf-rag-streamlit
pip install -r requirements.txt
