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
```
### 3. Install and Set Up Ollama

1. **Download and install Ollama** from [ollama.com](https://ollama.com/).
2. Once Ollama is installed, run **Ollama** locally to serve the **Llama 3.1-8b-Instruct** model:

   ```bash
   ollama run
   ```

### 3. Install and Set Up Ollama

1. **Download and install Ollama** from [ollama.com](https://ollama.com/).
2. Once Ollama is installed, run **Ollama** locally to serve the **Llama 3** model:

   ```bash
   ollama run
  

### 4. Running the Application

Once everything is set up, you can run the Streamlit app with the following command:

```bash
streamlit run app.py
```
## 🧑‍💻 Usage

### 1. Upload a PDF
- Click the **Upload PDF** button to upload any PDF document. The app will process the document, split it into chunks, and store the chunks in a vector database.

### 2. Select a Model
- After uploading, select one of the available **Ollama models** (e.g., **Llama 3**). The application will use this model to process your queries and generate responses.

### 3. Ask Questions
- Once the PDF is processed, type your question into the chat input box. The system will query the vector database for relevant document chunks and provide an answer based on the PDF content.

### 4. View PDF Pages
- You can view the uploaded PDF pages as images. Zoom in or out using the slider to better view the content.

### 5. Delete Vector Database
- If you want to clear the database or reset the application, simply click on the **Delete Collection** button to remove the vector database.

---

## 📊 How It Works

### **Step 1: PDF Processing**
- When a PDF is uploaded, the system extracts the text from it and splits the text into manageable chunks. These chunks are then embedded into vector representations using **OllamaEmbeddings**.

### **Step 2: Vector Database**
- The **Chroma** vector store is used to store the embeddings of the document chunks. This allows for fast retrieval of the most relevant chunks based on the user's query.

### **Step 3: Multi-query Retriever**
- When a user asks a question, the **MultiQueryRetriever** generates alternative phrasing of the question, allowing for more accurate retrieval from the vector store. This helps overcome limitations of traditional similarity searches.

### **Step 4: Generating Answers**
- The retrieved chunks are passed to the language model (**Llama 3**) to generate the answer based only on the relevant context from the document.

### **Step 5: RAG (Retrieval-Augmented Generation)**
- The system combines **document retrieval** and **generation** to provide a contextualized response. The answer is grounded in the document content, and the relevant excerpts used in the response are shown to the user.

---




## 🔗 Links

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Ollama Documentation](https://ollama.com/docs)
- [LangChain Documentation](https://langchain.com/docs/)
- [Chroma Documentation](https://www.trychroma.com/docs)
