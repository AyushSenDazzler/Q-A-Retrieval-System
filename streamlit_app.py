import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
import ollama
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.llms import Ollama

from typing import List, Dict, Any, Generator

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(_name_)

# Streamlit Page Configuration
st.set_page_config(
    page_title="🔍 Ollama PDF RAG Explorer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

class RAGProcessor:
    def _init_(self, model_name: str = "llama3.1"):
        """
        Initialize RAG processor with embedding and language models
        
        Args:
            model_name (str): Name of the Ollama model to use
        """
        self.model_name = model_name
        self.embeddings = OllamaEmbeddings(
            base_url='http://localhost:11434/', 
            model="nomic-embed-text", 
            show_progress=True
        )
        self.llm = Ollama(model=self.model_name)

    def create_vector_db(self, file_path: str, chunk_size: int = 500, chunk_overlap: int = 100) -> Chroma:
        """
        Create vector database from PDF
        
        Args:
            file_path (str): Path to PDF file
            chunk_size (int): Size of text chunks
            chunk_overlap (int): Overlap between chunks
        
        Returns:
            Chroma: Vector database
        """
        start_time = time.time()
        st.sidebar.info("📊 Processing Document...")
        
        # Document Loading
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        st.sidebar.info(f"✅ Loaded {len(pages)} pages")
        
        # Text Splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(pages)
        st.sidebar.info(f"🧩 Split into {len(chunks)} chunks")
        
        # Vector Store Creation
        vector_db = Chroma.from_documents(
            documents=chunks, 
            embedding=self.embeddings,
            collection_name="pdf_rag_collection"
        )
        
        end_time = time.time()
        st.sidebar.success(f"🌟 Vector DB Created in {end_time - start_time:.2f} seconds")
        
        return vector_db

    def generate_multi_query(self, question: str, num_queries: int = 3) -> List[str]:
        """
        Generate multiple query variants
        
        Args:
            question (str): Original user question
            num_queries (int): Number of query variants to generate
        
        Returns:
            List[str]: Generated query variants
        """
        multi_query_prompt = PromptTemplate(
            input_variables=["question"],
            template=f"""Generate {num_queries} different ways to ask the same question 
            that might help retrieve more comprehensive context:
            Original Question: {{question}}"""
        )
        
        query_generator = multi_query_prompt | self.llm | StrOutputParser()
        multi_queries = query_generator.invoke({"question": question}).split('\n')
        
        return [q.strip() for q in multi_queries if q.strip()]

    def retrieve_context(self, vector_db: Chroma, queries: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve context for multiple queries
        
        Args:
            vector_db (Chroma): Vector database
            queries (List[str]): List of queries
            top_k (int): Number of top similar documents to retrieve
        
        Returns:
            List[Dict[str, Any]]: Retrieved context documents
        """
        retriever = vector_db.as_retriever(search_kwargs={"k": top_k})
        
        all_contexts = []
        for query in queries:
            context_docs = retriever.get_relevant_documents(query)
            all_contexts.extend(context_docs)
        
        # Remove duplicates while preserving order
        unique_contexts = []
        seen = set()
        for doc in all_contexts:
            if doc.page_content not in seen:
                unique_contexts.append(doc)
                seen.add(doc.page_content)
        
        return unique_contexts

    def generate_response(self, question: str, context: List[Dict[str, Any]]) -> Generator[str, None, None]:
        """
        Generate response using retrieved context
        
        Args:
            question (str): User's question
            context (List[Dict[str, Any]]): Retrieved context documents
        
        Yields:
            str: Response chunks
        """
        context_text = "\n\n".join([doc.page_content for doc in context])
        
        response_template = """Answer the question based ONLY on the following context:
        Context:
        {context}

        Question: {question}

        Rules:
        - Be concise and direct
        - If the answer is not in the context, say "I cannot find the answer in the provided context."
        - Cite the sources (page numbers) where you found the information
        """
        
        prompt = ChatPromptTemplate.from_template(response_template)
        
        chain = (
            {"context": lambda x: context_text, "question": RunnablePassthrough()}
            | prompt
            | self.llm
        )
        
        response_chunks = ""
        for chunk in chain.stream(question):
            response_chunks += chunk
            yield chunk

def main():
    st.title("🧠 Q & A Retrieval System")
    
    # Sidebar for configuration
    st.sidebar.header("🔧 Configuration")
    model_options = ollama.list().models
    selected_model = st.sidebar.selectbox(
        "Select LLM Model", 
        [m.model for m in model_options]
    )
    
    # File Uploader
    uploaded_file = st.file_uploader(
        "📄 Upload PDF", 
        type=['pdf'], 
        help="Upload a PDF to start exploring"
    )
    
    if uploaded_file:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        
        # Initialize RAG Processor
        rag_processor = RAGProcessor(model_name=selected_model)
        
        # Create Vector DB
        vector_db = rag_processor.create_vector_db(temp_file_path)
        
        # PDF Preview
        st.subheader("📖 PDF Preview")
        with pdfplumber.open(temp_file_path) as pdf:
            first_page = pdf.pages[0]
            img = first_page.to_image(resolution=200)
            st.image(img.original, caption="First Page Preview", use_column_width=True)
        
        # Query Interface
        st.subheader("❓ Ask a Question")
        user_question = st.text_input("Enter your question about the document")
        
        if user_question:
            # Expander for showing queries and retrieved context
            with st.expander("🔍 Query Exploration"):
                st.write("### 🧩 Generated Query Variants")
                multi_queries = rag_processor.generate_multi_query(user_question)
                for i, query in enumerate(multi_queries, 1):
                    st.info(f"Query {i}: {query}")
                
                st.write("### 📄 Retrieved Context")
                retrieved_context = rag_processor.retrieve_context(vector_db, multi_queries)
                for i, doc in enumerate(retrieved_context, 1):
                    st.text_area(
                        f"Context Snippet {i}", 
                        value=doc.page_content, 
                        height=100
                    )
            
            # Response Generation
            st.subheader("💬 Response")
            response_placeholder = st.empty()
            
            full_response = ""
            for chunk in rag_processor.generate_response(user_question, retrieved_context):
                full_response += chunk
                response_placeholder.markdown(full_response)
        
        # Clean up temporary file
        os.unlink(temp_file_path)

if _name_ == "_main_":
    main()
