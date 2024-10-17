from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import os
import pandas as pd

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain, LLMChain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate

# Initialize session state for page navigation and chat history
if 'page' not in st.session_state:
    st.session_state.page = 'landing'
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = None

def main():
    load_dotenv()
    st.set_page_config(page_title="ChatDocx", page_icon="💬", layout="wide")
    
    if st.session_state.page == 'landing':
        show_landing_page()
    elif st.session_state.page == 'chat':
        show_chat_page()

def show_landing_page():
    # Custom CSS for styling and typewriter effect
    st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .landing-title {
        font-size: 60px;
        font-weight: bold;
        text-align: center;
        margin-top: 100px;
        color: #2c3e50;
    }
    .typewriter h2 {
        overflow: hidden;
        border-right: .15em solid #2c3e50;
        white-space: nowrap;
        margin: 0 auto;
        letter-spacing: .15em;
        animation: 
            typing 3.5s steps(40, end),
            blink-caret .75s step-end infinite;
        font-size: 28px;
        color: #34495e;
    }
    @keyframes typing {
        from { width: 0 }
        to { width: 100% }
    }
    @keyframes blink-caret {
        from, to { border-color: transparent }
        50% { border-color: #2c3e50 }
    }
    .stButton>button {
        background-color: #2c3e50;
        color: white;
        padding: 15px 30px;
        font-size: 18px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        margin:auto;
    }
    .stButton>button:hover {
        background-color: #34495e;
    }
    .stColumn {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    # Landing Page Content
    st.markdown(f"""
    <div class="landing-title">ChatDocx</div>
    <div class="typewriter"><h2>Your Intelligent Document Assistant</h2></div>
    """, unsafe_allow_html=True)

    # Center the button
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        # Try Notebook Button
        if st.button("Try Notebook"):
            st.session_state.page = 'chat'

def show_chat_page():
    st.title("📄💬 ChatDocx - Intelligent Document Query System")
    st.write("Upload your documents and start asking questions!")

    # Sidebar for file upload
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose your files",
            type=["pdf", "html", "xml", "csv"],
            accept_multiple_files=True
        )
        if uploaded_files:
            st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")
            if st.button("Process Documents"):
                process_documents(uploaded_files)
        else:
            st.info("Please upload files to proceed.")

    # Check if knowledge base is built
    if st.session_state.knowledge_base is not None:
        # Main content area
        st.header("Chat with your documents")
        chat_container = st.container()

        # User input
        user_question = st.text_input("Enter your question or request:", key="input")

        if user_question:
            # Use the existing knowledge base
            knowledge_base = st.session_state.knowledge_base
            docs = knowledge_base.similarity_search(user_question)
            llm = OpenAI()

            # Check if the user is requesting practice questions
            if "practice question" in user_question.lower() or "practice problems" in user_question.lower():
                # Combine the content from the retrieved documents
                context = "\n".join([doc.page_content for doc in docs])

                # Create a custom prompt for generating practice questions
                prompt_template = """
    You are an expert educator. Based on the following content, generate 3-5 practice questions that test understanding of the material. Make sure the questions are clear, relevant, and thought-provoking.

    Content:
    {context}

    Practice Questions:
    """
                PROMPT = PromptTemplate(
                    input_variables=["context"],
                    template=prompt_template,
                )

                chain = LLMChain(llm=llm, prompt=PROMPT)

                chain_input = {
                    "context": context,
                }

                with get_openai_callback() as cb:
                    response = chain(chain_input)
                    # print(cb)  # Optional: print callback info

                practice_questions = response['text']
                answer = practice_questions
            else:
                # Use RetrievalQAWithSourcesChain for regular QA
                retriever = knowledge_base.as_retriever()

                chain = RetrievalQAWithSourcesChain.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True
                )

                with get_openai_callback() as cb:
                    result = chain({"question": user_question})
                    # print(cb)  # Optional: print callback info

                answer = result['answer']
                source_documents = result['source_documents']

                # Prepare citations
                unique_citations = set()
                for doc in source_documents:
                    source = doc.metadata.get('source', 'Unknown')
                    page = doc.metadata.get('page')
                    row = doc.metadata.get('row')
                    citation = f"Source: {source}"
                    if page:
                        citation += f", Page: {page}"
                    if row:
                        citation += f", Row: {row}"
                    unique_citations.add(citation)

                citations = "\n".join(unique_citations)
                answer += f"\n\n**Citations:**\n{citations}"

            # Save the conversation
            st.session_state.conversation.append({"question": user_question, "answer": answer})

        # Display conversation history
        if st.session_state.conversation:
            for chat in st.session_state.conversation[::-1]:  # Reverse to show latest messages at the bottom
                st.write(f"**You:** {chat['question']}")
                st.write(f"**Assistant:** {chat['answer']}")
                st.write("---")
    else:
        st.info("Please upload and process documents to start chatting.")

def process_documents(uploaded_files):
    docs = []  # List to hold Document objects with metadata
    for uploaded_file in uploaded_files:
        # Get the file extension
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        if file_extension == '.pdf':
            file_docs = process_pdf(uploaded_file)
        elif file_extension in ['.html', '.xml']:
            file_docs = process_html_xml(uploaded_file)
        elif file_extension == '.csv':
            file_docs = process_csv(uploaded_file)
        else:
            st.error(f"Unsupported file type: {uploaded_file.name}")
            continue  # Skip to the next file

        docs.extend(file_docs)  # Add the documents from this file to the list

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    docs_with_chunks = []
    for doc in docs:
        splits = text_splitter.split_text(doc.page_content)
        for chunk in splits:
            chunk_doc = Document(
                page_content=chunk,
                metadata=doc.metadata  # Preserve the original metadata
            )
            docs_with_chunks.append(chunk_doc)

    # Generate embeddings and create the vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    knowledge_base = FAISS.from_documents(docs_with_chunks, embeddings)

    # Save the knowledge base in session state
    st.session_state.knowledge_base = knowledge_base
    st.success("Documents processed successfully! You can now start chatting.")

def process_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    documents = []
    for page_num, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if text:
            metadata = {
                'source': pdf_file.name,
                'page': page_num + 1  # Page numbers start at 1
            }
            documents.append(Document(page_content=text, metadata=metadata))
    return documents

def process_html_xml(file):
    file_content = file.read()
    soup = BeautifulSoup(file_content, 'html.parser')  # Adjust parser if needed
    # Remove script and style elements
    for script_or_style in soup(['script', 'style']):
        script_or_style.extract()
    text = soup.get_text(separator='\n')

    metadata = {'source': file.name}

    # Optionally, split the text into chunks here if the file is large
    documents = [Document(page_content=text, metadata=metadata)]
    return documents

def process_csv(csv_file):
    df = pd.read_csv(csv_file)
    df = df.fillna('N/A')
    documents = []
    for index, row in df.iterrows():
        row_text = ', '.join(f"{column}: {value}" for column, value in row.items())
        metadata = {
            'source': csv_file.name,
            'row': index + 1  # Row numbers start at 1
        }
        documents.append(Document(page_content=row_text, metadata=metadata))
    return documents

if __name__ == '__main__':
    main()
