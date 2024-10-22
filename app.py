import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import pandas as pd
import speech_recognition as sr  # Voice recognition library
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain, LLMChain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain.document_loaders import WebBaseLoader  # Import the web loader

# Initialize session state for page navigation and chat history
if 'page' not in st.session_state:
    st.session_state.page = 'landing'
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = None
if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = None
# Remove default initialization to prevent overwriting
# if 'current_chat_title' not in st.session_state:
#     st.session_state.current_chat_title = "No Active Chat"

def main():
    load_dotenv()
    st.set_page_config(page_title="ChatDocx", page_icon="💬", layout="wide")
    
    # Initialize MongoDB connection
    initialize_mongodb()
    
    if st.session_state.page == 'landing':
        show_landing_page()
    elif st.session_state.page == 'chat':
        show_chat_page()

def initialize_mongodb():
    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        st.error("MongoDB URI is not set. Please check your .env file.")
        st.stop()
    
    try:
        client = MongoClient(mongo_uri)
        db = client['chatdocx']
        st.session_state.mongodb_client = client
        st.session_state.mongodb_db = db
        st.session_state.chats_collection = db['chats']
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {e}")
        st.stop()

def show_landing_page():
    # Custom CSS for styling and typewriter effect
    st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .line {
        text-align:center;
        color:#2c3e50;
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
        if st.button("Try Notebook "):
            st.session_state.page = 'chat'
            
    st.markdown(f"""<div class="line">double click to create notebook</div>""", unsafe_allow_html=True)
            

def show_chat_page():
    # Display the currently active chat title at the top
    current_chat_title = st.session_state.get('current_chat_title', 'No Active Chat')
    st.sidebar.markdown(f"## Current Chat: **{current_chat_title}**")
    
    st.title("📄💬 ChatDocx - Intelligent Document Query System")
    st.write("Upload your documents or enter a URL to start asking questions!")

    # Sidebar for file upload, URL input, and chat history
    with st.sidebar:
        st.header("Add Documents or URLs")
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose your files",
            type=["pdf", "html", "xml", "csv"],
            accept_multiple_files=True
        )
        if uploaded_files:
            st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")
        else:
            st.info("You can upload files to process.")

        # URL input
        url_input = st.text_input("Enter a URL to process:", key='url_input')
        if url_input:
            st.success(f"URL '{url_input}' entered.")
        else:
            st.info("You can enter a URL to process.")

        if (uploaded_files or url_input) and st.button("Process Documents/URL"):
            process_documents(uploaded_files, url_input)
        else:
            st.info("Please upload files or enter a URL to proceed.")

        st.markdown("---")
        st.header("Chat Histories")

        # Fetch chat histories from MongoDB
        chat_histories = fetch_chat_histories()
        chat_options = {str(chat['_id']): chat['title'] for chat in chat_histories}

        if chat_options:
            selected_chat_id = st.selectbox("Select a Chat History", options=list(chat_options.keys()), format_func=lambda x: chat_options[x], key='chat_selectbox')
            if st.button("Load Selected Chat"):
                load_chat_history(ObjectId(selected_chat_id))
        else:
            st.info("No chat histories found.")

        st.markdown("---")
        st.header("Create New Chat")
        with st.form(key='new_chat_form'):
            default_title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            chat_title = st.text_input("Enter a title for the new chat:", placeholder="Enter chat title here...", key='new_chat_title')
            submit_button = st.form_submit_button(label='Create New Chat')

        if submit_button:
            if chat_title.strip() == "":
                chat_title = default_title
            create_new_chat(chat_title)
            # Update the current chat title in session state
            st.session_state.current_chat_title = chat_title

    # Check if knowledge base is built
    if st.session_state.knowledge_base is not None:
        # Main content area
        st.header("Chat with your documents")
        chat_container = st.container()

        # User input options: Text input or Voice input
        user_question = st.text_input("Enter your question or request:", key="input")

        # Add a button to capture voice input
        if st.button("Use Voice Input"):
            user_question = get_voice_input()
            if user_question:
                st.write(f"*You (voice):* {user_question}")

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
                answer += f"\n\n*Citations:*\n{citations}"

            # Save the conversation to session state
            st.session_state.conversation.append({"question": user_question, "answer": answer})

            # Save the conversation to MongoDB
            save_conversation_to_db()

        # Display conversation history
        if st.session_state.conversation:
            for chat in st.session_state.conversation[::-1]:  # Reverse to show latest messages at the bottom
                st.write(f"*You:* {chat['question']}")
                st.write(f"*Assistant:* {chat['answer']}")
                st.write("---")
    else:
        st.info("Please upload and process documents or enter a URL to start chatting.")

def fetch_chat_histories():
    """Fetch all chat histories from MongoDB."""
    try:
        chats = list(st.session_state.chats_collection.find().sort("last_updated", -1))
        return chats
    except Exception as e:
        st.error(f"Error fetching chat histories: {e}")
        return []

def load_chat_history(chat_id):
    """Load a specific chat history from MongoDB."""
    try:
        chat = st.session_state.chats_collection.find_one({"_id": chat_id})
        if chat:
            st.session_state.conversation = chat.get('conversation', [])
            st.session_state.current_chat_id = chat_id
            st.session_state.current_chat_title = chat.get('title', 'Untitled Chat')
            st.success(f"Loaded chat: {st.session_state.current_chat_title}")
            st.session_state.knowledge_base = None  # Reset knowledge base to force re-processing
            # Optionally, you can rebuild the knowledge base if documents are associated with the chat
        else:
            st.error("Chat history not found.")
    except Exception as e:
        st.error(f"Error loading chat history: {e}")

def create_new_chat(chat_title):
    """Create a new chat session."""
    try:
        # Create a new chat document in MongoDB with the provided title
        new_chat = {
            "title": chat_title,
            "conversation": [],
            "created_at": datetime.now(),
            "last_updated": datetime.now()
        }
        result = st.session_state.chats_collection.insert_one(new_chat)
        st.session_state.current_chat_id = result.inserted_id
        st.session_state.current_chat_title = chat_title
        # Clear the current conversation in the UI
        st.session_state.conversation = []
        # Reset the knowledge base
        st.session_state.knowledge_base = None
        st.success(f"New chat '{chat_title}' created successfully.")
    except Exception as e:
        st.error(f"Error creating new chat: {e}")

def save_conversation_to_db():
    """Save the current conversation to MongoDB."""
    try:
        if st.session_state.current_chat_id:
            # Update existing chat
            st.session_state.chats_collection.update_one(
                {"_id": st.session_state.current_chat_id},
                {
                    "$set": {
                        "conversation": st.session_state.conversation,
                        "last_updated": datetime.now()
                    }
                }
            )
        else:
            # Create a new chat with the current chat title
            chat_title = st.session_state.get('current_chat_title', f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            new_chat = {
                "title": chat_title,
                "conversation": st.session_state.conversation,
                "created_at": datetime.now(),
                "last_updated": datetime.now()
            }
            result = st.session_state.chats_collection.insert_one(new_chat)
            st.session_state.current_chat_id = result.inserted_id
            st.session_state.current_chat_title = chat_title
            st.success(f"Conversation saved to database with title '{chat_title}'.")
    except Exception as e:
        st.error(f"Error saving conversation: {e}")

def process_documents(uploaded_files, url_input):
    docs = []  # List to hold Document objects with metadata

    # Process uploaded files
    if uploaded_files:
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

    # Process URL input
    if url_input:
        url_docs = process_url(url_input)
        docs.extend(url_docs)

    if not docs:
        st.error("No documents were processed.")
        return

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

    # Store the knowledge base in session state for future use
    st.session_state.knowledge_base = knowledge_base

    st.success("Knowledge base created successfully!")

def process_pdf(uploaded_file):
    # Extract text from a PDF file
    reader = PdfReader(uploaded_file)
    num_pages = len(reader.pages)

    docs = []
    for page_num in range(num_pages):
        page = reader.pages[page_num]
        text = page.extract_text()

        if text:  # Ensure text is not None
            metadata = {
                'source': uploaded_file.name,
                'page': page_num + 1,
            }
            doc = Document(
                page_content=text,
                metadata=metadata
            )
            docs.append(doc)
    return docs

def process_html_xml(uploaded_file):
    # Extract text from HTML or XML files using BeautifulSoup
    try:
        content = uploaded_file.read().decode('utf-8')
    except UnicodeDecodeError:
        content = uploaded_file.read().decode('latin-1')
    soup = BeautifulSoup(content, 'html.parser')
    text = soup.get_text()

    metadata = {
        'source': uploaded_file.name
    }
    doc = Document(
        page_content=text,
        metadata=metadata
    )
    return [doc]

def process_csv(uploaded_file):
    # Extract data from CSV files using pandas
    try:
        df = pd.read_csv(uploaded_file)
        text = df.to_string()
    except Exception as e:
        st.error(f"Error processing CSV file {uploaded_file.name}: {e}")
        text = ""

    metadata = {
        'source': uploaded_file.name
    }
    doc = Document(
        page_content=text,
        metadata=metadata
    )
    return [doc]

def process_url(url):
    # Load content from the URL using WebBaseLoader
    try:
        loader = WebBaseLoader(url)
        url_docs = loader.load()

        # Add source metadata to each document
        for doc in url_docs:
            doc.metadata['source'] = url

        return url_docs
    except Exception as e:
        st.error(f"Error processing URL {url}: {e}")
        return []

def get_voice_input():
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Capture the voice input
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)

    # Convert the voice input to text
    try:
        st.write("Recognizing...")
        query = recognizer.recognize_google(audio)
        st.write(f"User said: {query}")
        return query
    except sr.UnknownValueError:
        st.error("Sorry, I could not understand the audio.")
        return ""
    except sr.RequestError:
        st.error("Sorry, there was an issue with the recognition service.")
        return ""

if __name__ == '__main__':
    main()
