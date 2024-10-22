# **ChatDocx** 💬

### An AI-Powered Document Chat Assistant

## Google Drive Link(PPT & Video)

https://drive.google.com/drive/folders/1-XxJpQvPYopeqxU09IhZcsnNQ0xgM7I0?usp=drive_link

---

## **Overview** 🎯

**ChatDocx** is a powerful AI-driven tool designed to make document interaction intuitive and dynamic. It allows users to upload documents or provide URLs for web-based content, from which they can ask questions conversationally, either by typing or using voice commands. The assistant extracts relevant information and provides responses complete with citations to ensure transparency and credibility. Additionally, the system stores and manages multiple chat sessions, which can be retrieved from a MongoDB database.

Whether you're a student, researcher, or professional working with multiple documents, **ChatDocx** empowers you to find information quickly and accurately.

---

## **Features** 🚀

1. **Document Upload and URL Processing**:
   - Supports file uploads in formats such as PDF, HTML, XML, and CSV.
   - Allows querying of documents or content retrieved from URLs.
2. **Conversational Query System**:
   - Ask questions directly related to the content of your uploaded documents.
   - Receive accurate, context-based answers with citations from the source material.
3. **Voice Input for Queries**:
   - Enables voice-based queries using speech recognition technology.
4. **Multiple Chat Sessions**:
   - Manage multiple chat sessions that are stored and retrieved from a MongoDB database.
5. **Practice Questions Generation**:

   - Based on the document content, users can generate relevant practice questions to test their understanding.

6. **User-Friendly Interface**:
   - Built using Streamlit, providing an intuitive and interactive web interface for easy usage.
7. **NLP-Powered Responses**:
   - Uses advanced Natural Language Processing (NLP) models via LangChain and OpenAI APIs to understand and respond to questions.

---

## **Project Architecture** 🏗

1. **Frontend**:

   - Developed using **Streamlit**, a Python-based framework to build interactive and visually appealing web applications.

2. **Backend**:
   - Powered by **Python** to handle document parsing, voice input, and question answering.
   - Integrates **LangChain** for document comprehension and **OpenAI API** for conversational responses.
3. **Document Handling**:
   - Document parsing is handled by libraries such as **PyPDF2** for PDFs, **BeautifulSoup** for HTML/XML, and custom parsing for CSVs.
4. **Database**:
   - Chat histories and session data are stored in **MongoDB**, ensuring that users can continue their conversations at any time.
5. **Voice Recognition**:
   - **SpeechRecognition** library enables voice-based questions, making the interaction even more seamless.

---

## **Tech Stack** 🛠

- **Frontend**: Streamlit
- **Backend**: Python, LangChain, OpenAI API
- **Database**: MongoDB
- **Libraries**: PyPDF2, BeautifulSoup, SpeechRecognition
- **NLP Models**: OpenAI (GPT-based models), LangChain

---

## **Installation Guide** 📦

Follow these steps to get **ChatDocx** running on your local machine:

### Run The Project

```bash
git clone https://github.com/yourusername/chatdocx.git
cd chatdocx
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
