# PDF Text Extraction and Knowledge Base Creation with FAISS and OpenAI Embeddings

## Google Drive Link
https://drive.google.com/drive/folders/1-XxJpQvPYopeqxU09IhZcsnNQ0xgM7I0?usp=drive_link

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Preparing the PDF](#preparing-the-pdf)
  - [Running the Script](#running-the-script)
- [Detailed Explanation](#detailed-explanation)
  - [Script Structure](#script-structure)
  - [Functions](#functions)
  - [Text Splitting Configuration](#text-splitting-configuration)
  - [Embedding and Vector Store Creation](#embedding-and-vector-store-creation)
- [Configuration](#configuration)
- [Error Handling and Troubleshooting](#error-handling-and-troubleshooting)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

This project provides a Python script that automates the process of extracting text from a PDF file, cleaning and formatting the text, splitting it into manageable chunks, embedding these chunks using OpenAI's embedding models, and storing them in a FAISS (Facebook AI Similarity Search) vector store. This setup is ideal for creating a searchable knowledge base from PDF documents, enabling efficient similarity searches and information retrieval.

---

## Features

- **PDF Text Extraction**: Utilizes `PyPDF2` to read and extract text from PDF files.
- **Text Cleaning**: Removes excessive newlines and ensures proper sentence spacing for readability.
- **Text Splitting**: Splits the cleaned text into chunks based on character count, facilitating efficient processing and embedding.
- **Embeddings Generation**: Uses OpenAI's embedding models to convert text chunks into vector representations.
- **Vector Store Creation**: Stores embeddings in a FAISS vector store, enabling fast similarity searches and retrievals.
- **Modular Design**: Structured in a way that allows easy modifications and extensions.

---

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **Python 3.7 or higher**: The script is written in Python and requires version 3.7 or later.
- **OpenAI API Key**: Necessary for generating text embeddings using OpenAI's models.
- **PDF File**: The PDF document you wish to process (`Sample_1.pdf` in this case).

---

## Installation

### 1. Clone the Repository

First, clone this repository to your local machine:

bash
git clone https://github.com/yourusername/pdf-knowledge-base.git
cd pdf-knowledge-base


### 2. Set Up a Virtual Environment (Optional but Recommended)

Create and activate a virtual environment to manage dependencies:

bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate


### 3. Install Required Dependencies

Install the necessary Python libraries using `pip`:

bash
pip install PyPDF2 langchain faiss-cpu openai


**Note**: If you encounter issues installing `faiss-cpu`, refer to [FAISS Installation Guide](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) for platform-specific instructions.

### 4. Set Up OpenAI API Key

The script uses OpenAI's embedding models, which require an API key. Set up your API key as an environment variable:

- **On Windows**:

  bash
  set OPENAI_API_KEY=your-api-key-here
  

- **On macOS/Linux**:

  bash
  export OPENAI_API_KEY=your-api-key-here
  

Alternatively, you can set the API key directly in the script, but it's not recommended for security reasons.

---

## Usage

### Preparing the PDF

1. **Place Your PDF**: Ensure your PDF file (`Sample_1.pdf`) is located in the same directory as the script. If your PDF has a different name or is located elsewhere, adjust the `pdf_path` variable accordingly in the script.

2. **Check File Path**: The script uses `Path(__file__).parent` to determine the current directory. Ensure that the script and PDF are in the correct locations.

### Running the Script

Execute the script using Python:

bash
python script_name.py


**Replace `script_name.py` with the actual name of your Python script.**

Upon execution, the script will:

1. Extract text from each page of the specified PDF.
2. Clean and format the extracted text.
3. Split the text into chunks based on the configured `chunk_size` and `chunk_overlap`.
4. Generate embeddings for each text chunk using OpenAI's embedding models.
5. Store the embeddings in a FAISS vector store for efficient similarity searches.

**Example Output**:

The script will print the cleaned text to the console and create a FAISS index file (e.g., `faiss_index`) in the working directory.

---

## Detailed Explanation

### Script Structure

python
from PyPDF2 import PdfReader
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

current_dir = Path(_file_).parent
pdf_path = current_dir / "Sample_1.pdf"

def clean_text(text):
    # Remove excessive newlines
    cleaned = ' '.join(text.split())
    # Add proper sentence spacing
    cleaned = cleaned.replace('. ', '.\n')
    return cleaned

def main():
    if pdf_path is not None:
        pdf_reader = PdfReader(pdf_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Clean the text before printing
        cleaned_text = clean_text(text)
        print(cleaned_text)
        
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=22,
            length_function=len
        )
        
        chunks = text_splitter.split_text(cleaned_text)
        
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

if _name_ == "_main_":
    main()


### Functions

#### `clean_text(text)`

- **Purpose**: Cleans the extracted text by removing excessive whitespace and adding proper sentence spacing.
- **Process**:
  - Joins all words with a single space to eliminate multiple consecutive spaces and newlines.
  - Replaces `. ` with `.\n` to ensure each sentence starts on a new line for better readability.

#### `main()`

- **Purpose**: Orchestrates the process of reading the PDF, cleaning the text, splitting it into chunks, embedding the chunks, and storing them in FAISS.
- **Process**:
  1. **PDF Reading**: Uses `PyPDF2.PdfReader` to read the PDF file and extract text from each page.
  2. **Text Cleaning**: Calls `clean_text` to format the extracted text.
  3. **Text Splitting**: Utilizes `CharacterTextSplitter` from `langchain` to divide the text into chunks based on character count and overlap.
  4. **Embeddings Generation**: Generates vector embeddings for each text chunk using `OpenAIEmbeddings`.
  5. **Vector Store Creation**: Creates a FAISS vector store from the embeddings for efficient similarity searches.

### Text Splitting Configuration

python
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=22,
    length_function=len
)


- **`separator`**: Defines the separator used to split the text. Here, it's set to a newline character.
- **`chunk_size`**: Maximum number of characters per chunk. Set to 1000.
- **`chunk_overlap`**: Number of characters to overlap between consecutive chunks. Set to 22.
- **`length_function`**: Function to compute the length of text, typically `len`.

**Note**: In your initial script, there was a typo (`chunke_overlapping`). It has been corrected to `chunk_overlap`.

### Embedding and Vector Store Creation

python
embeddings = OpenAIEmbeddings()
knowledge_base = FAISS.from_texts(chunks, embeddings)


- **`OpenAIEmbeddings`**: Initializes the embedding model using OpenAI's API.
- **`FAISS.from_texts`**: Creates a FAISS vector store from the list of text chunks and their corresponding embeddings. This enables efficient similarity searches and information retrieval.

---

## Configuration

### Adjusting Chunk Size and Overlap

Depending on the complexity and structure of your PDF, you might need to adjust the `chunk_size` and `chunk_overlap` parameters to optimize the embedding quality and search efficiency.

python
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1500,       # Increased chunk size
    chunk_overlap=50,      # Increased overlap
    length_function=len
)


### Changing the PDF Path

If your PDF is located in a different directory or has a different name, modify the `pdf_path` variable accordingly:

python
pdf_path = Path("/path/to/your/pdf/YourDocument.pdf")


### Selecting a Different Embedding Model

By default, `OpenAIEmbeddings` uses OpenAI's default embedding model. To use a specific model, you can specify it during initialization:

python
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")


Refer to [OpenAI's Embedding Models](https://beta.openai.com/docs/guides/embeddings) for available options.

---

## Error Handling and Troubleshooting

### Common Issues

1. **Missing PDF File**

   **Error Message**:
   
   FileNotFoundError: [Errno 2] No such file or directory: 'Sample_1.pdf'
   

   **Solution**: Ensure that `Sample_1.pdf` is placed in the same directory as the script. If it's located elsewhere, update the `pdf_path` variable with the correct path.

2. **Invalid OpenAI API Key**

   **Error Message**:
   
   openai.error.AuthenticationError: No API key provided.
   

   **Solution**: Set your OpenAI API key as an environment variable (`OPENAI_API_KEY`) or configure it directly in the script. Ensure there are no typos and that the key is active.

3. **FAISS Installation Issues**

   **Error Message**:
   
   ImportError: libfaiss_c.so: cannot open shared object file: No such file or directory
   

   **Solution**: Verify that FAISS is correctly installed. Refer to the [FAISS Installation Guide](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) for platform-specific instructions.

4. **Typographical Errors in Code**

   **Issue**: Typos like `chunke_overlapping` instead of `chunk_overlap` can cause attribute errors.

   **Solution**: Double-check the parameter names in the `CharacterTextSplitter` configuration.

5. **Empty Text Extraction**

   **Issue**: Some PDFs, especially those with scanned images or non-standard text encoding, might result in empty or incomplete text extraction.

   **Solution**: Use OCR tools like `PyTesseract` for scanned PDFs. Ensure that the PDF contains selectable text.

### Enhancing Error Handling

Consider adding try-except blocks to handle exceptions gracefully and provide informative error messages.

python
def main():
    try:
        if pdf_path is not None and pdf_path.exists():
            pdf_reader = PdfReader(pdf_path)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                else:
                    print(f"Warning: No text found on page {page.number}")
            
            if not text:
                raise ValueError("No text extracted from the PDF.")
            
            cleaned_text = clean_text(text)
            print(cleaned_text)
            
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=22,
                length_function=len
            )
            
            chunks = text_splitter.split_text(cleaned_text)
            
            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)
            
            print("Knowledge base created successfully.")
        else:
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


---

## Future Enhancements

1. **Query Interface**: Implement a user-friendly interface (CLI or web-based) to query the FAISS vector store and retrieve relevant information.

2. **Support for Multiple PDFs**: Extend the script to process multiple PDF files and aggregate their embeddings into a single vector store.

3. **Advanced Text Cleaning**: Incorporate more sophisticated text preprocessing techniques, such as removing headers, footers, and handling special characters.

4. **Integration with Databases**: Store metadata alongside embeddings in databases like PostgreSQL or Elasticsearch for enhanced retrieval capabilities.

5. **Automated OCR**: Integrate OCR capabilities to handle scanned PDFs using libraries like `PyTesseract`.

6. **Scheduled Processing**: Set up scheduled tasks to automatically process new PDFs added to a specific directory.

7. **Dockerization**: Containerize the application using Docker for easier deployment and scalability.

---

## Contributing

Contributions are welcome! Follow these steps to contribute:

1. **Fork the Repository**

2. **Create a Feature Branch**

   bash
   git checkout -b feature/YourFeature
   

3. **Commit Your Changes**

   bash
   git commit -m "Add your feature"
   

4. **Push to the Branch**

   bash
   git push origin feature/YourFeature
   

5. **Open a Pull Request**

Ensure your code adheres to the project's coding standards and include relevant tests.

---

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this software as per the license terms.

---

## Acknowledgements

- [PyPDF2](https://github.com/py-pdf/PyPDF2): A library for reading and extracting information from PDF files.
- [LangChain](https://github.com/hwchase17/langchain): A framework for developing applications powered by language models.
- [FAISS](https://github.com/facebookresearch/faiss): A library for efficient similarity search and clustering of dense vectors.
- [OpenAI](https://openai.com/): For providing powerful language models and embedding services.
- [Pathlib](https://docs.python.org/3/library/pathlib.html): For intuitive and powerful filesystem paths handling in Python.

---

## Contact

For any questions or suggestions, feel free to open an issue or contact [your.email@example.com](mailto:your.email@example.com).



---

### Additional Notes

1. *Typographical Corrections in the Script*:
   - *chunk_overlap*: Ensure the parameter is correctly spelled as chunk_overlap instead of chunke_overlapping.
   - *split_text Method*: The CharacterTextSplitter should use split_text or split_texts correctly by passing the text to be split.

   *Corrected Code Snippet*:

   python
   chunks = text_splitter.split_text(cleaned_text)
   

2. *Script Execution Path*:
   - The script uses Path(__file__).parent to determine the current directory. Ensure that when running the script, it's executed from its containing directory to correctly locate the PDF.

3. *Environment Variables Security*:
   - Avoid hardcoding sensitive information like API keys in the script. Use environment variables or secure storage mechanisms.

4. *Extensibility*:
   - The script is designed to be modular. You can extend it by adding functions for querying the FAISS vector store or integrating with other services.

5. *Logging*:
   - Implement logging instead of using print statements for better monitoring and debugging, especially for larger projects.

---

By following this comprehensive README.md, users and contributors will have a clear understanding of the project's purpose, setup, usage, and potential areas for improvement. Ensure that the actual repository includes this README.md and any other necessary documentation or files to facilitate smooth collaboration and usage.
