import os
import pdfplumber
import streamlit as st
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import nltk
from keybert import KeyBERT
from concurrent.futures import ThreadPoolExecutor
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from bson.objectid import ObjectId  # Import ObjectId

# MongoDB connection string
uri = "YOUR_API_KEY"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Select the database and collection
db = client['my_database']  # Replace 'my_database' with your database name
results_collection = db['results']  # Replace 'results' with your results collection name

# Download 'punkt' for tokenization
nltk.download('punkt')
nltk.download('punkt_tab')

# Configuration
LANGUAGE = "english"
SENTENCES_COUNT = 5  # Number of sentences to extract for the summary

# Function to extract text from PDF using pdfplumber
def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() if page.extract_text() else ''
    except Exception as e:
        text = f"Error extracting text: {e}"
    return text

# Function to summarize text using Sumy (LSA method)
def summarize_text(text, sentence_count=SENTENCES_COUNT):
    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    summary = summarizer(parser.document, sentence_count)
    return ' '.join([str(sentence) for sentence in summary])

# Function to extract keywords using KeyBERT
def extract_keywords(text):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(text)
    return ', '.join([keyword for keyword, _ in keywords])

# Function to process a single PDF (summarize and extract keywords)
def process_pdf(file):
    text = extract_text_from_pdf(file)
    summary = summarize_text(text)
    keywords = extract_keywords(text)
    return summary, keywords, file.name, file.size  # Return additional metadata

# Function to fetch stored results from MongoDB
def fetch_results():
    results = list(results_collection.find())
    return results

# Streamlit interface
def main():
    st.title("PDF Summarizer with Keywords Extraction")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload PDF", "View Stored Data"])

    if page == "Upload PDF":
        # Upload multiple PDFs
        uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

        if uploaded_files:
            # Confirm before starting summarization
            if st.button("Summarize PDFs"):
                st.write("Summarizing...")

                # Create a progress bar
                progress_bar = st.progress(0)
                total_files = len(uploaded_files)

                # Initialize ThreadPoolExecutor for concurrency
                results = []
                with ThreadPoolExecutor() as executor:
                    for i, file in enumerate(uploaded_files):
                        # Submit task to the executor
                        future = executor.submit(process_pdf, file)
                        results.append(future)

                        # Update the progress bar
                        progress_bar.progress((i + 1) / total_files)

                # Collect the summaries and keywords
                for i, future in enumerate(results):
                    summary, keywords, doc_name, doc_size = future.result()  # Unpack additional metadata
                    doc_size_mb = doc_size / (1024 * 1024)  # Convert size from bytes to MB
                    st.subheader(f"Summary of {uploaded_files[i].name}:")
                    st.write(summary)
                    st.subheader(f"Keywords of {uploaded_files[i].name}:")
                    st.write(keywords)

                    # Generate a unique ID
                    unique_id = ObjectId()

                    # Save results and metadata in the 'results' collection
                    result_document = {
                        '_id': unique_id,  # Assign the unique ID
                        'summary': summary,
                        'keywords': keywords,
                        'document_name': doc_name,
                        'path': uploaded_files[i].name,  # Adjust if you need the full path
                        'size_mb': round(doc_size_mb, 2)  # Save size in MB, rounded to 2 decimal places
                    }
                    results_collection.insert_one(result_document)
                    st.write('Results and metadata saved in results collection.')

    elif page == "View Stored Data":
        st.write("Stored PDF Metadata")
        
        # Fetch and display stored results from MongoDB
        results = fetch_results()

        if results:
            for result in results:
                # Debug: Print the result document to check its structure
                # st.write(result)  # Check the structure of the fetched document
                
                # Display the metadata
                try:
                    st.write(f"**Document Name:** {result['document_name']}")
                    st.write(f"**Path:** {result['path']}")
                    st.write(f"**Size (MB):** {result['size_mb']}")
                    # st.write(f"**Result ID:** {result['_id']}")  # Display the result ID
                    
                    # Button to view summary and keywords
                    if st.button(f"View Summary and Keywords for {result['document_name']}"):
                        st.subheader("Summary:")
                        st.write(result['summary'])
                        st.subheader("Keywords:")
                        st.write(result['keywords'])
                except KeyError as e:
                    st.error(f"Missing key in document: {e}")
                st.write("---")
        else:
            st.write("No summaries found in the results collection.")

if __name__ == "__main__":
    main()
