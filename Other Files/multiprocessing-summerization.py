import os
import math
import PyPDF2
import logging
import threading
import multiprocessing
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
os.environ['NUMEXPR_MAX_THREADS'] = '10'  # Change this to the desired number of threads


# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Function to extract text from a PDF using PyPDF2 (I/O-bound task)
def extract_text_from_pdf(file_path):
    text = ''
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            logging.info(f"Extracting text from PDF: {file_path}")
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        logging.info(f"Successfully extracted text from {len(pdf_reader.pages)} pages.")
    except Exception as e:
        logging.error(f"Failed to extract text from PDF: {e}")
    return text

# Function to split text by characters
def split_text_by_characters(text, chunk_size_chars=1000):
    chunks = [text[i:i + chunk_size_chars] for i in range(0, len(text), chunk_size_chars)]
    logging.info(f"Split text into {len(chunks)} chunks.")
    return chunks

# Function to ensure token safety
def ensure_safe_token_length(chunks, tokenizer, max_chunk_length=512):
    validated_chunks = []
    for chunk in chunks:
        tokens = tokenizer.encode(chunk)
        if len(tokens) > max_chunk_length:
            token_chunks = [tokens[i:i + max_chunk_length] for i in range(0, len(tokens), max_chunk_length)]
            for token_chunk in token_chunks:
                validated_chunks.append(tokenizer.decode(token_chunk, skip_special_tokens=True))
        else:
            validated_chunks.append(chunk)
    logging.info(f"Ensured all chunks are token-safe. Total chunks: {len(validated_chunks)}.")
    return validated_chunks

# Function to summarize a chunk of text (CPU-bound task)
def summarize_chunk(chunk, tokenizer, model):
    try:
        logging.info(f"Summarizing chunk of size {len(chunk)} characters.")
        inputs = tokenizer.encode("summarize: " + chunk, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        logging.info("Successfully summarized the chunk.")
        return summary
    except Exception as e:
        logging.error(f"Failed to summarize chunk: {e}")
        return ""

# Function to summarize a chunk using a global model and tokenizer
def summarize_chunk_with_model(args):
    chunk, tokenizer, model = args
    return summarize_chunk(chunk, tokenizer, model)

# Summarize PDF document (Combining both multithreading and multiprocessing)
def summarize_pdf(pdf_file_path):
    logging.info(f"Starting PDF summarization for file: {pdf_file_path}")
    
    document_text = extract_text_from_pdf(pdf_file_path)
    if not document_text:
        logging.error("No text extracted from PDF. Exiting process.")
        return ""

    char_chunks = split_text_by_characters(document_text, chunk_size_chars=1000)

    logging.info("Loading tokenizer and model for summarization.")
    tokenizer = AutoTokenizer.from_pretrained("Falconsai/text_summarization")
    model = AutoModelForSeq2SeqLM.from_pretrained("Falconsai/text_summarization")

    validated_text_chunks = ensure_safe_token_length(char_chunks, tokenizer, max_chunk_length=512)

    logging.info("Starting multiprocessing for summarization.")
    with ProcessPoolExecutor() as executor:
        # Pass a tuple of (chunk, tokenizer, model) to the new function
        summarized_chunks = list(executor.map(summarize_chunk_with_model, [(chunk, tokenizer, model) for chunk in validated_text_chunks]))

    logging.info("PDF summarization completed successfully.")
    return '\n\n'.join(summarized_chunks)

# Example usage
if __name__ == "__main__":
    pdf_file_paths = ['C:/Users/mango/Documents/GitHub/PDF-summerizer/Data/pdf2.pdf']

    logging.info("Starting multithreading for PDF extraction.")
    with ThreadPoolExecutor() as executor:
        summaries = list(executor.map(summarize_pdf, pdf_file_paths))

    for idx, summary in enumerate(summaries):
        print(f"\nSummary of PDF {idx + 1}:\n{summary}\n")
