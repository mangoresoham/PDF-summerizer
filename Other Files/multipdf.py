import os
import math
import PyPDF2
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Function to extract text from a PDF using PyPDF2
def extract_text_from_pdf(file_path):
    text = ''
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            logging.info(f"Extracting text from PDF: {file_path}")
            # Loop through each page and extract the text
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        logging.info(f"Successfully extracted text from {len(pdf_reader.pages)} pages.")
    except Exception as e:
        logging.error(f"Failed to extract text from PDF: {e}")
    return text

# Function to split the text into smaller chunks by a character limit
def split_text_by_characters(text, chunk_size_chars=1000):
    chunks = [text[i:i + chunk_size_chars] for i in range(0, len(text), chunk_size_chars)]
    logging.info(f"Split text into {len(chunks)} chunks, each up to {chunk_size_chars} characters.")
    return chunks

# Function to ensure each text chunk fits within a safe token limit for the model
def ensure_safe_token_length(chunks, tokenizer, max_chunk_length=512):
    validated_chunks = []
    
    for chunk in chunks:
        tokens = tokenizer.encode(chunk)
        if len(tokens) > max_chunk_length:
            logging.info(f"Chunk exceeds max token length of {max_chunk_length}. Splitting further.")
            token_chunks = [tokens[i:i + max_chunk_length] for i in range(0, len(tokens), max_chunk_length)]
            for token_chunk in token_chunks:
                validated_chunks.append(tokenizer.decode(token_chunk, skip_special_tokens=True))
        else:
            validated_chunks.append(chunk)
    
    logging.info(f"Ensured all chunks are token-safe. Total chunks after validation: {len(validated_chunks)}")
    return validated_chunks

# Function to summarize a single chunk of text
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

# Main function to summarize a PDF
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
    summarized_chunks = [summarize_chunk(chunk, tokenizer, model) for chunk in validated_text_chunks]

    logging.info("PDF summarization completed successfully.")
    return '\n\n'.join(summarized_chunks)

# Example usage
if __name__ == "__main__":
    # List of PDF file paths to summarize
    pdf_file_paths = [
        'C:/Users/mango/Documents/GitHub/PDF-summerizer/Data/pdf1.pdf',
        'C:/Users/mango/Documents/GitHub/PDF-summerizer/Data/pdf2.pdf',
        'C:/Users/mango/Documents/GitHub/PDF-summerizer/Data/pdf3.pdf',
        # Add more PDF file paths as needed
    ]
    
    # Store summaries in a dictionary
    summaries = {}

    logging.info("Starting the summarization process.")
    
    # Iterate through each PDF file and summarize
    for pdf_file in pdf_file_paths:
        summary = summarize_pdf(pdf_file)
        summaries[pdf_file] = summary

    # Print out all summaries
    for pdf_file, summary in summaries.items():
        print(f"\nSummary of {os.path.basename(pdf_file)}:\n{summary}\n")
