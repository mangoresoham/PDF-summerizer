import spacy 

# Load the Spacy model and create a new document 
nlp = spacy.load("en_core_web_md") 
doc = nlp("This is a sample text for keyword extraction.") 

# Use the noun_chunks property of the document to identify the noun phrases in the text 
noun_phrases = [chunk.text for chunk in doc.noun_chunks] 

# Use term frequency-inverse document frequency (TF-IDF) analysis to rank the noun phrases 
from sklearn.feature_extraction.text import TfidfVectorizer 
vectorizer = TfidfVectorizer() 
tfidf = vectorizer.fit_transform([doc.text]) 

# Get the top 3 most important noun phrases 
top_phrases = sorted(vectorizer.vocabulary_, key=lambda x: tfidf[0, vectorizer.vocabulary_[x]], reverse=True) 

# Print the top 3 keywords 
print(top_phrases)