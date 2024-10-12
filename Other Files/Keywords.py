import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the SpaCy model
# spacy.cli.download("en_core_web_md")
nlp = spacy.load('en_core_web_md')

def extract_keywords(text, num_keywords=5):
    # Process the text with SpaCy
    doc = nlp(text)

    # Extract noun phrases and filter out stop words
    phrases = [chunk.text for chunk in doc.noun_chunks if chunk.root.text.lower() not in nlp.Defaults.stop_words]
    
    # Use TF-IDF to rank the phrases
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(phrases)
    
    # Get the feature names and their corresponding scores
    scores = tfidf_matrix.sum(axis=0).A1
    keywords = vectorizer.get_feature_names_out()
    
    # Create a dictionary of keywords and their scores
    keyword_scores = {keywords[i]: scores[i] for i in range(len(keywords))}
    
    # Sort keywords by score and get the top keywords
    sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)

    return [keyword for keyword, score in sorted_keywords[:num_keywords]]

# Example usage
text = """
Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.
"""

keywords = extract_keywords(text, num_keywords=20)
print("Extracted Keywords:", keywords)
