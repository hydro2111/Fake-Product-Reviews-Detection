import re
import emoji
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

# Initialize lemmatizer and stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Contractions dictionary
contractions = {
    "ain't": "am not", "aren't": "are not", "can't": "can not", "can't've": "can not have", "cause": "because",
    "could've": "could have", "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not",
    "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
    "he's": "he is", "how's": "how is", "i'm": "i am", "i've": "i have", "isn't": "is not", "it's": "it is",
    "let's": "let us", "should've": "should have", "shouldn't": "should not", "that's": "that is",
    "there's": "there is", "they're": "they are", "they've": "they have", "wasn't": "was not",
    "we're": "we are", "we've": "we have", "weren't": "were not", "what's": "what is", "where's": "where is",
    "who's": "who is", "why's": "why is", "won't": "will not", "would've": "would have", "you'd": "you would",
    "you'll": "you will", "you're": "you are", "you've": "you have"
}

def handle_emojis(text):
    """Converts emojis into text descriptions."""
    return emoji.demojize(text)

def correct_spelling(r):
    """Performs spelling correction on the input text. Can be computationally intensive."""
    return str(TextBlob(r).correct())

def lemmatize_and_stem(r):
    """Performs tokenization, removes stopwords, applies lemmatization, and stemming."""
    words = word_tokenize(r)
    stop_words = set(stopwords.words('english'))
    # Filter out stopwords
    words = [word for word in words if word not in stop_words]
    # Apply lemmatization
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    # Apply stemming to lemmatized words
    stemmed_words = [stemmer.stem(word) for word in lemmatized_words]
    return ' '.join(stemmed_words)

def preprocess_text(r, apply_spelling_correction=False):
    """
    Performs text preprocessing including lowercasing, removing special characters,
    handling numbers, expanding contractions, handling emojis, and applying lemmatization and stemming.
    
    Args:
        r (str): The input text.
        apply_spelling_correction (bool): Whether to apply spelling correction. Set to True with caution
                                         as it can be very slow.
    Returns:
        str: The preprocessed text.
    """
    # Handle emojis before other processing
    if not isinstance(r, str): # Handle potential non-string inputs (e.g., NaN)
        return ""
    r = handle_emojis(r)
    
    # Convert to lowercase
    r = r.lower()

    # Remove URLs
    r = re.sub(r'http\S+|www\S+|https\S+', '', r, flags=re.MULTILINE)
    
    # Remove user mentions and hashtags (if applicable)
    r = re.sub(r'@\w+|#\w+', '', r)
    
    # Remove special characters and numbers (keeping only letters and spaces)
    r = re.sub(r'[^a-zA-Z\s]', '', r)
    
    # Expand contractions
    r = ' '.join([contractions[word] if word in contractions else word for word in r.split()])

    # Correct spelling (optional, and can be very slow)
    if apply_spelling_correction:
        r = correct_spelling(r)

    # Apply lemmatization and stemming, and remove stopwords
    r = lemmatize_and_stem(r)
    
    # Remove extra spaces
    r = re.sub(r'\s+', ' ', r).strip()
    
    return r

def get_sentiment(text):
    """
    Calculates sentiment polarity and subjectivity using TextBlob.
    
    Args:
        text (str): The input text.
    
    Returns:
        tuple: A tuple containing (polarity, subjectivity).
               Polarity is a float in [-1.0, 1.0] where 1.0 is positive.
               Subjectivity is a float in [0.0, 1.0] where 1.0 is very subjective.
    """
    if not isinstance(text, str): # Handle potential non-string inputs
        return 0.0, 0.0
    analysis = TextBlob(text)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity

if __name__ == '__main__':
    # Example usage:
    test_text = "This product is absolutely amazing! I love it so much. 😊 It's the best."
    
    print(f"Original text: {test_text}")
    
    # Preprocess text (without spelling correction for speed)
    processed_text = preprocess_text(test_text, apply_spelling_correction=False)
    print(f"Preprocessed text: {processed_text}")
    
    # Get sentiment
    polarity, subjectivity = get_sentiment(test_text)
    print(f"Sentiment Polarity: {polarity:.2f}")
    print(f"Sentiment Subjectivity: {subjectivity:.2f}")

    test_text_negative = "This product is terrible and utterly useless. What a waste of money. 😠"
    processed_text_negative = preprocess_text(test_text_negative)
    polarity_neg, subjectivity_neg = get_sentiment(test_text_negative)
    print(f"Original text: {test_text_negative}")
    print(f"Preprocessed text: {processed_text_negative}")
    print(f"Sentiment Polarity: {polarity_neg:.2f}")
    print(f"Sentiment Subjectivity: {subjectivity_neg:.2f}")