import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from gensim.models import Word2Vec
import joblib
import os
import re
import emoji
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import warnings

# Suppress NLTK download warnings if data is already present
warnings.filterwarnings("ignore", category=UserWarning, module='nltk')

# --- Re-including necessary preprocessing functions here for self-containment ---
# It's good practice to have core preprocessing functions available directly
# within the training script, or ensure they are imported correctly.
# For simplicity, I'm including them directly.
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

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
    words = [word for word in words if word not in stop_words]
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
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
    if not isinstance(r, str): # Handle potential non-string inputs (e.g., NaN)
        return "" 
        
    r = handle_emojis(r)
    r = r.lower()
    r = re.sub(r'http\S+|www\S+|https\S+', '', r, flags=re.MULTILINE)
    r = re.sub(r'@\w+|#\w+', '', r)
    r = re.sub(r'[^a-zA-Z\s]', '', r)
    r = ' '.join([contractions[word] if word in contractions else word for word in r.split()])
    if apply_spelling_correction:
        r = correct_spelling(r)
    r = lemmatize_and_stem(r)
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

# --- End of preprocessing functions ---

def train_and_save_models(data_path="raw_data.csv", word2vec_model_path="word2vec_model.model", svm_model_path="SVM_model.pkl", rf_model_path="RF_model.pkl"):
    """
    Trains the Word2Vec, SVM, and Random Forest models for fake review detection and saves them.

    Args:
        data_path (str): Path to the dataset CSV file (e.g., raw_data.csv).
        word2vec_model_path (str): Path to save the trained Word2Vec model.
        svm_model_path (str): Path to save the trained SVM model.
        rf_model_path (str): Path to save the trained Random Forest model.
    """
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Dataset '{data_path}' not found. Please ensure the CSV file is in the correct directory.")
        return

    # Check for essential columns
    if 'text_' not in df.columns or 'rating' not in df.columns or 'label' not in df.columns:
        print("Error: Missing essential columns ('text_', 'rating', 'label') in the dataset.")
        return

    # Map labels: 'OR' (Original/Real) to 0, 'CG' (Computer Generated/Fake) to 1
    df['label_numeric'] = df['label'].map({'OR': 0, 'CG': 1})

    # Drop rows where 'text_' is NaN or not a string, or 'label' is missing
    df.dropna(subset=['text_', 'label'], inplace=True)
    df = df[df['text_'].apply(lambda x: isinstance(x, str))]
    df = df[df['label'].isin(['OR', 'CG'])]
    
    print("Preprocessing text data for Word2Vec training...")
    # Preprocess text for Word2Vec (no spelling correction for speed during training)
    # Store original text for TextBlob sentiment if needed
    df['preprocessed_text'] = df['text_'].apply(lambda x: preprocess_text(x, apply_spelling_correction=False))
    
    # Prepare sentences for Word2Vec training
    sentences = [text.split() for text in df['preprocessed_text'] if text.strip()]

    if not sentences:
        print("No valid sentences found after preprocessing. Cannot train Word2Vec model.")
        return

    print("Training Word2Vec model...")
    # Train Word2Vec model
    # vector_size: Dimensionality of the word vectors.
    # window: Maximum distance between the current and predicted word within a sentence.
    # min_count: Ignores all words with total frequency lower than this.
    # workers: Use these many worker threads to train the model.
    word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    word2vec_model.save(word2vec_model_path)
    print(f"Word2Vec model trained and saved to {word2vec_model_path}")

    # Feature Extraction for models
    print("Extracting features for model training (Word2Vec embeddings, rating, length, sentiment)...")
    X = [] # Features
    y = [] # Labels

    for index, row in df.iterrows():
        review_text_original = row['text_']
        preprocessed_text = row['preprocessed_text']
        rating = row['rating']
        label = row['label_numeric']

        # Get Word2Vec vector for the review
        words = preprocessed_text.split()
        vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
        
        if vectors:
            text_vector = np.mean(vectors, axis=0) # Average word vectors
        else:
            # Handle cases where no words in review are in Word2Vec vocabulary
            text_vector = np.zeros(word2vec_model.vector_size) 

        # Get sentiment features (polarity and subjectivity) using TextBlob
        polarity, subjectivity = get_sentiment(review_text_original)

        # Combine all features
        # Ensure 'rating' is a float; handle potential errors
        try:
            rating_float = float(rating)
        except (ValueError, TypeError):
            rating_float = 3.0 # Default if conversion fails

        features = np.hstack([
            np.array([rating_float]),         # Rating as a feature
            np.array([len(words)]),           # Review length as a feature
            text_vector,                      # Word2Vec text vector
            np.array([polarity, subjectivity]) # Sentiment as features
        ])
        
        X.append(features)
        y.append(label)

    X = np.array(X)
    y = np.array(y)
    
    if X.size == 0 or y.size == 0:
        print("No features extracted. Training aborted.")
        return

    # Split data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training SVM model...")
    # Train SVM model
    # Consider adjusting C and kernel for better performance
    svm_model = SVC(kernel='linear', C=1.0, random_state=42) 
    svm_model.fit(X_train, y_train)

    # Save the SVM model
    joblib.dump(svm_model, svm_model_path)
    print(f"SVM model trained and saved to {svm_model_path}")

    print("Training Random Forest model...")
    # Train Random Forest model
    # n_estimators: Number of trees in the forest
    # max_depth: Maximum depth of the tree
    # random_state: For reproducibility
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)

    # Save the Random Forest model
    joblib.dump(rf_model, rf_model_path)
    print(f"Random Forest model trained and saved to {rf_model_path}")

    # Evaluate both models on test set
    from sklearn.metrics import classification_report, accuracy_score
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*50)
    
    # SVM Evaluation
    print("\nSVM Model Performance:")
    print("-" * 30)
    y_pred_svm = svm_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_svm, target_names=['Real', 'Fake']))
    
    # Random Forest Evaluation
    print("\nRandom Forest Model Performance:")
    print("-" * 35)
    y_pred_rf = rf_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_rf, target_names=['Real', 'Fake']))
    
    # Feature importance for Random Forest
    print("\nRandom Forest Feature Importance:")
    print("-" * 35)
    feature_names = ['Rating', 'Review_Length'] + [f'Word2Vec_{i}' for i in range(word2vec_model.vector_size)] + ['Sentiment_Polarity', 'Sentiment_Subjectivity']
    feature_importance = rf_model.feature_importances_
    
    # Create a sorted list of (feature_name, importance) tuples
    feature_importance_pairs = list(zip(feature_names, feature_importance))
    feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 10 Most Important Features:")
    for i, (feature, importance) in enumerate(feature_importance_pairs[:10]):
        print(f"{i+1:2d}. {feature}: {importance:.4f}")

    print("\nTraining complete! All models are ready to be used.")

# In train_model.py, find this block at the very end of the file:
if __name__ == "__main__":
    # Ensure NLTK data is downloaded if not already
    try:
        import nltk
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/wordnet')
    except LookupError: # Correctly catch LookupError for missing resources
        print("Downloading NLTK data (stopwords, punkt, wordnet)...")
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        print("NLTK data download complete.")
    except Exception as e: # Catch any other unexpected errors during NLTK setup
        print(f"An unexpected error occurred during NLTK data check/download: {e}")
        # You might want to exit or handle this error differently based on severity
        import sys
        sys.exit(1) # Exit if NLTK data setup fails in an unexpected way

    # Pass the correct data path to the training function
    # Assuming 'raw_data.csv' is directly in a 'Data' subfolder relative to train_model.py
    train_and_save_models(data_path="Data/raw_data.csv")