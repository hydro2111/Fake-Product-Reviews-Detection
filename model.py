import joblib
from gensim.models import Word2Vec
import numpy as np
from preprocessing import preprocess_text, get_sentiment  # Make sure this exists

# Load models
def load_models():
    try:
        word2vec_model = Word2Vec.load('word2vec_model.model')
        svm_model = joblib.load('SVM_model.pkl')
        rf_model = joblib.load('RF_model.pkl')
        return word2vec_model, svm_model, rf_model
    except FileNotFoundError as e:
        print(f"Model file not found: {e}")
        raise

# Classify using SVM and RF only (no ensemble)
def classify_reviews(reviews, word2vec_model, svm_model, rf_model):
    svm_predictions = []
    rf_predictions = []
    svm_probabilities = []
    rf_probabilities = []
    
    for review in reviews:
        original_review_text = review.get('Original Review Text', review['Review Text'])
        preprocessed_text = preprocess_text(review['Review Text'], apply_spelling_correction=False)
        words = preprocessed_text.split()
        vectors = np.array([word2vec_model.wv[word] for word in words if word in word2vec_model.wv])

        if vectors.size > 0:
            text_vector = np.mean(vectors, axis=0).reshape(1, -1)

            try:
                rating = float(review['Rating'])
            except ValueError:
                rating = 3.0  # default fallback

            review_length = len(words)
            polarity, subjectivity = get_sentiment(original_review_text)

            features = np.hstack([
                np.array([[rating]]),
                np.array([[review_length]]),
                text_vector,
                np.array([[polarity, subjectivity]])
            ])

            svm_pred = svm_model.predict(features)
            rf_pred = rf_model.predict(features)

            try:
                svm_prob = svm_model.predict_proba(features)[0]
                rf_prob = rf_model.predict_proba(features)[0]
            except:
                svm_prob = [0.5, 0.5]
                rf_prob = [0.5, 0.5]

            svm_predictions.append(svm_pred[0])
            rf_predictions.append(rf_pred[0])
            svm_probabilities.append(svm_prob)
            rf_probabilities.append(rf_prob)
        else:
            # fallback for empty or OOV reviews
            svm_predictions.append(0)
            rf_predictions.append(0)
            svm_probabilities.append([0.5, 0.5])
            rf_probabilities.append([0.5, 0.5])

    return {
        'svm_predictions': svm_predictions,
        'rf_predictions': rf_predictions,
        'svm_probabilities': svm_probabilities,
        'rf_probabilities': rf_probabilities
    }

# Optional: test code block
if __name__ == "__main__":
    try:
        word2vec_model, svm_model, rf_model = load_models()

        sample_reviews = [
            {"Review Text": "Very useful and good value for money.", "Rating": 5},
            {"Review Text": "Fake review totally not trustworthy!!!", "Rating": 1}
        ]

        processed_reviews = []
        for r in sample_reviews:
            processed_reviews.append({
                "Original Review Text": r["Review Text"],
                "Review Text": preprocess_text(r["Review Text"], apply_spelling_correction=False),
                "Rating": r["Rating"]
            })

        results = classify_reviews(processed_reviews, word2vec_model, svm_model, rf_model)

        def readable(preds): return ["Real" if p == 0 else "Fake" for p in preds]

        print("SVM:", readable(results['svm_predictions']))
        print("RF :", readable(results['rf_predictions']))

    except Exception as e:
        print(f"Error: {e}")
