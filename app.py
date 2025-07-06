from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from scraper import scrape_reviews
from model import load_models, classify_reviews
from preprocessing import preprocess_text, get_sentiment
import pandas as pd
import os

# Initialize Flask app
# Make sure your 'index.html' is in a 'templates' folder
# and your static files (if any) are in a 'static' folder.
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app) 

# --- Model Loading ---
# Load the models once when the application starts.
try:
    word2vec_model, svm_model, rf_model = load_models()
    print("Models loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load machine learning models. The server will not be able to process requests. Error: {e}")
    word2vec_model, svm_model, rf_model = None, None, None

def get_sentiment_category(polarity):
    """
    Categorizes sentiment polarity into 'Positive', 'Negative', or 'Neutral'.
    A slightly wider neutral band can improve categorization.
    """
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

@app.route('/')
def index():
    """
    Renders the main page of the application.
    Flask will look for 'index.html' in the 'templates' directory.
    """
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    API endpoint to analyze reviews from a given product URL.
    It scrapes reviews, classifies them using both SVM and Random Forest models,
    calculates summary statistics, and returns a comprehensive JSON response.
    """
    # Check if models were loaded successfully on startup
    if not word2vec_model or not svm_model or not rf_model:
        return jsonify({"error": "Models are not loaded. The server is unable to process the request. Please check server logs."}), 500

    data = request.json
    url = data.get('url')
    selected_model = data.get('selectedModel', 'svm')  # Default to SVM
    
    if not url:
        return jsonify({"error": "URL is required."}), 400
    
    # --- 1. Scrape Reviews ---
    try:
        reviews_df = scrape_reviews(url)
        if reviews_df.empty:
            return jsonify({"error": "Could not find any reviews on the page. The product may have no reviews, or the URL is for a different site format."}), 404
    except Exception as e:
        print(f"Error during scraping for URL '{url}': {e}")
        return jsonify({"error": "Failed to scrape the provided URL. It might be invalid, or the website's structure has changed."}), 500

    if "Review Text" not in reviews_df.columns or "Rating" not in reviews_df.columns:
        return jsonify({"error": "The scraped data is in an unexpected format. Cannot process reviews."}), 500
    
    # --- 2. Prepare Data for Classification and Display ---
    reviews_for_classification = []
    detailed_results = []

    for index, row in reviews_df.iterrows():
        original_text = row["Review Text"]
        rating = row["Rating"]
        
        # Data for the ML model
        reviews_for_classification.append({
            "Original Review Text": original_text, # For sentiment analysis
            "Review Text": original_text, # The 'classify_reviews' function handles its own preprocessing
            "Rating": rating
        })
        
        # Data for the frontend display
        polarity, _ = get_sentiment(original_text)
        sentiment_category = get_sentiment_category(polarity)
        
        detailed_results.append({
            "reviewText": original_text,
            "rating": rating,
            "sentiment": sentiment_category,
            "svm_prediction": None,  # This will be filled in the next step
            "rf_prediction": None,   # This will be filled in the next step
            "primary_prediction": None  # This will be the selected model's prediction
        })

    # --- 3. Classify Reviews using both ML Models ---
    results = classify_reviews(reviews_for_classification, word2vec_model, svm_model, rf_model)
    
    # --- 4. Compile Final Results & Calculate Statistics ---
    svm_real_count = 0
    svm_fake_count = 0
    rf_real_count = 0
    rf_fake_count = 0
    primary_real_count = 0
    primary_fake_count = 0
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}

    for i, result in enumerate(detailed_results):
        # SVM predictions
        svm_prediction_label = "Real" if results['svm_predictions'][i] == 0 else "Fake"
        result["svm_prediction"] = svm_prediction_label
        if svm_prediction_label == "Real":
            svm_real_count += 1
        else:
            svm_fake_count += 1
        
        # Random Forest predictions
        rf_prediction_label = "Real" if results['rf_predictions'][i] == 0 else "Fake"
        result["rf_prediction"] = rf_prediction_label
        if rf_prediction_label == "Real":
            rf_real_count += 1
        else:
            rf_fake_count += 1
        
        # Set primary prediction based on selected model
        if selected_model == 'svm':
            result["primary_prediction"] = svm_prediction_label
            if svm_prediction_label == "Real":
                primary_real_count += 1
                # Count sentiment for real reviews to calculate trust score
                sentiment_counts[result["sentiment"]] += 1
            else:
                primary_fake_count += 1
        elif selected_model == 'rf':
            result["primary_prediction"] = rf_prediction_label
            if rf_prediction_label == "Real":
                primary_real_count += 1
                # Count sentiment for real reviews to calculate trust score
                sentiment_counts[result["sentiment"]] += 1
            else:
                primary_fake_count += 1
    
    total_reviews = len(detailed_results)
    
    # Calculate percentages for each model
    svm_real_percentage = (svm_real_count / total_reviews * 100) if total_reviews > 0 else 0
    svm_fake_percentage = (svm_fake_count / total_reviews * 100) if total_reviews > 0 else 0
    
    rf_real_percentage = (rf_real_count / total_reviews * 100) if total_reviews > 0 else 0
    rf_fake_percentage = (rf_fake_count / total_reviews * 100) if total_reviews > 0 else 0
    
    primary_real_percentage = (primary_real_count / total_reviews * 100) if total_reviews > 0 else 0
    primary_fake_percentage = (primary_fake_count / total_reviews * 100) if total_reviews > 0 else 0
    
    # Calculate an overall "Trust Score" based on the sentiment of authentic reviews
    trust_score = 0
    if primary_real_count > 0:
        # A simple score: percentage of positive reviews among all real reviews
        positive_real_reviews = sentiment_counts["Positive"]
        trust_score = (positive_real_reviews / primary_real_count) * 100

    summary_stats = {
        "totalReviews": total_reviews,
        "selectedModel": selected_model,
        "primary": {
            "realCount": primary_real_count,
            "fakeCount": primary_fake_count,
            "realPercentage": round(primary_real_percentage, 1),
            "fakePercentage": round(primary_fake_percentage, 1)
        },
        "svm": {
            "realCount": svm_real_count,
            "fakeCount": svm_fake_count,
            "realPercentage": round(svm_real_percentage, 1),
            "fakePercentage": round(svm_fake_percentage, 1)
        },
        "randomForest": {
            "realCount": rf_real_count,
            "fakeCount": rf_fake_count,
            "realPercentage": round(rf_real_percentage, 1),
            "fakePercentage": round(rf_fake_percentage, 1)
        },
        "sentimentCounts": sentiment_counts,
        "trustScore": round(trust_score)
    }

    # --- 5. Return the full JSON Response ---
    return jsonify({
        "summary": summary_stats,
        "reviews": detailed_results
    })

if __name__ == '__main__':
    # It's good practice to get the port from the environment, with a default
    port = int(os.environ.get("PORT", 5001)) 
    app.run(host='0.0.0.0', port=port, debug=True)
