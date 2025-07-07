# ReviewGuard: An AI-Based System for Detecting Fake Product Reviews Using Machine Learning

A machine learning-powered web application that detects fake reviews and analyzes sentiment using AI models.

## Features

- **Dual-Model Analysis**: Choose between SVM and Random Forest
- **Real-time Web Scraping**: Automatically extracts reviews from product URLs
- **Sentiment Analysis**: Analyzes the emotional tone of reviews
- **Visual Analytics**: Interactive charts and detailed statistics
- **Trust Score**: Overall product trustworthiness based on authentic reviews

## How to Use

1. **Start the Application**:
   ```bash
   python app.py
   ```

2. **Access the Web Interface**:
   - Open your browser and go to `http://localhost:5001`
   - Paste a product URL (supports major e-commerce sites)
   - Select your preferred model from the dropdown
   - Click "Analyze" to process the reviews

3. **View Results**:
   - Summary statistics show the primary model's predictions
   - Model section displays results from the models
   - Individual review cards show detailed analysis
   - Interactive charts visualize the data

## Technical Details

### Models Used
- **SVM (Support Vector Machine)**: Linear classifier with Word2Vec embeddings
- **Random Forest**: Decision trees with multiple features

### Features Analyzed
- Review text content (Word2Vec embeddings)
- Star ratings
- Review length and structure
- Sentiment polarity (positive/negative/neutral)

### Dependencies
- Flask (web framework)
- scikit-learn (machine learning)
- pandas (data processing)
- gensim (Word2Vec)
- Chart.js (visualization)

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app.py
   ```

## File Structure

- `app.py` - Main Flask application
- `model.py` - Machine learning model
- `preprocessing.py` - Text preprocessing utilities
- `scraper.py` - Web scraping functionality
- `templates/` - HTML templates
- `static/` - CSS, JavaScript, and static assets
- `Data/` - Training data and model files
