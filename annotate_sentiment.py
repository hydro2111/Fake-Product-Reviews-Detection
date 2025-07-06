import pandas as pd
from transformers import pipeline
import time

def annotate_sentiment_with_transformer(input_csv="raw_data.csv", output_csv="raw_data_with_sentiment.csv"):
    """
    Reads a CSV file, annotates the sentiment of review texts using a pre-trained
    Hugging Face transformer model, and saves the results to a new CSV.

    Args:
        input_csv (str): Path to the input CSV file (e.g., raw_data.csv).
                         Assumes a 'text_' column contains the review text.
        output_csv (str): Path where the new CSV with sentiment annotations will be saved.
    """
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: Input file '{input_csv}' not found. Please make sure it's in the same directory.")
        return

    if 'text_' not in df.columns:
        print(f"Error: Column 'text_' not found in '{input_csv}'. Please ensure your review text column is named 'text_'.")
        return

    print(f"Loading sentiment analysis model... (This may take a moment)")
    # Using 'cardiffnlp/twitter-roberta-base-sentiment' for 3-class sentiment (positive, negative, neutral)
    # You can explore other models on Hugging Face Hub (e.g., 'distilbert-base-uncased-finetuned-sst-2-english' for 2-class)
    try:
        sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    except Exception as e:
        print(f"Error loading transformer model. Make sure you have 'transformers' and 'torch' installed, and an internet connection.")
        print(f"Detailed error: {e}")
        return

    sentiments = []
    confidence_scores = []
    
    print(f"Annotating sentiments for {len(df)} reviews...")
    start_time = time.time()

    # Process in batches to potentially speed up inference and reduce memory issues for very large datasets
    batch_size = 32 # Adjust based on your system's memory and GPU (if available)
    for i in range(0, len(df), batch_size):
        batch_texts = df['text_'].iloc[i:i+batch_size].tolist()
        # Filter out NaN or non-string values from the batch
        batch_texts = [str(text) for text in batch_texts if pd.notna(text)] 

        if not batch_texts: # Skip if batch is empty after filtering
            sentiments.extend(['N/A'] * (len(df.iloc[i:i+batch_size]) - len(batch_texts)))
            confidence_scores.extend([0.0] * (len(df.iloc[i:i+batch_size]) - len(batch_texts)))
            continue

        try:
            results = sentiment_pipeline(batch_texts)
            for res in results:
                sentiments.append(res['label'])
                confidence_scores.append(res['score'])
        except Exception as e:
            print(f"Error processing batch starting at index {i}: {e}. Filling with 'N/A'.")
            # Fill remaining for the batch with 'N/A' if an error occurs
            sentiments.extend(['N/A'] * len(batch_texts))
            confidence_scores.extend([0.0] * len(batch_texts))
            
        if (i // batch_size + 1) % 10 == 0: # Print progress every 10 batches
            print(f"Processed {i + len(batch_texts)} reviews...")

    # Ensure sentiments and confidence_scores list length matches DataFrame length
    # This might be needed if some texts were filtered out or if an error occurred mid-batch
    if len(sentiments) < len(df):
        num_missing = len(df) - len(sentiments)
        sentiments.extend(['N/A'] * num_missing)
        confidence_scores.extend([0.0] * num_missing)
    elif len(sentiments) > len(df):
        sentiments = sentiments[:len(df)]
        confidence_scores = confidence_scores[:len(df)]


    df['transformer_sentiment_label'] = sentiments
    df['transformer_sentiment_confidence'] = confidence_scores

    df.to_csv(output_csv, index=False)
    end_time = time.time()
    print(f"\nSentiment annotation complete! Results saved to '{output_csv}'.")
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")
    print("Example of annotated data (first 5 rows):")
    print(df[['text_', 'transformer_sentiment_label', 'transformer_sentiment_confidence']].head())

if __name__ == "__main__":
    annotate_sentiment_with_transformer()