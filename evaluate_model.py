import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from model import load_models, classify_reviews
from preprocessing import preprocess_text
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model_performance(data_path="raw_data.csv"):
    """
    Evaluates the fake review detection models' performance using a labeled dataset.
    Generates formatted tables and confusion matrix analysis for SVM and Random Forest models.

    Args:
        data_path (str): The path to the raw_data.csv file.
    """
    # Load the dataset
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_path}. Please ensure raw_data.csv is in the correct directory.")
        return

    # Map string labels ('OR', 'CG') to numerical values (0, 1)
    # Assuming 'OR' (Original/Real) maps to 0, and 'CG' (Computer Generated/Fake) maps to 1
    df['numeric_label'] = df['label'].map({'OR': 0, 'CG': 1})

    # Prepare data for classification
    reviews_for_prediction = []
    actual_labels = []

    for index, row in df.iterrows():
        reviews_for_prediction.append({
            "Original Review Text": row['text_'], # Original for sentiment (if used in model)
            "Review Text": preprocess_text(row['text_'], apply_spelling_correction=False), # Preprocessed for Word2Vec
            "Rating": row['rating']
        })
        actual_labels.append(row['numeric_label'])

    # Load the trained models
    try:
        word2vec_model, svm_model, rf_model = load_models()
    except Exception as e:
        print(f"Failed to load models: {e}. Ensure models are trained and saved correctly.")
        return

    # Classify the reviews using both models
    print("Classifying reviews with SVM and Random Forest models... This might take a moment.")
    results = classify_reviews(reviews_for_prediction, word2vec_model, svm_model, rf_model)
    
    print("Classification complete.")

    # Convert lists to numpy arrays for scikit-learn functions
    actual_labels = np.array(actual_labels)
    svm_predictions = np.array(results['svm_predictions'])
    rf_predictions = np.array(results['rf_predictions'])

    # Evaluate both models
    models = {
        'SVM': svm_predictions,
        'Random Forest': rf_predictions
    }

    print("\n" + "="*80)
    print("FAKE REVIEW DETECTION MODELS EVALUATION RESULTS")
    print("="*80)

    for model_name, predictions in models.items():
        print(f"\n{model_name.upper()} MODEL EVALUATION")
        print("=" * 60)
        
        # Calculate overall metrics
        overall_accuracy = accuracy_score(actual_labels, predictions)
        overall_precision = precision_score(actual_labels, predictions, average='weighted')
        overall_recall = recall_score(actual_labels, predictions, average='weighted')
        overall_f1 = f1_score(actual_labels, predictions, average='weighted')

        # Calculate class-specific metrics
        fake_precision = precision_score(actual_labels, predictions, pos_label=1)
        fake_recall = recall_score(actual_labels, predictions, pos_label=1)
        fake_f1 = f1_score(actual_labels, predictions, pos_label=1)
        
        real_precision = precision_score(actual_labels, predictions, pos_label=0)
        real_recall = recall_score(actual_labels, predictions, pos_label=0)
        real_f1 = f1_score(actual_labels, predictions, pos_label=0)

        # Generate confusion matrix
        cm = confusion_matrix(actual_labels, predictions)
        tn, fp, fn, tp = cm.ravel()

        # Display results in formatted tables
        print(f"\nTable: {model_name} Overall Evaluation Matrix")
        print("-" * 50)
        print(f"{'Metric':<15} {'Value':<10}")
        print("-" * 50)
        print(f"{'Accuracy':<15} {overall_accuracy:.4f}")
        print(f"{'Precision':<15} {overall_precision:.4f}")
        print(f"{'Recall':<15} {overall_recall:.4f}")
        print(f"{'F1-score':<15} {overall_f1:.4f}")
        print("-" * 50)

        print(f"\nTable: {model_name} Class-Specific Evaluation")
        print("-" * 60)
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 60)
        print(f"{'Fake':<15} {fake_precision:<10.4f} {fake_recall:<10.4f} {fake_f1:<10.4f}")
        print(f"{'Real':<15} {real_precision:<10.4f} {real_recall:<10.4f} {real_f1:<10.4f}")
        print("-" * 60)

        # Confusion Matrix Analysis
        print(f"\nTable: {model_name} Confusion Matrix Analysis")
        print("-" * 60)
        print(f"{'Metric':<20} {'Predicted: Fake':<15} {'Predicted: Real':<15}")
        print("-" * 60)
        print(f"{'Actual: Real':<20} {fp:<15} {tn:<15}")
        print(f"{'Actual: Fake':<20} {tp:<15} {fn:<15}")
        print("-" * 60)

        print(f"\n{model_name} PERFORMANCE ANALYSIS:")
        print(f"Overall accuracy: {overall_accuracy:.4f} ({'Strong' if overall_accuracy > 0.8 else 'Moderate' if overall_accuracy > 0.6 else 'Weak'} performance)")
        print(f"F1-score: {overall_f1:.4f} ({'Excellent' if overall_f1 > 0.85 else 'Good' if overall_f1 > 0.75 else 'Acceptable'} balance)")
        print(f"Fake review detection: {fake_recall:.4f} recall ({fake_recall*100:.1f}% of fake reviews caught)")
        print(f"Real review precision: {real_precision:.4f} ({real_precision*100:.1f}% of flagged real reviews are actually genuine)")

        # Store metrics for plotting
        if model_name == 'SVM':
            svm_metrics = {
                'accuracy': overall_accuracy,
                'f1': overall_f1,
                'cm': cm
            }
        else:
            rf_metrics = {
                'accuracy': overall_accuracy,
                'f1': overall_f1,
                'cm': cm
            }

    # Model Comparison Summary
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    
    comparison_data = []
    for model_name, predictions in models.items():
        accuracy = accuracy_score(actual_labels, predictions)
        f1 = f1_score(actual_labels, predictions, average='weighted')
        fake_recall = recall_score(actual_labels, predictions, pos_label=1)
        real_precision = precision_score(actual_labels, predictions, pos_label=0)
        
        comparison_data.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'F1-Score': f1,
            'Fake_Recall': fake_recall,
            'Real_Precision': real_precision
        })
    
    print(f"\n{'Model':<15} {'Accuracy':<10} {'F1-Score':<10} {'Fake_Recall':<12} {'Real_Precision':<15}")
    print("-" * 80)
    for data in comparison_data:
        print(f"{data['Model']:<15} {data['Accuracy']:<10.4f} {data['F1-Score']:<10.4f} {data['Fake_Recall']:<12.4f} {data['Real_Precision']:<15.4f}")
    print("-" * 80)

    # Find best performing model
    best_accuracy_model = max(comparison_data, key=lambda x: x['Accuracy'])
    best_f1_model = max(comparison_data, key=lambda x: x['F1-Score'])
    
    print(f"\nBEST PERFORMING MODELS:")
    print(f"Highest Accuracy: {best_accuracy_model['Model']} ({best_accuracy_model['Accuracy']:.4f})")
    print(f"Highest F1-Score: {best_f1_model['Model']} ({best_f1_model['F1-Score']:.4f})")

    # Additional insights
    print(f"\nKEY INSIGHTS:")
    print("="*80)
    total_samples = len(actual_labels)
    fake_samples = np.sum(actual_labels == 1)
    real_samples = np.sum(actual_labels == 0)
    
    print(f"Total samples evaluated: {total_samples}")
    print(f"Real reviews: {real_samples} ({real_samples/total_samples*100:.1f}%)")
    print(f"Fake reviews: {fake_samples} ({fake_samples/total_samples*100:.1f}%)")
    
    # Show model agreement/disagreement
    svm_correct = np.sum(svm_predictions == actual_labels)
    rf_correct = np.sum(rf_predictions == actual_labels)
    models_agree = np.sum(svm_predictions == rf_predictions)
    
    print(f"SVM correctly classified: {svm_correct} ({svm_correct/total_samples*100:.1f}%)")
    print(f"Random Forest correctly classified: {rf_correct} ({rf_correct/total_samples*100:.1f}%)")
    print(f"Models agreed on: {models_agree} predictions ({models_agree/total_samples*100:.1f}%)")
    print(f"Models disagreed on: {total_samples - models_agree} predictions ({(total_samples - models_agree)/total_samples*100:.1f}%)")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

    # Plotting section
    model_names = ['SVM', 'Random Forest']
    accuracies = [svm_metrics['accuracy'], rf_metrics['accuracy']]
    f1_scores = [svm_metrics['f1'], rf_metrics['f1']]
    # Bar plot for accuracy and F1-score
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(model_names, accuracies, color=['skyblue', 'orange'])
    plt.ylim(0, 1)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.subplot(1, 2, 2)
    plt.bar(model_names, f1_scores, color=['skyblue', 'orange'])
    plt.ylim(0, 1)
    plt.title('Model F1-Score')
    plt.ylabel('F1-Score')
    plt.tight_layout()
    plt.show()
    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cm, title in zip(axes, [svm_metrics['cm'], rf_metrics['cm']], model_names):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{title} Confusion Matrix')
    plt.tight_layout()
    plt.show()


def save_results_to_csv(actual_labels, svm_predictions, rf_predictions, output_path="evaluation_results.csv"):
    """
    Save detailed evaluation results to a CSV file for further analysis.
    
    Args:
        actual_labels (array): True labels
        svm_predictions (array): SVM predicted labels
        rf_predictions (array): Random Forest predicted labels
        output_path (str): Path to save the results CSV
    """
    models = {
        'SVM': svm_predictions,
        'Random_Forest': rf_predictions
    }
    
    results_data = []
    
    for model_name, predictions in models.items():
        # Calculate all metrics
        overall_accuracy = accuracy_score(actual_labels, predictions)
        overall_precision = precision_score(actual_labels, predictions, average='weighted')
        overall_recall = recall_score(actual_labels, predictions, average='weighted')
        overall_f1 = f1_score(actual_labels, predictions, average='weighted')
        
        fake_precision = precision_score(actual_labels, predictions, pos_label=1)
        fake_recall = recall_score(actual_labels, predictions, pos_label=1)
        fake_f1 = f1_score(actual_labels, predictions, pos_label=1)
        
        real_precision = precision_score(actual_labels, predictions, pos_label=0)
        real_recall = recall_score(actual_labels, predictions, pos_label=0)
        real_f1 = f1_score(actual_labels, predictions, pos_label=0)
        
        cm = confusion_matrix(actual_labels, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        # Add results for this model
        model_results = [
            {'Model': model_name, 'Metric_Category': 'Overall', 'Metric_Name': 'Accuracy', 'Value': overall_accuracy},
            {'Model': model_name, 'Metric_Category': 'Overall', 'Metric_Name': 'Precision', 'Value': overall_precision},
            {'Model': model_name, 'Metric_Category': 'Overall', 'Metric_Name': 'Recall', 'Value': overall_recall},
            {'Model': model_name, 'Metric_Category': 'Overall', 'Metric_Name': 'F1_Score', 'Value': overall_f1},
            {'Model': model_name, 'Metric_Category': 'Fake_Class', 'Metric_Name': 'Precision', 'Value': fake_precision},
            {'Model': model_name, 'Metric_Category': 'Fake_Class', 'Metric_Name': 'Recall', 'Value': fake_recall},
            {'Model': model_name, 'Metric_Category': 'Fake_Class', 'Metric_Name': 'F1_Score', 'Value': fake_f1},
            {'Model': model_name, 'Metric_Category': 'Real_Class', 'Metric_Name': 'Precision', 'Value': real_precision},
            {'Model': model_name, 'Metric_Category': 'Real_Class', 'Metric_Name': 'Recall', 'Value': real_recall},
            {'Model': model_name, 'Metric_Category': 'Real_Class', 'Metric_Name': 'F1_Score', 'Value': real_f1},
            {'Model': model_name, 'Metric_Category': 'Confusion_Matrix', 'Metric_Name': 'True_Positives', 'Value': tp},
            {'Model': model_name, 'Metric_Category': 'Confusion_Matrix', 'Metric_Name': 'True_Negatives', 'Value': tn},
            {'Model': model_name, 'Metric_Category': 'Confusion_Matrix', 'Metric_Name': 'False_Positives', 'Value': fp},
            {'Model': model_name, 'Metric_Category': 'Confusion_Matrix', 'Metric_Name': 'False_Negatives', 'Value': fn}
        ]
        
        results_data.extend(model_results)
    
    # Create results dataframe and save
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(output_path, index=False)
    print(f"Detailed evaluation results saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    evaluate_model_performance("Data/raw_data.csv")