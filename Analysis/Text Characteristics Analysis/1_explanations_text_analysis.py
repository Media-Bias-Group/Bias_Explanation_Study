import numpy as np
import pandas as pd
from collections import Counter
from transformers import AutoTokenizer, pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import requests

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import json

# Initialize the tokenizer and model
model_path = "MMADS/MoralFoundationsClassifier"
model = RobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained(model_path)

# Ensure you have the necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# 1. Complexity / Entropy Calculation
def calculate_entropy_perplexity(text):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer.tokenize(text)
    counts = Counter(tokens)
    total = sum(counts.values())
    probs = np.array([count / total for count in counts.values()])
    entropy = -np.sum(probs * np.log2(probs))
    perplexity = 2 ** entropy
    return entropy, perplexity

# 2. Sentiment Analysis
def analyze_sentiment(text):
    #sentiment_pipeline = pipeline("sentiment-analysis", model="rmayormartins/sentiment-analysis-committee")
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    sentiment = sentiment_pipeline(text)
    return sentiment[0]


# 3. Moral Foundation Analysis
# Load label names
label_names = [
    "care_virtue", "care_vice", "fairness_virtue", "fairness_vice",
    "loyalty_virtue", "loyalty_vice", "authority_virtue", "authority_vice",
    "sanctity_virtue", "sanctity_vice"
]

# Your function to make predictions
def analyze_moral_foundation(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Perform the inference (no gradients needed)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the prediction (logits)
    logits = outputs.logits
    softmax_scores = torch.softmax(logits, dim=-1).squeeze(0)  # Softmax on logits to get probabilities
    
    # Get confidence scores for all labels
    confidence_scores = softmax_scores.tolist()  # Convert to list for easy manipulation
    
    # Create a list of labels with their confidence scores
    label_confidence = {label_names[i]: confidence_scores[i] for i in range(len(label_names))}
    
    return label_confidence



# API request code for Hugging Face API inference (if needed)
def analyze_moral_foundation_api(text):
    # Replace with your Hugging Face token
    headers = {"Authorization": "Bearer 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"}
    data = {"inputs": text}
    
    # Make the request to Hugging Face Inference API
    response = requests.post(f"https://api-inference.huggingface.co/models/{model_path}", headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}

# 4. Flesch-Kincaid Readability Score
def flesch_kincaid_readability(text):
    sentences = nltk.sent_tokenize(text)
    words = word_tokenize(text)
    num_words = len(words)
    num_sentences = len(sentences)
    num_syllables = sum([syllable_count(word) for word in words])
    
    # Calculate the Flesch-Kincaid readability score
    fk_score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
    return fk_score

# Helper function for counting syllables
def syllable_count(word):
    vowels = "aeiouy"
    word = word.lower()
    count = 0
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count = 1
    return count

# Main analysis function to process the CSV and generate results
def analyze_texts(input_file, output_file):
    # Load the input CSV with texts
    texts_df = pd.read_csv(input_file)
    
    results = []
    
    for index, row in texts_df.iterrows():
        text = row['Text']  # Assuming the column containing texts is labeled "Text"
        
        # Perform analysis
        entropy, perplexity = calculate_entropy_perplexity(text)
        sentiment = analyze_sentiment(text)
        moral_foundations = analyze_moral_foundation(text)
        fk_score = flesch_kincaid_readability(text)
        
        # Collect the results in a dictionary
        result = {
            'Text': text,
            'Entropy': entropy,
            'Perplexity': perplexity,
            'Sentiment Label': sentiment['label'],
            'Sentiment Score': sentiment['score'],
            'Flesch-Kincaid Readability Score': fk_score,
        }
        
        # Add moral foundation results
        for label, confidence in moral_foundations.items():
            result[label] = confidence
        
        # Append to the result list
        results.append(result)
    
    # Convert the results into a DataFrame
    results_df = pd.DataFrame(results)
    
    # Save the results to a new CSV
    results_df.to_csv(output_file, index=False)
    print(f"Analysis complete! Results saved to {output_file}")

# Example usage: Specify the input and output file paths
input_file = 'texts.csv'  # Replace with the path to your input file
output_file = 'text_analysis_results_human.csv'  # Replace with your desired output file path

# Run the analysis
analyze_texts(input_file, output_file)