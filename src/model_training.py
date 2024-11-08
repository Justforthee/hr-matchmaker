from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transformer_model import ResumeMatchingTransformer

import pickle
import nltk
import numpy as np
import os
import time

# Define required NLTK resources
required_resources = {
    'tokenizers': ['punkt'],
    'corpora': ['stopwords', 'wordnet']
}

# Check and download required NLTK resources if not already present
for resource_type, resources in required_resources.items():
    for resource in resources:
        try:
            # Try to find the resource first
            nltk.data.find(f'{resource_type}/{resource}')
        except LookupError:
            # Download only if resource is not found
            nltk.download(resource)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    # Convert to lowercase and tokenize
    tokens = nltk.word_tokenize(text.lower())
    
    # remove punctuation and special characters
    tokens = [token for token in tokens if token.isalnum()]
    
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
        # Add bigram phrases
    bigrams = list(nltk.bigrams(tokens))
    bigram_phrases = ['_'.join(bigram) for bigram in bigrams]
    # tokens.extend(bigram_phrases)

    # return ' '.join(tokens)
    return ' '.join(tokens + bigram_phrases)

def train_model(job_posts, applicants):
    # Preprocess text data
    job_posts['processed_requirements'] = job_posts['requirements'].apply(preprocess_text)
    applicants['processed_resume'] = applicants['resume'].apply(preprocess_text)
    
    # Configure and train vectorizer with custom parameters
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),      # Include up to trigrams
        max_features=15000,      # Increased vocabulary size
        min_df=2,               # Minimum document frequency
        max_df=0.85,           # Maximum document frequency
        sublinear_tf=True,     # Apply sublinear scaling
        use_idf=True,
        smooth_idf=True
    )
    
    # Vectorize job requirements and applicant resumes
    job_vectors = vectorizer.fit_transform(job_posts['processed_requirements'])
    applicant_vectors = vectorizer.transform(applicants['processed_resume'])
    
    # Calculate similarity scores
    similarity_scores = cosine_similarity(applicant_vectors, job_vectors)
    
    # Save the vectorizer and similarity scores for later use
    with open('models/screening_model.pkl', 'wb') as f:
        pickle.dump((vectorizer, similarity_scores), f) 

    ## Is that necessary?
    return vectorizer, similarity_scores
    
    # Replace the existing save operation with:
    # version = save_model(vectorizer, similarity_scores)
    # return vectorizer, similarity_scores, version

def train_transformer_model(job_posts, applicants):
    # Initialize transformer model
    transformer = ResumeMatchingTransformer()
    
    # Train model and get similarity scores
    similarity_scores = transformer.train_model(
        job_posts['requirements'], 
        applicants['resume']
    )
    
    return transformer, similarity_scores

def evaluate_model(similarity_scores, test_data):
    """
    Evaluate model performance using various metrics
    """
    metrics = {}
    
    # Precision at k
    k = min(5, len(test_data))
    top_k_accuracy = np.mean([1.0 if test_data['actual_match'][i] in 
                            np.argsort(similarity_scores[i])[-k:] 
                            else 0.0 for i in range(len(test_data))])
    metrics['precision_at_k'] = top_k_accuracy
    
    # Mean Reciprocal Rank
    mrr = np.mean([1.0 / (np.where(np.argsort(similarity_scores[i])[::-1] == 
                  test_data['actual_match'][i])[0][0] + 1) 
                  for i in range(len(test_data))])
    metrics['mrr'] = mrr
    
    return metrics

def save_model(model, similarity_scores, version=None):
    """
    Save model with versioning support
    """
    if version is None:
        version = int(time.time())
    
    model_dir = 'models/versions'
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = f'{model_dir}/model_v{version}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump((model, similarity_scores), f)
    
    # Update symlink to latest version
    latest_path = 'models/screening_model.pkl'
    if os.path.exists(latest_path):
        os.remove(latest_path)
    os.symlink(model_path, latest_path)
    
    return version