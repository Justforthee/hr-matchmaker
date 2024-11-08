import pickle
import numpy as np
from transformer_model import ResumeMatchingTransformer

def screen_applicants(applicants, job_posts, batch_size=100):
    with open('models/screening_model.pkl', 'rb') as f:
        model, similarity_scores = pickle.load(f)
    
    matched_applicants = []
    threshold = 0.4 if isinstance(model, ResumeMatchingTransformer) else 0.5
    
    # Process in batches for better memory management
    for i in range(0, len(applicants), batch_size):
        batch = applicants.iloc[i:i+batch_size]
        batch_scores = similarity_scores[i:i+batch_size]
        
        # Vectorized operations for better performance
        job_indices = np.argmax(batch_scores, axis=1)
        match_mask = np.max(batch_scores, axis=1) > threshold
        
        for j, (idx, matches) in enumerate(zip(job_indices, match_mask)):
            if matches:
                matched_applicants.append((
                    batch.iloc[j]['name'],
                    job_posts.iloc[idx]['title']
                ))
    
    return matched_applicants 