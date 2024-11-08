from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import pickle
import logging
import platform

class ResumeMatchingTransformer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        
        # Check if running on Apple Silicon
        if platform.processor() == 'arm' and platform.system() == 'Darwin':
            # Use MPS (Metal Performance Shaders) for Apple Silicon
            self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        else:
            # Fall back to CUDA or CPU for other systems
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        logging.info(f"Using device: {self.device}")
        self.model.to(self.device)

    def encode_texts(self, texts, batch_size=32):
        try:
            # Convert texts to tensor and move to appropriate device
            return self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                device=self.device,
                convert_to_tensor=True
            ).cpu().numpy()  # Convert back to numpy for cosine similarity
        except RuntimeError as e:
            logging.warning(f"Error using {self.device}, falling back to CPU: {e}")
            self.device = 'cpu'
            self.model.to(self.device)
            return self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                device=self.device
            )

    def train_model(self, job_requirements, resumes, batch_size=32):
        logging.info(f"Starting encoding on {self.device}")
        logging.info("Encoding job requirements...")
        job_embeddings = self.encode_texts(job_requirements, batch_size)
        
        logging.info("Encoding resumes...")
        resume_embeddings = self.encode_texts(resumes, batch_size)
        
        logging.info("Calculating similarity scores...")
        # Move calculations to MPS if available
        if self.device == 'mps':
            job_embeddings_tensor = torch.tensor(job_embeddings).to(self.device)
            resume_embeddings_tensor = torch.tensor(resume_embeddings).to(self.device)
            
            # Calculate cosine similarity on MPS
            similarity_scores = torch.nn.functional.cosine_similarity(
                resume_embeddings_tensor.unsqueeze(1),
                job_embeddings_tensor.unsqueeze(0),
                dim=2
            ).cpu().numpy()
        else:
            similarity_scores = cosine_similarity(resume_embeddings, job_embeddings)
        
        # Save model and scores
        with open('models/screening_model.pkl', 'wb') as f:
            pickle.dump((self, similarity_scores), f)
            
        return similarity_scores

    def __getstate__(self):
        """Custom serialization method"""
        state = self.__dict__.copy()
        # Don't pickle the device
        state['device'] = 'cpu'
        return state

    def __setstate__(self, state):
        """Custom deserialization method"""
        self.__dict__.update(state)
        # Reinitialize device on load
        if platform.processor() == 'arm' and platform.system() == 'Darwin':
            self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
