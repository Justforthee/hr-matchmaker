from data_preprocessing import preprocess_data
from model_training import train_model, train_transformer_model
from applicant_screening import screen_applicants
from logger import setup_logger, log_screening_results
from file_monitor import start_monitoring
import os
import time

def main():
    setup_logger()
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Initial data processing and model training
    job_posts, applicants = preprocess_data('data/job_posts.csv', 'data/applicants.csv')
    
    # Use environment variable to choose model type
    model_type = os.getenv('MODEL_TYPE', 'transformer')  # default to transformer

    if model_type == 'transformer':
        train_transformer_model(job_posts, applicants)
    else:
        train_model(job_posts, applicants)
        
    results = screen_applicants(applicants, job_posts)
    log_screening_results(results)
    
    # Start file monitoring
    observer = start_monitoring()
    
    print("\nModel training and screening completed!")
    print("File monitoring started - model will automatically retrain when CSV files change")
    print("To launch the dashboard, run: streamlit run src/dashboard.py")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nFile monitoring stopped")
    
    observer.join()

if __name__ == "__main__":
    main() 