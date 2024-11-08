import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
from data_preprocessing import preprocess_data
from model_training import train_model, train_transformer_model
import os

class CSVChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith('.csv'):
            logging.info(f"Detected change in {event.src_path}")
            self.retrain_model()

    def on_created(self, event):
        if event.src_path.endswith('.csv'):
            logging.info(f"New CSV file created: {event.src_path}")
            self.retrain_model()

    def retrain_model(self):
        try:
            # Add a small delay to ensure file writing is complete
            time.sleep(1)
            
            logging.info("Starting model retraining...")
            
            # Create models directory if it doesn't exist
            if not os.path.exists('models'):
                os.makedirs('models')
            
            # Preprocess data
            job_posts, applicants = preprocess_data('data/job_posts.csv', 'data/applicants.csv')
            
            # Train model
            model_type = os.getenv('MODEL_TYPE', 'transformer')
            if model_type == 'transformer':
                train_transformer_model(job_posts, applicants)
            else:
                train_model(job_posts, applicants)
            
            logging.info("Model retraining completed successfully")
        except Exception as e:
            logging.error(f"Error during model retraining: {str(e)}")

def start_monitoring():
    event_handler = CSVChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path='data', recursive=False)
    observer.start()
    logging.info("Started monitoring CSV files in data directory")
    return observer 