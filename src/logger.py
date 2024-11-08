import logging

def setup_logger():
    # Configure logging to file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/screening.log'),
            logging.StreamHandler()  # This will print logs to the console
        ]
    )

def log_screening_results(results):
    for applicant, job in results:
        logging.info(f"Applicant {applicant} matched with job {job}") 