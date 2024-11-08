import pandas as pd

def preprocess_data(job_posts_file, applicants_file):
    # Load job posts and applicants data
    job_posts = pd.read_csv(job_posts_file)
    applicants = pd.read_csv(applicants_file)
    
    # Example preprocessing: lowercasing and removing special characters
    job_posts['requirements'] = job_posts['requirements'].str.lower().str.replace('[^a-z0-9 ]', '')
    applicants['resume'] = applicants['resume'].str.lower().str.replace('[^a-z0-9 ]', '')
    
    return job_posts, applicants 