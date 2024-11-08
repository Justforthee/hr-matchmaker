import pandas as pd
import logging
from typing import Tuple, Dict, Any
import yaml

def load_config() -> Dict[str, Any]:
    try:
        with open('src/config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        raise

def validate_data(job_posts: pd.DataFrame, applicants: pd.DataFrame) -> Tuple[bool, str]:
    """Validate input data format and content"""
    try:
        # Check required columns
        required_job_cols = ['id', 'title', 'requirements']
        required_app_cols = ['id', 'name', 'resume']
        
        if not all(col in job_posts.columns for col in required_job_cols):
            return False, "Job posts missing required columns"
            
        if not all(col in applicants.columns for col in required_app_cols):
            return False, "Applicants missing required columns"
            
        # Check for empty values
        if job_posts[required_job_cols].isnull().any().any():
            return False, "Job posts contain empty values"
            
        if applicants[required_app_cols].isnull().any().any():
            return False, "Applicants contain empty values"
            
        # Check for duplicate IDs
        if job_posts['id'].duplicated().any():
            return False, "Duplicate job post IDs found"
            
        if applicants['id'].duplicated().any():
            return False, "Duplicate applicant IDs found"
            
        return True, "Data validation successful"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}" 