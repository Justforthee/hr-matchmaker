# HR Matchmaker

## Overview

The HR Matchmaker is an advanced application that automates recruitment screening by matching job applicants with positions using state-of-the-art Natural Language Processing (NLP) and machine learning techniques. The system leverages transformer models and optimized processing for Apple Silicon, providing real-time matching and interactive visualization.

## System Architecture

### 1. Machine Learning Core
- **Dual Model Support**
  - Transformer-based Model (Default)
    - Uses SentenceTransformer with MiniLM architecture
    - Optimized for Apple Silicon (M1/M2) using MPS
    - Automatic fallback to CPU when needed
  - TF-IDF Model (Alternative)
    - N-gram analysis (up to trigrams)
    - Custom vocabulary size and document frequency thresholds
    - Sublinear scaling for better term weighting

### 2. Data Processing Pipeline
- **Input Processing**
  - Automated CSV file monitoring
  - Real-time data validation
  - Text preprocessing and standardization
  
- **Model Training**
  - Dynamic model selection via environment variables
  - Automatic retraining on data updates
  - Optimized batch processing
  - Hardware-specific optimizations

- **Matching Engine**
  - Cosine similarity computation
  - Configurable matching thresholds
  - Batch-processed scoring
  - Memory-efficient operations

### 3. Interactive Dashboard
- **Control Panel**
  - Job position selection
  - Minimum match score adjustment
  - Maximum applicant display limit
  - Real-time job requirement display

- **Visualization**
  - Interactive match score charts
  - Detailed applicant profiles
  - Expandable resume sections
  - Score-based ranking

## Technical Features

### Machine Learning
- **Transformer Model**
  - Model: all-MiniLM-L6-v2
  - Hardware acceleration support
  - Batch processing optimization
  - Automatic device selection

- **TF-IDF Model**
  - N-gram range: (1, 3)
  - Max features: 15,000
  - Frequency thresholds: min_df=2, max_df=0.85
  - Sublinear TF scaling

### Data Management
- **File Monitoring**
  - Real-time CSV change detection
  - Automatic model retraining
  - Error handling and logging

- **Data Validation**
  - Schema validation
  - Data completeness checks
  - Duplicate detection
  - Error reporting

## Installation

1. **Environment Setup**
   ```bash
   # Create Python 3.11 virtual environment
   python3.11 -m venv thevenv
   source thevenv/bin/activate  # or `thevenv\Scripts\activate` on Windows
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the Application**
   ```bash
   # Set model type (optional)
   export MODEL_TYPE=transformer  # or 'tfidf'
   
   # Run the main application
   python src/main.py
   ```

2. **Launch Dashboard**
   ```bash
   streamlit run src/dashboard.py
   ```

## Configuration

### Model Selection
- Set `MODEL_TYPE` environment variable:
  - `transformer`: Use transformer-based model (default)
  - `tfidf`: Use TF-IDF based model

### Hardware Optimization
- Automatic detection of Apple Silicon
- MPS acceleration when available
- Graceful fallback to CPU processing

## System Components

### Core Files
- `transformer_model.py`: Transformer-based matching implementation
- `model_training.py`: Model training and evaluation logic
- `data_preprocessing.py`: Data cleaning and preparation
- `applicant_screening.py`: Candidate matching logic
- `dashboard.py`: Interactive UI implementation
- `file_monitor.py`: Data change monitoring
- `validation.py`: Data validation
- `config.yaml`: System configuration
- `logger.py`: Logging system

### Data Files
- `job_posts.csv`: Job requirements and descriptions
- `applicants.csv`: Applicant profiles and resumes
- `screening.log`: System activity logs

## Performance Considerations

- Batch processing for large datasets
- Hardware-specific optimizations
- Memory-efficient operations
- Automatic model versioning

## Future Enhancements

1. **ML Improvements**
  - Additional transformer models
  - Custom model fine-tuning
  - Multi-lingual support

2. **System Features**
  - API integration
  - Automated report generation
  - Advanced analytics dashboard

3. **Performance**
  - Distributed processing
  - Advanced caching
  - Real-time updates

## Maintenance

- Monitor `logs/screening.log`
- Regular model performance evaluation
- Data quality checks
- System health monitoring

## Support

For technical issues:
- Check system logs
- Verify data format
- Review configuration
- Contact system administrator

## Contributing

1. Fork repository
2. Create feature branch
3. Implement changes
4. Submit pull request

## License

MIT License
 