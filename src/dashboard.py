import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
from data_preprocessing import preprocess_data
from model_training import train_model
from applicant_screening import screen_applicants

def load_model_and_scores():
    with open('models/screening_model.pkl', 'rb') as f:
        model, similarity_scores = pickle.load(f)
    return similarity_scores

def create_dashboard():
    st.title("HR Screening & Matching Dashboard")
    st.sidebar.header("Controls")

    # Load data
    job_posts, applicants = preprocess_data('data/job_posts.csv', 'data/applicants.csv')
    
    # Load similarity scores
    try:
        similarity_scores = load_model_and_scores()
    except FileNotFoundError:
        st.warning("Training model first...")
        train_model(job_posts, applicants)
        similarity_scores = load_model_and_scores()

    # Sidebar filters
    min_score = st.sidebar.slider("Minimum Match Score", 0.0, 1.0, 0.5, 0.1)
    max_applicants = st.sidebar.slider("Maximum Applicants to Show", 1, 10, 10, 1)
    selected_job = st.sidebar.selectbox(
        "Select Job Position",
        options=job_posts['title'].unique()
    )

    # Display job requirements in sidebar
    st.sidebar.markdown("---")  # Add a visual separator
    st.sidebar.subheader(f"Job Requirements for _{selected_job}_")
    st.sidebar.write(job_posts[job_posts['title'] == selected_job]['requirements'].iloc[0])
    st.sidebar.markdown("---")  # Add a visual separator

    # Main content
    st.header(f"Matching Results for {selected_job}")

    # Get job index
    job_idx = job_posts[job_posts['title'] == selected_job].index[0]
    
    # Get matching scores for this job
    job_matches = []
    for i, applicant in applicants.iterrows():
        score = similarity_scores[i][job_idx]
        if score >= min_score:
            job_matches.append({
                'Applicant': applicant['name'],
                'Match Score': score,
                'Resume': applicant['resume']
            })

    # Sort matches by score and limit to max_applicants
    job_matches.sort(key=lambda x: x['Match Score'], reverse=True)
    job_matches = job_matches[:max_applicants]

    # Display matches
    if job_matches:
        # Create a bar chart of match scores
        fig = go.Figure(data=[
            go.Bar(
                x=[match['Applicant'] for match in job_matches],
                y=[match['Match Score'] for match in job_matches],
                marker_color='rgb(26, 118, 255)'
            )
        ])
        
        fig.update_layout(
            title='Applicant Match Scores',
            xaxis_title='Applicants',
            yaxis_title='Match Score',
            yaxis_range=[0, 1]
        )
        
        st.plotly_chart(fig)

        # Detailed matches table
        st.subheader("Detailed Matches")
        for match in job_matches:
            with st.expander(f"{match['Applicant']} - Score: {match['Match Score']:.2f}"):
                st.write("**Resume Highlights:**")
                st.write(match['Resume'])

    else:
        st.warning("No matches found with the current minimum score threshold.")

if __name__ == "__main__":
    create_dashboard()