"""
Streamlit Web Application for Student Placement Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-top: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
    }
    .failure-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_encoders():
    """Load the trained model, scaler, and encoders"""
    try:
        model = joblib.load('placement_model.pkl')
        scaler = joblib.load('scaler.pkl')
        le_extracurricular = joblib.load('label_encoder_extracurricular.pkl')
        le_department = joblib.load('label_encoder_department.pkl')
        return model, scaler, le_extracurricular, le_department
    except FileNotFoundError as e:
        st.error(f"Model files not found. Please run train_model.py first to train the model.")
        st.stop()
        return None, None, None, None

def predict_placement(features, model, scaler):
    """Make prediction using the trained model"""
    # Scale features
    features_scaled = scaler.transform([features])
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    return prediction, probability

def main():
    """Main Streamlit application"""
    # Header
    st.markdown('<h1 class="main-header">🎓 Student Placement Predictor</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model
    model, scaler, le_extracurricular, le_department = load_model_and_encoders()
    
    if model is None:
        return
    
    # Sidebar for input
    st.sidebar.header("📊 Student Information")
    st.sidebar.markdown("Enter the student's details to predict placement probability")
    
    # Input fields
    cgpa = st.sidebar.slider(
        "CGPA (Cumulative Grade Point Average)",
        min_value=6.0,
        max_value=10.0,
        value=8.0,
        step=0.1
    )
    
    iq = st.sidebar.slider(
        "IQ Score",
        min_value=80,
        max_value=140,
        value=110,
        step=1
    )
    
    communication_skills = st.sidebar.slider(
        "Communication Skills (1-10)",
        min_value=1,
        max_value=10,
        value=7,
        step=1
    )
    
    technical_skills = st.sidebar.slider(
        "Technical Skills (1-10)",
        min_value=1,
        max_value=10,
        value=7,
        step=1
    )
    
    internships = st.sidebar.slider(
        "Number of Internships",
        min_value=0,
        max_value=3,
        value=1,
        step=1
    )
    
    projects = st.sidebar.slider(
        "Number of Projects",
        min_value=0,
        max_value=5,
        value=2,
        step=1
    )
    
    extracurricular = st.sidebar.selectbox(
        "Extracurricular Activities Level",
        options=['High', 'Medium', 'Low']
    )
    
    department = st.sidebar.selectbox(
        "Department",
        options=['CSE', 'ECE', 'EEE', 'ME', 'CE']
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Student Profile Summary")
        
        # Display input summary
        profile_data = {
            'Attribute': [
                'CGPA', 'IQ Score', 'Communication Skills', 'Technical Skills',
                'Internships', 'Projects', 'Extracurricular', 'Department'
            ],
            'Value': [
                f"{cgpa:.1f}",
                f"{iq}",
                f"{communication_skills}/10",
                f"{technical_skills}/10",
                f"{internships}",
                f"{projects}",
                extracurricular,
                department
            ]
        }
        
        profile_df = pd.DataFrame(profile_data)
        st.dataframe(profile_df, use_container_width=True, hide_index=True)
        
        # Predict button
        if st.button("🔮 Predict Placement", type="primary", use_container_width=True):
            # Encode categorical variables
            extracurricular_encoded = le_extracurricular.transform([extracurricular])[0]
            department_encoded = le_department.transform([department])[0]
            
            # Prepare features
            features = np.array([
                cgpa, iq, communication_skills, technical_skills,
                internships, projects, extracurricular_encoded, department_encoded
            ])
            
            # Make prediction
            prediction, probability = predict_placement(features, model, scaler)
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            if prediction == 1:
                st.markdown(
                    f'<div class="success-box">'
                    f'<h3>✅ Placement Prediction: <strong>PLACED</strong></h3>'
                    f'<p>Probability of Placement: <strong>{probability[1]*100:.2f}%</strong></p>'
                    f'<p>Probability of Not Placed: {probability[0]*100:.2f}%</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="failure-box">'
                    f'<h3>❌ Placement Prediction: <strong>NOT PLACED</strong></h3>'
                    f'<p>Probability of Placement: {probability[1]*100:.2f}%</p>'
                    f'<p>Probability of Not Placed: <strong>{probability[0]*100:.2f}%</strong></p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            # Probability visualization
            st.markdown("### 📊 Probability Distribution")
            prob_df = pd.DataFrame({
                'Outcome': ['Not Placed', 'Placed'],
                'Probability': [probability[0]*100, probability[1]*100]
            })
            st.bar_chart(prob_df.set_index('Outcome'))
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            if prediction == 0:
                st.warning("To improve placement chances, consider:")
                recommendations = []
                if cgpa < 8.0:
                    recommendations.append("📈 Improve CGPA (aim for 8.0+)")
                if communication_skills < 7:
                    recommendations.append("💬 Enhance communication skills through practice")
                if technical_skills < 7:
                    recommendations.append("💻 Strengthen technical skills with projects")
                if internships < 2:
                    recommendations.append("🏢 Complete more internships")
                if projects < 3:
                    recommendations.append("🔧 Work on more projects")
                if extracurricular == 'Low':
                    recommendations.append("🎯 Increase extracurricular activities")
                
                for rec in recommendations:
                    st.write(rec)
            else:
                st.success("Great profile! Keep maintaining these standards.")
    
    with col2:
        st.subheader("ℹ️ About")
        st.markdown("""
        This application uses a **Random Forest Classifier** 
        trained on student data to predict placement probability.
        
        **Features Considered:**
        - Academic Performance (CGPA)
        - Cognitive Ability (IQ)
        - Soft Skills
        - Technical Skills
        - Work Experience
        - Projects
        - Extracurricular Activities
        - Department
        
        **Model Performance:**
        - Trained on 500+ student records
        - Uses ensemble learning for accuracy
        """)
        
        st.markdown("---")
        st.subheader("📈 Feature Importance")
        
        # Get feature importance from model
        feature_names = [
            'CGPA', 'IQ', 'Comm Skills', 'Tech Skills',
            'Internships', 'Projects', 'Extracurricular', 'Department'
        ]
        importance = model.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=True)
        
        st.bar_chart(importance_df.set_index('Feature'))
        
        # Instructions
        st.markdown("---")
        st.subheader("📝 Instructions")
        st.markdown("""
        1. Adjust the sliders and select options in the sidebar
        2. Click **Predict Placement** button
        3. View the prediction results and recommendations
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Built with ❤️ using Python, Pandas, Scikit-learn, and Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
