import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🔍",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model-training/model.h5')

model = load_model()

# Load encoders and scaler
@st.cache_resource
def load_encoders_and_scaler():
    with open('encoders/onehot_enoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)

    with open('encoders/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    return onehot_encoder_geo, scaler

onehot_encoder_geo, scaler = load_encoders_and_scaler()

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("🔍 Customer Churn Prediction Dashboard")
st.markdown("""
Welcome to the enhanced **Customer Churn Prediction Dashboard**. This tool provides detailed insights 
into customer churn probability along with comprehensive analysis of customer metrics.
""")

# Create two columns for the main layout
col1, col2 = st.columns([2, 1])

# Sidebar for inputs
with st.sidebar:
    st.header("Customer Details Input")
    
    # Create tabs for organized input
    demo_tab, financial_tab, account_tab = st.tabs(["Demographics", "Financial", "Account"])
    
    with demo_tab:
        geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
        gender = st.selectbox('Gender', ['Male', 'Female'])
        age = st.slider('Age', 18, 92, value=30)
        
    with financial_tab:
        balance = st.number_input('Balance', value=0.0, min_value=0.0, step=100.0)
        credit_score = st.number_input('Credit Score', value=650, min_value=300, max_value=900, step=10)
        estimated_salary = st.number_input('Estimated Salary', value=50000.0, step=1000.0)
        
    with account_tab:
        tenure = st.slider('Tenure', 0, 10, value=5)
        num_of_products = st.slider('Number of Products', 1, 4, value=2)
        has_cr_card = st.selectbox('Has Credit Card', ['Yes', 'No'])
        is_active_member = st.selectbox('Is Active Member', ['Yes', 'No'])

# Prepare input data
encoded_gender = 0 if gender == 'Male' else 1
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [encoded_gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [1 if has_cr_card == 'Yes' else 0],
    'IsActiveMember': [1 if is_active_member == 'Yes' else 0],
    'EstimatedSalary': [estimated_salary]
})

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
input_data_scaled = scaler.transform(input_data)

# Get prediction
prediction = model.predict(input_data_scaled)[0][0]

# Main column - Prediction and Charts
with col1:
    # Prediction Section
    st.header("Churn Prediction Analysis")
    
    # Enhanced Gauge Chart
    gauge_color = "red" if prediction > 0.5 else "green"

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction * 100,
        title={'text': "Churn Probability (%)", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "darkblue"},
            'bar': {'color': gauge_color},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))

    st.plotly_chart(fig_gauge, use_container_width=True)

    # Add Prediction Message
    if prediction > 0.5:
        st.markdown(
            "<h3 style='color: red;'>The customer is most likely to churn.</h3>", 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<h3 style='color: green;'>The customer is not likely to churn.</h3>", 
            unsafe_allow_html=True
        )

    # Bar Chart for Customer Attributes
    customer_data = pd.DataFrame({
        "Attribute": ["Credit Score", "Balance", "Tenure", "Products"],
        "Value": [credit_score, balance, tenure, num_of_products]
    })

    fig_bar = px.bar(
        customer_data,
        x="Attribute",
        y="Value",
        color="Value",
        text_auto=True,
        title="Customer Key Attributes",
        labels={"Value": "Metric Value", "Attribute": "Customer Attribute"}
    )
    fig_bar.update_traces(marker_color='indigo', textfont_size=12)
    st.plotly_chart(fig_bar, use_container_width=True)

# Side column - Customer Profile and Metrics
with col2:
    st.header("Customer Profile")
    
    # Demographics
    st.subheader("Demographics")
    st.metric("Geography", geography)
    st.metric("Gender", gender)
    st.metric("Age", f"{age} years")
    
    # Financial Metrics
    st.subheader("Financial Metrics")
    st.metric("Credit Score", credit_score)
    st.metric("Balance", f"${balance:,.2f}")
    st.metric("Estimated Salary", f"${estimated_salary:,.2f}")
    
    # Account Details
    st.subheader("Account Details")
    st.metric("Number of Products", num_of_products)
    st.metric("Credit Card", has_cr_card)
    st.metric("Active Member", is_active_member)

# Footer
st.markdown("---")
with st.expander("About this Dashboard"):
    st.markdown("""
    This enhanced dashboard provides a comprehensive view of customer churn prediction:
    - Real-time churn probability calculation
    - Detailed customer profiling
    - Key risk indicators
    
    The model uses machine learning to analyze multiple factors that influence customer churn.
    """)

st.markdown("Developed with ❤️ using TensorFlow, Streamlit, and Scikit-learn")
 