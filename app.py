# app.py - Complete AML Fraud Detection Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set the page configuration
st.set_page_config(
    page_title="Malawi AML Investigator Dashboard",
    page_icon="🕵️",
    layout="wide"
)

# App title and description
st.title("🕵️ Malawi AML Fraud Detection Dashboard")
st.markdown("""
**Welcome, Investigator.** This tool prioritizes AML alerts using machine learning.
It helps you focus on the highest-risk cases first, saving time and improving efficiency.
""")

# Sidebar for controls
st.sidebar.header("Control Panel")
st.sidebar.info("Configure your analysis settings here")

# 1. Data Selection
data_option = st.sidebar.radio(
    "Choose data source:",
    ("Generate Sample Data", "Upload Your Data")
)

# Initialize session state for data
if 'current_data' not in st.session_state:
    st.session_state.current_data = None

if data_option == "Generate Sample Data":
    # Generate synthetic Malawian financial data
    np.random.seed(42)
    n_samples = st.sidebar.slider("Number of alerts to generate", 10, 200, 50)
    
    sample_data = {
        'alert_id': range(1, n_samples + 1),
        'customer_age_days': np.random.lognormal(7.8, 1.3, n_samples).astype(int),
        'transaction_count_7d': np.random.poisson(12, n_samples) + np.random.randint(0, 25, n_samples),
        'avg_amount_mwk': np.abs(np.random.normal(280000, 120000, n_samples)),
        'amount_volatility': np.abs(np.random.normal(95000, 60000, n_samples)),
        'is_cross_border': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
        'is_pep': np.random.choice([0, 1], n_samples, p=[0.96, 0.04]),
        'time_since_last_alert': np.random.exponential(30, n_samples).astype(int),
    }
    
    df = pd.DataFrame(sample_data)
    st.session_state.current_data = df
    st.sidebar.success(f"✅ Generated {n_samples} sample alerts")

else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file with alerts",
        type=["csv"],
        help="File should include columns like transaction amounts, customer info, etc."
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.current_data = df
            st.sidebar.success("✅ File uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
    else:
        st.info("👆 Please upload a CSV file or use sample data")
        st.stop()

# 2. Display the data
st.subheader("📋 Alert Data Overview")
st.dataframe(st.session_state.current_data, use_container_width=True)

# Show basic statistics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Alerts", len(st.session_state.current_data))
with col2:
    pep_count = st.session_state.current_data['is_pep'].sum() if 'is_pep' in st.session_state.current_data.columns else 0
    st.metric("PEP Alerts", pep_count)
with col3:
    cross_border = st.session_state.current_data['is_cross_border'].sum() if 'is_cross_border' in st.session_state.current_data.columns else 0
    st.metric("Cross-Border", cross_border)

# 3. Risk Analysis Section
st.subheader("🎯 Risk Analysis")

# Simple rule-based scoring as a placeholder
# In a real app, you would load a trained ML model here
if st.button("🚀 Calculate Risk Scores", type="primary"):
    with st.spinner("Analyzing transactions and calculating risks..."):
        df = st.session_state.current_data.copy()
        
        # Simple heuristic risk scoring (replace with your ML model)
        risk_score = (
            (df.get('is_pep', 0) * 0.3) +
            (df.get('is_cross_border', 0) * 0.2) +
            (df.get('avg_amount_mwk', 0) / 1000000 * 0.25) +
            (df.get('amount_volatility', 0) / 500000 * 0.15) +
            (np.log1p(df.get('transaction_count_7d', 0)) * 0.1)
        )
        
        # Normalize to 0-1 scale
        risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min() + 1e-8)
        df['risk_score'] = risk_score
        
        # Sort by risk
        df_prioritized = df.sort_values('risk_score', ascending=False)
        st.session_state.current_data = df_prioritized
        
        st.success("✅ Risk analysis complete!")

# Display results if we have risk scores
if 'risk_score' in st.session_state.current_data.columns:
    # Show prioritized table
    st.subheader("📊 Prioritized Investigation List")
    st.info("Investigate alerts from the top down (highest risk first)")
    
    # Color code by risk
    styled_df = st.session_state.current_data.style.background_gradient(
        subset=['risk_score'],
        cmap='Reds',
        vmin=0,
        vmax=1
    )
    st.dataframe(styled_df, use_container_width=True)
    
    # High-risk alerts section
    st.subheader("🔴 High-Risk Alerts")
    high_risk_threshold = st.slider("High-risk threshold", 0.0, 1.0, 0.7, 0.05)
    high_risk_alerts = st.session_state.current_data[st.session_state.current_data['risk_score'] >= high_risk_threshold]
    
    st.metric("High-Risk Alerts", len(high_risk_alerts), 
              f"{len(high_risk_alerts)/len(st.session_state.current_data)*100:.1f}% of total")
    
    if not high_risk_alerts.empty:
        st.dataframe(high_risk_alerts, use_container_width=True)
    else:
        st.warning("No alerts meet the high-risk threshold")
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Risk Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(st.session_state.current_data['risk_score'], bins=20, alpha=0.7, color='red', edgecolor='black')
        ax.axvline(high_risk_threshold, color='darkred', linestyle='--', label=f'Threshold ({high_risk_threshold})')
        ax.set_xlabel('Risk Score')
        ax.set_ylabel('Number of Alerts')
        ax.legend()
        ax.set_title('Distribution of Risk Scores')
        st.pyplot(fig)
    
    with col2:
        st.subheader("🔄 Export Results")
        if st.download_button(
            label="Download Prioritized CSV",
            data=st.session_state.current_data.to_csv(index=False),
            file_name="prioritized_aml_alerts.csv",
            mime="text/csv"
        ):
            st.success("Download ready!")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built for Malawi's Financial Sector • Powered by Streamlit")
