"""
Streamlit Web Application for Alzheimer's Disease Detection

This application provides an interactive interface for predicting
Alzheimer's disease using MRI biomarker data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import sys
from src.explainability import ModelExplainer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_loader import DataLoader
from src.models import ModelTrainer


# Page configuration
st.set_page_config(
    page_title="Alzheimer's Disease Detection",
    page_icon="🧠",
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
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def load_model(model_path: str):
    """Load a saved model from file."""
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None


def predict_alzheimers(model, features: dict, scaler) -> dict:
    """
    Make prediction using the trained model.
    
    Args:
        model: Trained ML model
        features: Dictionary of feature values
        scaler: Fitted scaler for feature normalization
        
    Returns:
        Dictionary with prediction and probability
    """
    # Create feature array in correct order
    feature_order = ['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
    feature_array = np.array([[features[feat] for feat in feature_order]])
    
    # Scale features
    feature_array_scaled = scaler.transform(feature_array)
    
    # Make prediction
    prediction = model.predict(feature_array_scaled)[0]
    probability = model.predict_proba(feature_array_scaled)[0]
    
    return {
        'prediction': 'Demented' if prediction == 1 else 'Nondemented',
        'probability_demented': probability[1] if len(probability) > 1 else 0.0,
        'probability_nondemented': probability[0]
    }


def explainability_page():
    """Explainability interface for tabular data."""
    st.header("🔍 Model Explainability")
    st.markdown("Understand **why** the model makes its predictions using SHAP values.")
    
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    import shap
    
    model_path = Path("models/best_model.pkl")
    scaler_path = Path("models/scaler.pkl")
    
    if not model_path.exists():
        st.warning("⚠️ No trained model found. Please train a model first.")
        return
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        scaler = None
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        
        st.success("✅ Model loaded!")
        
        st.subheader("📥 Input Features")
        st.markdown("Enter feature values for explanation:")
        
        # Feature inputs (adjust based on your features)
        col1, col2 = st.columns(2)
        with col1:
            feature1 = st.number_input("Feature 1", value=0.0)
            feature2 = st.number_input("Feature 2", value=0.0)
        with col2:
            feature3 = st.number_input("Feature 3", value=0.0)
            feature4 = st.number_input("Feature 4", value=0.0)
        
        instance = np.array([[feature1, feature2, feature3, feature4]])
        
        if scaler:
            instance = scaler.transform(instance)
        
        if st.button("🔍 Explain"):
            with st.spinner("Computing SHAP values..."):
                try:
                    # Create explainer
                    explainer = ModelExplainer(model, instance, feature_names=None)
                    shap_values = explainer.explain_instance(instance, plot=False)
                    
                    st.subheader("📊 SHAP Explanation")
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=shap_values[0],
                            base_values=0,
                            data=instance[0]
                        ),
                        show=False
                    )
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Alzheimer\'s Disease Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Early Detection Using MRI Biomarker Data</p>', unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose a mode",
        ["Prediction", "Model Training", "Explainability", "About"]
    )
    
    if app_mode == "Prediction":
        prediction_page()
    elif app_mode == "Model Training":
        training_page()
    
    elif app_mode == "Explainability":
        explainability_page()
    elif app_mode == "About":
        about_page()


def prediction_page():
    """Prediction interface page."""
    st.header("🔮 Make a Prediction")
    st.markdown("Enter patient MRI biomarker data to predict Alzheimer's disease risk.")
    
    # Check if model exists
    model_path = "models/best_model.pkl"
    scaler_path = "models/scaler.pkl"
    
    if not os.path.exists(model_path):
        st.warning("⚠️ No trained model found. Please train a model first using the 'Model Training' page.")
        st.info("💡 You can also upload a pre-trained model file.")
        uploaded_model = st.file_uploader("Upload Model File", type=['pkl'])
        if uploaded_model:
            # Save uploaded model temporarily
            with open("temp_model.pkl", "wb") as f:
                f.write(uploaded_model.read())
            model_path = "temp_model.pkl"
    
    if os.path.exists(model_path):
        # Load model and scaler
        model = load_model(model_path)
        scaler = load_model(scaler_path) if os.path.exists(scaler_path) else None
        
        if model is None:
            st.error("❌ Error loading model. Please check the model file.")
            return
        
        # Input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Patient Demographics")
                gender = st.selectbox("Gender", ["Female", "Male"])
                age = st.number_input("Age (years)", min_value=60, max_value=100, value=75)
                education = st.number_input("Years of Education (EDUC)", min_value=6, max_value=23, value=14)
                ses = st.selectbox("Socioeconomic Status (SES)", [1, 2, 3, 4, 5], index=2)
            
            with col2:
                st.subheader("MRI Biomarkers")
                mmse = st.number_input(
                    "Mini Mental State Examination (MMSE)",
                    min_value=0,
                    max_value=30,
                    value=27,
                    help="Score range: 0-30 (higher is better)"
                )
                etiv = st.number_input(
                    "Estimated Total Intracranial Volume (eTIV)",
                    min_value=1000,
                    max_value=2000,
                    value=1490
                )
                nwbv = st.slider(
                    "Normalized Whole Brain Volume (nWBV)",
                    min_value=0.60,
                    max_value=0.85,
                    value=0.73,
                    step=0.01,
                    help="Ratio of brain volume (0.6-0.85)"
                )
                asf = st.slider(
                    "Atlas Scaling Factor (ASF)",
                    min_value=0.80,
                    max_value=1.60,
                    value=1.20,
                    step=0.01
                )
            
            submitted = st.form_submit_button("🔍 Predict", use_container_width=True)
            
            if submitted:
                # Prepare features
                features = {
                    'M/F': 1 if gender == "Male" else 0,
                    'Age': age,
                    'EDUC': education,
                    'SES': ses,
                    'MMSE': mmse,
                    'eTIV': etiv,
                    'nWBV': nwbv,
                    'ASF': asf
                }
                
                # Make prediction
                if scaler is None:
                    st.warning("⚠️ Scaler not found. Using raw features (may affect accuracy).")
                    result = predict_alzheimers(model, features, None)
                else:
                    result = predict_alzheimers(model, features, scaler)
                
                # Display results
                st.success("✅ Prediction Complete!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Prediction",
                        result['prediction'],
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "Probability (Demented)",
                        f"{result['probability_demented']:.1%}",
                        delta=None
                    )
                
                with col3:
                    st.metric(
                        "Probability (Nondemented)",
                        f"{result['probability_nondemented']:.1%}",
                        delta=None
                    )
                
                # Visualize probability
                st.subheader("📊 Prediction Confidence")
                prob_data = pd.DataFrame({
                    'Status': ['Nondemented', 'Demented'],
                    'Probability': [
                        result['probability_nondemented'],
                        result['probability_demented']
                    ]
                })
                st.bar_chart(prob_data.set_index('Status'))
                
                # Interpretation
                st.subheader("💡 Interpretation")
                if result['prediction'] == 'Demented':
                    st.warning(
                        f"⚠️ The model predicts **Demented** with {result['probability_demented']:.1%} confidence. "
                        "Please consult with a healthcare professional for further evaluation."
                    )
                else:
                    st.success(
                        f"✅ The model predicts **Nondemented** with {result['probability_nondemented']:.1%} confidence. "
                        "However, regular monitoring is still recommended."
                    )


def training_page():
    """Model training interface page."""
    st.header("🤖 Train a Model")
    st.markdown("Train machine learning models on MRI biomarker data.")
    
    # File upload
    st.subheader("📁 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload CSV file with MRI data",
        type=['csv'],
        help="The CSV should contain columns: M/F, Age, EDUC, SES, MMSE, eTIV, nWBV, ASF, Group"
    )
    
    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df)} records")
        
        # Display data preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10))
        
        # Model selection
        st.subheader("🎯 Select Models to Train")
        models_to_train = st.multiselect(
            "Choose models",
            ["Logistic Regression", "SVM", "Decision Tree", "Random Forest", "AdaBoost"],
            default=["Random Forest"]
        )
        
        # Training parameters
        with st.expander("⚙️ Advanced Settings"):
            test_size = st.slider("Test Set Size", 0.1, 0.4, 0.25, 0.05)
            kfolds = st.number_input("Cross-Validation Folds", 3, 10, 5)
            random_state = st.number_input("Random State", 0, 100, 0)
        
        # Train button
        if st.button("🚀 Start Training", use_container_width=True):
            if len(models_to_train) == 0:
                st.error("Please select at least one model to train.")
            else:
                with st.spinner("Training models... This may take a while."):
                    try:
                        # Initialize data loader
                        data_loader = DataLoader(uploaded_file.name)
                        data_loader.df = df  # Use uploaded data
                        data_loader.preprocess_data()
                        
                        # Split data
                        X_trainval, X_test, Y_trainval, Y_test = data_loader.split_data(
                            test_size=test_size,
                            random_state=random_state
                        )
                        
                        # Scale features
                        X_trainval_scaled, X_test_scaled = data_loader.scale_features()
                        
                        # Initialize trainer
                        trainer = ModelTrainer(kfolds=kfolds)
                        
                        # Train selected models
                        progress_bar = st.progress(0)
                        results = []
                        
                        for i, model_name in enumerate(models_to_train):
                            progress_bar.progress((i + 1) / len(models_to_train))
                            
                            if model_name == "Logistic Regression":
                                result = trainer.train_logistic_regression(
                                    X_trainval_scaled, Y_trainval,
                                    X_test_scaled, Y_test
                                )
                            elif model_name == "SVM":
                                result = trainer.train_svm(
                                    X_trainval_scaled, Y_trainval,
                                    X_test_scaled, Y_test
                                )
                            elif model_name == "Decision Tree":
                                result = trainer.train_decision_tree(
                                    X_trainval_scaled, Y_trainval,
                                    X_test_scaled, Y_test
                                )
                            elif model_name == "Random Forest":
                                result = trainer.train_random_forest(
                                    X_trainval_scaled, Y_trainval,
                                    X_test_scaled, Y_test
                                )
                            elif model_name == "AdaBoost":
                                result = trainer.train_adaboost(
                                    X_trainval_scaled, Y_trainval,
                                    X_test_scaled, Y_test
                                )
                            
                            results.append(result)
                        
                        progress_bar.empty()
                        
                        # Display results
                        st.success("✅ Training Complete!")
                        
                        # Results summary
                        summary = trainer.get_results_summary()
                        st.subheader("📊 Model Performance Summary")
                        st.dataframe(summary.style.highlight_max(axis=0))
                        
                        # Find best model
                        best_model_idx = summary['Accuracy'].idxmax()
                        best_model_name = summary.loc[best_model_idx, 'Model']
                        best_result = results[best_model_idx]
                        
                        st.success(f"🏆 Best Model: **{best_model_name}** (Accuracy: {summary.loc[best_model_idx, 'Accuracy']:.2%})")
                        
                        # Save best model
                        os.makedirs("models", exist_ok=True)
                        with open("models/best_model.pkl", "wb") as f:
                            pickle.dump(best_result['model'], f)
                        with open("models/scaler.pkl", "wb") as f:
                            pickle.dump(data_loader.scaler, f)
                        
                        st.info("💾 Best model saved to `models/best_model.pkl`")
                        
                    except Exception as e:
                        st.error(f"❌ Error during training: {str(e)}")
                        st.exception(e)


def about_page():
    """About page with project information."""
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ## 🧠 Alzheimer's Disease Detection System
    
    This application uses machine learning to detect early signs of Alzheimer's disease
    using MRI biomarker data from the Open Access Series of Imaging Studies (OASIS) dataset.
    
    ### 📊 Features Used
    
    The model analyzes the following MRI biomarkers and patient characteristics:
    
    - **Demographics**: Gender, Age, Education Level, Socioeconomic Status
    - **Cognitive Assessment**: Mini Mental State Examination (MMSE) score
    - **Brain Volume Metrics**: 
      - Estimated Total Intracranial Volume (eTIV)
      - Normalized Whole Brain Volume (nWBV)
      - Atlas Scaling Factor (ASF)
    
    ### 🤖 Models Available
    
    The system supports multiple machine learning algorithms:
    
    1. **Logistic Regression** - Linear classification model
    2. **Support Vector Machine (SVM)** - Non-linear classification with various kernels
    3. **Decision Tree** - Interpretable tree-based model
    4. **Random Forest** - Ensemble of decision trees
    5. **AdaBoost** - Adaptive boosting ensemble method
    
    ### 📈 Performance Metrics
    
    Models are evaluated using:
    - **Accuracy**: Overall classification accuracy
    - **Recall**: Ability to detect true positive cases
    - **AUC**: Area Under the ROC Curve
    
    ### ⚠️ Important Disclaimer
    
    This tool is for research and educational purposes only. It should not be used
    as a substitute for professional medical diagnosis or treatment. Always consult
    with qualified healthcare professionals for medical decisions.
    
    ### 📚 References
    
    - OASIS Dataset: [www.oasis-brains.org](http://www.oasis-brains.org)
    - Original Research: Marcus DS, et al. (2010). Open Access Series of Imaging Studies (OASIS)
    """)


if __name__ == "__main__":
    main()

