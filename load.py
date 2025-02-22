import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

def main():
    st.title("Dataset Loader and Predictor")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of the dataset:")
        st.write(df.head())
        
        model = load_model()
        
        # Select features
        feature_columns = st.multiselect("Select feature columns", df.columns)
        
        if st.button("Predict"):
            if feature_columns:
                X = df[feature_columns]
                predictions = model.predict(X)
                df['Prediction'] = predictions
                st.write("### Predictions:")
                st.write(df[['Prediction']])
            else:
                st.warning("Please select at least one feature column.")
    
if __name__ == "__main__":
    main()
