import streamlit as st
from Model_train import Model_acc
from preprocess import automatic_preprocessing, load_data
import pandas as pd
from fecRec import detect_features_from_dataframe, predict_algorithm_for_dataset

def main():
    st.title("Spot Checking Website")
    st.write("Upload your dataset and let us find the best model for you!")

    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        # Preprocess data
        df_preprocessed = automatic_preprocessing(df)
        
        # Feature detection of dataset
        features_detected = detect_features_from_dataframe(df_preprocessed)
        
        # Predicted Algorithm Type
        predicted_algorithm = predict_algorithm_for_dataset(features_detected)
        
        # Model Training and Evaluation
        st.write("Model Training and Evaluation:")
        target = st.text_input("Enter the target column name (y_col): ", "")
        
        if st.button("Train Model"):
            if target and target in df_preprocessed.columns:
                table, model, acc = Model_acc(target, predicted_algorithm, df_preprocessed)
                st.write(table)
                st.write("Best Algorithm:", str(model))
                st.write("Best Accuracy:", acc)
            elif target:
                st.error(f"Selected column '{target}' is not in the preprocessed DataFrame.")
            else:
                st.error("Please enter a target column name.")

if __name__ == "__main__":
    main()
