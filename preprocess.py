import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from scipy.stats import zscore
import numpy as np

def missing_data_imputation(df):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    imputer = SimpleImputer(strategy='mean')
    
    for col in numerical_cols:
        df[col] = imputer.fit_transform(df[[col]])
    
    return df

def encode_categorical_variables(df, threshold=10):
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if not categorical_cols:
        print("No categorical variables found.")
        return df

    df_encoded = df.copy()

    for col in categorical_cols:
        unique_categories = df[col].nunique()
        if unique_categories <= threshold:
            onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
            onehot_encoded = onehot_encoder.fit_transform(df[col].values.reshape(-1, 1))
            new_col_names = [f"{col}_{i}" for i in range(onehot_encoded.shape[1])]
            df_encoded.drop(columns=[col], inplace=True)
            df_encoded[new_col_names] = onehot_encoded
        else:
            label_encoder = LabelEncoder()
            label_encoded = label_encoder.fit_transform(df[col])
            df_encoded[col] = label_encoded

    return df_encoded

def scale_numerical_features(df):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    print("Feature scaling (min-max normalization) applied.")
    return df

def remove_outliers(df):
    z_scores = zscore(df.select_dtypes(include=['float64', 'int64']))
    outlier_indices = ((z_scores > 3) | (z_scores < -3)).any(axis=1)
    if outlier_indices.sum() > 0:
        print("Outliers detected and removed.")
        df = df[~outlier_indices]
    else:
        print("No outliers detected.")
    return df

def remove_highly_correlated_features(df, threshold=0.8):
    correlation_matrix = df.corr().abs()
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    correlated_features = [col for col in upper_triangle.columns if any(upper_triangle[col] > threshold)]
    if correlated_features:
        print("Highly correlated features detected and removed.")
        df.drop(columns=correlated_features, inplace=True)
    else:
        print("No highly correlated features detected.")
    return df

def automatic_preprocessing(df):
    df_original = df.copy()  # Preserve the original DataFrame

    # Step 1: Check for missing data
    if df.isnull().values.any():
        print("Missing data detected.")
        df = missing_data_imputation(df)
    else:
        print("No missing data.")

    # Step 2: Check for categorical variables
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        print("Categorical variables detected.")
        df = encode_categorical_variables(df)
    else:
        print("No categorical variables.")

    # Step 3: Scale numerical features
    df = scale_numerical_features(df)

    # Step 4: Check for outliers
    df = remove_outliers(df)

    # Step 5: Remove highly correlated features
    df = remove_highly_correlated_features(df) 

    return df

from bestcol import identify_target_columns, select_best_columns

#def load_data(file_name):
 #   path = f'C:/Users/Swetha Pooduru/Desktop/spot_check_final/spot_check/webapp/datasets/'+file_name
  #  if file_name.endswith('.csv'):
   #     df = pd.read_csv(path)
    #elif file_name.endswith('.xlsx'):
     #   df = pd.read_excel(path)
    #else:
        # Assuming you have defined the connection string and table name
     #   connection_string = "your_connection_string"
      #  table_name = "your_table_name"
       # engine = create_engine(connection_string)
        # Define your SQL query
        #sql_query = f"SELECT * FROM {table_name}"
        # Execute the query and retrieve data as a DataFrame
       # df = pd.read_sql_query(sql_query, engine)
        #engine.dispose()
    #return df

def load_data(uploaded_file):
    # Check if file is uploaded
    if uploaded_file is not None:
        # Read the file contents
        df = pd.read_csv(uploaded_file)
        return df
    else:
        return None

