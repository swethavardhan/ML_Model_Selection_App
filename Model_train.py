import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from time import time
import datetime
import time
from tabulate import tabulate

def Model_acc(target, pred_algo, df_preprocessed, custom_metric=None):
    Accuracy = []
    R2_Score = []
    Mean_Squared_Error = []
    names = []
    TIME = []
    print(df_preprocessed)
    if pred_algo == 'classification':
        X = df_preprocessed.loc[:,df_preprocessed.columns!=target]
        y = df_preprocessed[target]

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define classifiers to be evaluated
        classifiers = {
            'Logistic Regression': LogisticRegression(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(),
            'AdaBoost': AdaBoostClassifier(),
            'Support Vector Machine': SVC(),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB(),
            'MLP Classifier': MLPClassifier(),
            'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
            'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }

        # Evaluate each classifier
        for name, clf in classifiers.items():
            start_time = time.time()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            end_time = time()

            accuracy = accuracy_score(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            Accuracy.append(accuracy)
            R2_Score.append(r2)
            Mean_Squared_Error.append(mse)
            names.append(name)
            TIME.append(time.time() - start)

            if custom_metric is not None:
                custom_metric_value = custom_metric(y_test, y_pred)
                Custom_Metric.append(custom_metric_value)

                if self.verbose > 0:
                    print({
                        "Model": name,
                        "Accuracy": accuracy,
                        "R^2 Score": r2,
                        "Mean Squared Error": mse,
                        custom_metric.__name__: custom_metric_value,
                        "Time taken": end_time - start
                    })

    elif pred_algo == 'regression':
        
        X = df_preprocessed.loc[:,df_preprocessed.columns!=target]
        #print(X.head())
        y = df_preprocessed[target]

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define regressors to be evaluated
        regressors = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'ElasticNet Regression': ElasticNet(),
            'Decision Tree': DecisionTreeRegressor(),
            'Random Forest': RandomForestRegressor(),
            'Gradient Boosting': GradientBoostingRegressor(),
            'AdaBoost': AdaBoostRegressor(),
            'Support Vector Machine': SVR(),
            'K-Nearest Neighbors': KNeighborsRegressor(),
            'MLP Regressor': MLPRegressor(),
            'XGBoost': XGBRegressor()
        }

        # Evaluate each regressor
        for name, reg in regressors.items():
            start = time.time()
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            end_time = time.time()

            accuracy = reg.score(X_test, y_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            Accuracy.append(accuracy)
            Mean_Squared_Error.append(mse)
            R2_Score.append(r2)
            names.append(name)
            TIME.append(end_time - start)

            if custom_metric is not None:
                custom_metric_value = custom_metric(y_test, y_pred)
                Custom_Metric.append(custom_metric_value)

                if self.verbose > 0:
                    print({
                        "Model": name,
                        "Accuracy": accuracy,
                        "R^2 Score": r2,
                        "Mean Squared Error": mse,
                        custom_metric.__name__: custom_metric_value,
                        "Time taken": end_time - start
                    })

    # Create a DataFrame
    df_results = pd.DataFrame({
        'Model': names,
        'Accuracy': Accuracy,
        'R^2 Score': R2_Score,
        'Mean Squared Error': Mean_Squared_Error,
        'Time Taken': TIME
    })
    
    # Display results in tabular form
    table = tabulate(df_results, headers='keys', tablefmt='pipe', showindex=False)
    best_model_index = df_results['Accuracy'].idxmax()
    best_model_name = df_results.loc[best_model_index, 'Model']
    best_model_accuracy = df_results.loc[best_model_index, 'Accuracy']
    
    return table, best_model_name, best_model_accuracy
