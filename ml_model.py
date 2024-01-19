# pcos_detection_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE


def load_data():
    # Read data from the Excel file
    df = pd.read_excel(r"C:\Users\ananya\Downloads\PCOS_data.xlsx", sheet_name="Full_new")


    # Preprocess the data (you may replace this with your actual preprocessing steps)
    df['II    beta-HCG(mIU/mL)'] = pd.to_numeric(df['II    beta-HCG(mIU/mL)'], errors='coerce')
    df['AMH(ng/mL)'] = pd.to_numeric(df['AMH(ng/mL)'], errors='coerce')
    df.drop('Unnamed: 44', axis=1, inplace=True)

    # Feature selection
    df_new = df_new = df[['PCOS (Y/N)', 'Follicle No. (R)', 'Follicle No. (L)', 'Skin darkening (Y/N)',
             'hair growth(Y/N)', 'Weight gain(Y/N)', 'Cycle(R/I)', 'Fast food (Y/N)',
             'Pimples(Y/N)', 'AMH(ng/mL)', 'Weight (Kg)']].copy()


    # Handling missing values
    df_new.dropna(inplace=True)

    return df_new
def train_model(df):
    # Define features (X) and target variable (y)
    X = df.drop(['PCOS (Y/N)'], axis=1)
    y = df['PCOS (Y/N)']

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize logistic regression model
    logreg = LogisticRegression(max_iter=1000)  

    # Train the model
    logreg.fit(x_train, y_train)

    return logreg


def predict_probability(model, input_data):

    # Extract features from the input data
    features = pd.DataFrame([input_data])

    # Use the trained model to get the probability
    probability = model.predict_proba(features)[:, 1].item()

    return probability
