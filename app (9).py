import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the pre-trained model and label encoder
model = joblib.load('random_forest_model.pkl')
le = joblib.load('label_encoder.pkl')

# --- Streamlit Application ----
st.title('Diabetic Risk Prediction App')
st.write('Enter the patient details below to predict their diabetic risk category.')

# Define a function to calculate the TYG Index
def calculate_tyg_index(fasting_triglycerides, fasting_glucose):
    if fasting_triglycerides <= 0 or fasting_glucose <= 0:
        return 0 # Handle non-positive inputs, though domain knowledge suggests values should be positive
    return np.log((fasting_triglycerides * fasting_glucose) / 2)

# Input widgets for features
st.sidebar.header('Patient Input Features')

age = st.sidebar.number_input('Age', min_value=18, max_value=100, value=30)
gender = st.sidebar.selectbox('Gender', ('Female', 'Male'))
waist_circumference = st.sidebar.number_input('Waist Circumference (cm)', min_value=40.0, max_value=150.0, value=80.0)
physical_activity_level = st.sidebar.selectbox(
    'Physical Activity Level',
    ('Vigorous exercise or strenuous at work', 'Moderate exercise at work/home', 'No exercise and sedentary')
)
family_history_diabetes = st.sidebar.selectbox(
    'Family History of Diabetes',
    ('Either parent diabetic', 'Two non-diabetic parents')
)
age_score = st.sidebar.number_input('Age Score', min_value=0, max_value=30, value=0)
abdominal_obesity_score = st.sidebar.number_input('Abdominal Obesity Score', min_value=0, max_value=20, value=0)
physical_activity_score = st.sidebar.number_input('Physical Activity Score', min_value=0, max_value=30, value=10)
family_history_score = st.sidebar.number_input('Family History Score', min_value=0, max_value=20, value=0)
total_diabetic_risk_score = st.sidebar.number_input('Total Diabetic Risk Score', min_value=0, max_value=100, value=10)
fasting_glucose = st.sidebar.number_input('Fasting Glucose (mg/dL)', min_value=50.0, max_value=300.0, value=90.0)
fasting_triglycerides = st.sidebar.number_input('Fasting Triglycerides (mg/dL)', min_value=50.0, max_value=400.0, value=100.0)

# Button to predict
if st.sidebar.button('Predict Diabetic Risk'):
    # Calculate TYG Index
    tyg_index = calculate_tyg_index(fasting_triglycerides, fasting_glucose)

    # Prepare input DataFrame for prediction
    # Ensure the order and names of columns match the training data's X DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'Waist Circumference (cm)': [waist_circumference],
        'Age Score': [age_score],
        'Abdominal Obesity Score': [abdominal_obesity_score],
        'Physical Activity Score': [physical_activity_score],
        'Family History Score': [family_history_score],
        'Total Diabetic Risk Score': [total_diabetic_risk_score],
        'Fasting Glucose (mg/dL)': [fasting_glucose],
        'Fasting Triglycerides (mg/dL)': [fasting_triglycerides],
        'TYG Index': [tyg_index],
        'Gender_Male': [1 if gender == 'Male' else 0],
        'Physical Activity Level_Moderate exercise at work/home': [1 if physical_activity_level == 'Moderate exercise at work/home' else 0],
        'Physical Activity Level_No exercise and sedentary': [1 if physical_activity_level == 'No exercise and sedentary' else 0],
        'Physical Activity Level_Vigorous exercise or strenuous at work': [1 if physical_activity_level == 'Vigorous exercise or strenuous at work' else 0],
        'Family History of Diabetes_Either parent diabetic': [1 if family_history_diabetes == 'Either parent diabetic' else 0],
        'Family History of Diabetes_Two non-diabetic parents': [1 if family_history_diabetes == 'Two non-diabetic parents' else 0]
    })

    # Ensure all columns exist, even if not selected in `get_dummies` due to `drop_first=True`
    # and that their dtypes are consistent with the training data (e.g. bool to int)
    # This part needs careful handling to match the exact `X` DataFrame structure from training.
    # Based on the provided `X.head()`, the boolean columns from `get_dummies` are `False` or `True`.
    # So we should match that type, or convert to int if the model expects int (RandomForest usually handles both).
    # For simplicity, converting booleans to int for consistency if the model was trained with int representations

    # Column order needs to exactly match X from the training data.
    # Re-creating the column names based on the X.head() provided in the kernel state.
    # The columns in X were: Age, Waist Circumference (cm), Age Score, Abdominal Obesity Score, Physical Activity Score, 
    # Family History Score, Total Diabetic Risk Score, Fasting Glucose (mg/dL), Fasting Triglycerides (mg/dL), TYG Index, 
    # Gender_Male, Physical Activity Level_Moderate exercise at work/home, Physical Activity Level_No exercise and sedentary, 
    # Physical Activity Level_Vigorous exercise or strenuous at work, Family History of Diabetes_Either parent diabetic, 
    # Family History of Diabetes_Two non-diabetic parents

    expected_columns = [
        'Age',
        'Waist Circumference (cm)',
        'Age Score',
        'Abdominal Obesity Score',
        'Physical Activity Score',
        'Family History Score',
        'Total Diabetic Risk Score',
        'Fasting Glucose (mg/dL)',
        'Fasting Triglycerides (mg/dL)',
        'TYG Index',
        'Gender_Male',
        'Physical Activity Level_Moderate exercise at work/home',
        'Physical Activity Level_No exercise and sedentary',
        'Physical Activity Level_Vigorous exercise or strenuous at work',
        'Family History of Diabetes_Either parent diabetic',
        'Family History of Diabetes_Two non-diabetic parents'
    ]

    # Reindex the input_data DataFrame to match the expected column order
    input_data = input_data[expected_columns]

    # Convert boolean columns to integer 0 or 1 if necessary for model compatibility
    # The RandomForestClassifier can handle boolean columns directly if trained with them.
    # However, if the model was trained with integer representations (as pandas converts bools to int in some ops),
    # it's safer to ensure the input types match. Checking X.head() in the kernel state, these columns are indeed boolean (True/False).

    # Make prediction
    prediction_numeric = model.predict(input_data)
    prediction_category = le.inverse_transform(prediction_numeric)

    st.success(f'Predicted Diabetic Risk Category: {prediction_category[0]}')
