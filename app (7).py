import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="Diabetic Risk Predictor", layout="centered")

# --- Load Model and Features ---
# The model.pkl contains the trained RandomForestClassifier model.
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# The features.pkl contains the list of feature columns the model was trained on,
# ensuring consistent input structure.
with open('features.pkl', 'rb') as file:
    model_features = pickle.load(file)

# --- Streamlit App UI ---
st.title("Diabetic Risk Predictor")
st.markdown("Enter the patient's information to predict their diabetic risk category.")

st.sidebar.header("Patient Input Features")

def get_user_input():
    # Numerical Inputs for raw values and scores
    age = st.sidebar.number_input('Age', min_value=18, max_value=100, value=30)
    waist_circumference = st.sidebar.number_input('Waist Circumference (cm)', min_value=50.0, max_value=200.0, value=90.0)
    fasting_glucose = st.sidebar.number_input('Fasting Glucose (mg/dL)', min_value=50.0, max_value=300.0, value=90.0)
    fasting_triglycerides = st.sidebar.number_input('Fasting Triglycerides (mg/dL)', min_value=30.0, max_value=1000.0, value=100.0)

    # Score Inputs
    age_score = st.sidebar.number_input('Age Score', min_value=0, max_value=30, value=0)
    abdominal_obesity_score = st.sidebar.number_input('Abdominal Obesity Score', min_value=0, max_value=20, value=0)
    physical_activity_score = st.sidebar.number_input('Physical Activity Score', min_value=0, max_value=10, value=0)
    family_history_score = st.sidebar.number_input('Family History Score', min_value=0, max_value=10, value=0)

    # Categorical Inputs
    gender = st.sidebar.selectbox('Gender', ['Female', 'Male'])
    physical_activity_level = st.sidebar.selectbox('Physical Activity Level', [
        'Vigorous exercise or strenuous at work',
        'Moderate exercise at work/home',
        'No exercise and sedentary'
    ])
    family_history_diabetes = st.sidebar.selectbox('Family History of Diabetes', [
        'Either parent diabetic',
        'Two non-diabetic parents'
    ])

    # Calculate TYG Index and Total Diabetic Risk Score internally
    tyg_index = np.log(fasting_glucose * fasting_triglycerides / 2)
    total_diabetic_risk_score = age_score + abdominal_obesity_score + physical_activity_score + family_history_score

    user_data = {
        'Age': age,
        'Waist Circumference (cm)': waist_circumference,
        'Fasting Glucose (mg/dL)': fasting_glucose,
        'Fasting Triglycerides (mg/dL)': fasting_triglycerides,
        'TYG Index': tyg_index, # Calculated
        'Age Score': age_score,
        'Abdominal Obesity Score': abdominal_obesity_score,
        'Physical Activity Score': physical_activity_score,
        'Family History Score': family_history_score,
        'Gender': gender,
        'Physical Activity Level': physical_activity_level,
        'Family History of Diabetes': family_history_diabetes,
        'Total Diabetic Risk Score': total_diabetic_risk_score # Calculated
    }
    return pd.DataFrame(user_data, index=[0])

input_df_raw = get_user_input()

# --- Preprocess User Input to Match Model Features ---
def preprocess_input(input_df_raw, model_features):
    # Create an empty DataFrame with all expected model features, initialized to 0
    df_processed = pd.DataFrame(0, index=[0], columns=model_features)

    # Populate numerical features directly from raw input (including calculated TYG Index)
    numerical_cols = ['Age', 'Waist Circumference (cm)', 'Fasting Glucose (mg/dL)',
                      'Fasting Triglycerides (mg/dL)', 'TYG Index', 'Age Score',
                      'Abdominal Obesity Score', 'Physical Activity Score', 'Family History Score']

    for col in numerical_cols:
        if col in input_df_raw.columns and col in df_processed.columns:
            df_processed[col] = input_df_raw[col].values[0]

    # Handle 'Gender' one-hot encoding (Gender_Male)
    if 'Gender_Male' in df_processed.columns:
        if input_df_raw['Gender'].values[0] == 'Male':
            df_processed['Gender_Male'] = 1

    # Handle 'Physical Activity Level' one-hot encoding
    pa_level_map = {
        'Vigorous exercise or strenuous at work': 'Physical Activity Level_Vigorous exercise or strenuous at work',
        'Moderate exercise at work/home': 'Physical Activity Level_Moderate exercise at work/home',
        'No exercise and sedentary': 'Physical Activity Level_No exercise and sedentary'
    }
    selected_pa_col = pa_level_map.get(input_df_raw['Physical Activity Level'].values[0])
    if selected_pa_col and selected_pa_col in df_processed.columns:
        df_processed[selected_pa_col] = 1

    # Handle 'Family History of Diabetes' one-hot encoding
    fh_diabetes_map = {
        'Either parent diabetic': 'Family History of Diabetes_Either parent diabetic',
        'Two non-diabetic parents': 'Family History of Diabetes_Two non-diabetic parents'
    }
    selected_fh_col = fh_diabetes_map.get(input_df_raw['Family History of Diabetes'].values[0])
    if selected_fh_col and selected_fh_col in df_processed.columns:
        df_processed[selected_fh_col] = 1

    return df_processed

input_for_prediction = preprocess_input(input_df_raw, model_features)

# Display User Input
st.subheader('User Input:')
st.write(input_df_raw.drop(columns=['TYG Index', 'Total Diabetic Risk Score'], errors='ignore'))

st.subheader('Calculated Metrics:')
st.write(f"**TYG Index:** {input_df_raw['TYG Index'].values[0]:.4f}")
st.write(f"**Total Diabetic Risk Score:** {int(input_df_raw['Total Diabetic Risk Score'].values[0])}")

# --- Make Prediction ---
st.subheader('Diabetic Risk Prediction:')

if st.button('Predict'):
    try:
        prediction = model.predict(input_for_prediction)
        st.success(f"The predicted Diabetic Risk Category is: **{prediction[0]}**")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("""
---
### Important Note on Model Performance

This model achieved **perfect accuracy, precision, recall, F1-score, and ROC AUC (1.00)** on the test set during development. This level of performance is highly unusual for real-world datasets and strongly suggests potential issues such as:

*   **Data Leakage**: Information from the target variable might have inadvertently influenced the features.
*   **Synthetic or Overly Simplistic Dataset**: The dataset might be too clean or simplistic, making the prediction task trivial.

While the model performs flawlessly on the provided data, caution is advised when interpreting its predictions for new, unverified data. Further investigation into the dataset's origin and potential leakage is recommended for building a robust and generalizable model.
""")
