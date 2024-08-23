import streamlit as st
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt

# Function to calculate and display results
def calculate_and_display_results(train_data, user_input):
    # Train Cox Proportional Hazards model
    coxph = CoxPHFitter()
    coxph.fit(train_data, duration_col='Survival months', event_col='Vital status')

    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])
    
    # Predict mean survival time
    predicted_mean = coxph.predict_expectation(input_df)[0]
    st.write("Predicted Survival Time (Mean):", predicted_mean)

    # Estimate the survival function for the user input
    survival_function = coxph.predict_survival_function(input_df, times=np.linspace(0, max(train_data['Survival months']), 100))

    # Plot the survival curve
    st.write("Individual Survival Curve")
    fig, ax = plt.subplots()
    ax.step(survival_function.index, survival_function.iloc[:, 0], where="post")
    ax.set_xlabel('Time')
    ax.set_ylabel('Survival Probability')
    ax.set_title('Individual Survival Curve')
    ax.grid(True)
    st.pyplot(fig)

# Streamlit application
def main():
    st.title("Survival Prediction")

    st.sidebar.header("User Input Parameters")
    
    # Variables for user input
    variables = {
        "Sex": st.sidebar.selectbox("Sex", options=["Male", "Female"]),
        "Age": st.sidebar.number_input("Age", min_value=0),
        "Race": st.sidebar.selectbox("Race", options=["White", "Black", "Asian or Pacific Islander", "American Indian/Alaska Native"]),
        "Median household income": st.sidebar.selectbox("Median household income", options=["< $35,000", "$35,000 - $39,999", "$40,000 - $44,999", "$45,000 - $49,999", "$50,000 - $54,999",
                  "$55,000 - $59,999", "$60,000 - $64,999", "$65,000 - $69,999", "$70,000 - $74,999", "≥ $75,000"]),
        "Marital status at diagnosis": st.sidebar.selectbox("Marital status at diagnosis", options=["Single (never married)", "Married (including common law)", "Divorced", "Widowed", "Separated", "Unmarried or Domestic Partner"]),
        "Tumor size": st.sidebar.number_input("Tumor size (mm)", min_value=0),
        "Grade": st.sidebar.selectbox("Grade", options=["I", "II", "III", "IV"]),
        "Stage Group": st.sidebar.selectbox("Stage Group", options=["I", "II", "III", "IV"]),
        "Combined Summary Stage": st.sidebar.selectbox("Combined Summary Stage", options=["Localized", "Regional", "Distant"]),
        "Surgery": st.sidebar.selectbox("Surgery", options=["Yes", "No"]),
        "Radiation": st.sidebar.selectbox("Radiation", options=["Yes", "No"]),
        "Chemotherapy": st.sidebar.selectbox("Chemotherapy", options=["Yes", "No"]),
        "Mets at DX-bone": st.sidebar.selectbox("Mets at DX-bone", options=["Yes", "No"]),
        "Mets at DX-brain": st.sidebar.selectbox("Mets at DX-brain", options=["Yes", "No"]),
        "Mets at DX-lung": st.sidebar.selectbox("Mets at DX-lung", options=["Yes", "No"]),
        "Mets at DX-liver": st.sidebar.selectbox("Mets at DX-liver", options=["Yes", "No"])
    }

    st.sidebar.subheader("Upload Training Data")
    train_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if train_file is not None:
        train_data = pd.read_csv(train_file)
        if st.sidebar.button("Predict"):
            calculate_and_display_results(train_data, variables)

if __name__ == "__main__":
    main()
