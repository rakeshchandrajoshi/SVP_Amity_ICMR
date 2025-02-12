import streamlit as st
import pandas as pd
import joblib

# List of states from your dataset
states = [
    'Uttar Pradesh', 'Goa', 'Kerala', 'Karnataka', 'Odisha', 'Chhattisgarh',
    'Rajasthan', 'Delhi', 'Haryana', 'Andaman And Nicobar Islands',
    'Madhya Pradesh', 'Himachal Pradesh', 'Andhra Pradesh', 'Jammu And Kashmir',
    'Punjab', 'Tamil Nadu', 'Chandigarh', 'Maharashtra', 'Uttarakhand', 'Gujarat',
    'West Bengal', 'Bihar', 'Tripura', 'Telangana', 'Assam', 'Arunachal Pradesh',
    'Jharkhand', 'Puducherry', 'Meghalaya', 'Mizoram', 'Manipur', 'Nagaland',
    'Ladakh', 'Sikkim', 'Lakshadweep',
    'The Dadra And Nagar Haveli And Daman And Diu'
]

# Disease categories and related features
disease_groups = {
    "Diarrheal Diseases": [
        'diarrhoea', 'dia_fever', 'dia_diarrhoea', 'dia_dysentery', 'dia_pain', 'dia_vomiting'
    ],
    "Respiratory Infections": [
        'respiratory_c', 'res_sore', 'res_cough', 'res_rhinorrhoe', 'res_breath', 'res_fever'
    ],
    "Fever and Inflammatory Responses": [
        'fev_fever', 'fev_any_loc_sym', 'rash_mac', 'rash_papule', 'rash_mac_pop', 'rash_eschar',
        'rash_pustule', 'rash_bullae', 'rash_fev'
    ],
    "Jaundice and Hepatic Issues": [
        'jaundice', 'jau_fever', 'jau_jaundice', 'jau_urine', 'jau_hep', 'jau_nausea', 'jau_vomiting',
        'jau_abpain'
    ],
    "Neurological Symptoms (Encephalitis)": [
        'encephalitis', 'enc_fever', 'enc_seizures', 'enc_rigidity', 'enc_sensorium', 'enc_ment_status',
        'enc_somnelen', 'enc_irritab'
    ],
    "Hemorrhagic Symptoms": [
        'hem_fever', 'hem_rigors', 'hem_headache', 'hem_chills', 'hem_malaise', 'hem_artharalgia',
        'hem_myalgia', 'hem_hemanifestat', 'hem_retro_orbital'
    ],
    "Conjunctivitis Symptoms": [
        'conjuctivities', 'con_fever', 'con_redness', 'con_discharge', 'con_scrusting'
    ]
}

features = [
    'state_patient', 'gender', 'durationofillness', 'diarrhoea', 'dia_fever', 'dia_diarrhoea',
    'dia_dysentery', 'dia_pain', 'dia_vomiting', 'respiratory_c', 'res_sore', 'res_cough',
    'res_rhinorrhoe', 'res_breath', 'res_fever', 'fev_fever', 'fev_any_loc_sym', 'rash_mac',
    'rash_papule', 'rash_mac_pop', 'rash_eschar', 'rash_pustule', 'rash_bullae', 'rash_fev',
    'jaundice', 'jau_fever', 'jau_jaundice', 'jau_urine', 'jau_hep', 'jau_nausea', 'jau_vomiting',
    'jau_abpain', 'encephalitis', 'enc_fever', 'enc_seizures', 'enc_rigidity', 'enc_sensorium',
    'enc_ment_status', 'enc_somnelen', 'enc_irritab', 'hem_fever', 'hem_rigors', 'hem_headache',
    'hem_chills', 'hem_malaise', 'hem_artharalgia', 'hem_myalgia', 'hem_hemanifestat',
    'hem_retro_orbital', 'conjuctivities', 'con_fever', 'con_redness', 'con_discharge',
    'con_scrusting', 'age_year', 'month'
]

def initialize_defaults():
    """Return a dictionary of default values for all inputs."""
    defaults = {
        'state_patient': states[0],
        'gender': "Male",
        'age_year': 25.0,
        'month': 1,
        'durationofillness': 10,
    }
    # Set default value "No" for each symptom
    for disease, symptoms in disease_groups.items():
        for symptom in symptoms:
            defaults[symptom] = "No"
    return defaults

def main():
    st.set_page_config(page_title="Virus Prediction App", layout="wide")
    
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        st.image("Amity_logo.png", width=200)  
    with col3:
        st.image("ICMR_logo.png", width=200) 

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["Home", "Prediction", "About"])

    if page == "Home":
        st.markdown(
            "<h1 style='text-align: center;'>Personalized Recommender System for Virus Research and Diagnosis Laboratory Network</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<h2 style='text-align: center;'>Advancing Diagnostic Decision-Making through Artificial Intelligence</h2>",
            unsafe_allow_html=True
        )
        st.write("""
        Welcome to the Virus Prediction App!
        
        AI-driven Personalized Recommender System for healthcare, aiming to optimize diagnostic accuracy, 
        streamline resource allocation, and improve patient outcomes by tailoring laboratory test recommendations, 
        based on patient’s symptoms and other relevant details.
        
        Navigate to the **Prediction** page using the sidebar to input patient details 
        and get a virus classification prediction.
        """)
        st.warning("**Disclaimer**: This site provides general information and is not a substitute for professional medical advice.")

    elif page == "About":
        st.title("Virus Prediction App")
        st.write("""
        This application assists healthcare professionals by predicting 
        the most probable viral infection based on patient symptoms and demographic details.
        """)
        st.warning("**Disclaimer**: This site provides general information and is not a substitute for professional medical advice.")

    elif page == "Prediction":
        st.title("Symptoms Based Virus Prediction")
        
        # Initialize default values in session_state if they don't exist
        defaults = initialize_defaults()
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

        # "Reset Selections" button: update session_state variables to their default values
        if st.button("Reset Selections"):
            for key, value in defaults.items():
                st.session_state[key] = value
            st.success("Selections have been reset to default values.")

        with st.form("patient_form"):
            st.header("Patient Demographics")
            user_input = {}
            user_input['state_patient'] = st.selectbox("State", states, key="state_patient")
            user_input['gender'] = st.radio("Gender", ['Male', 'Female'], key="gender")
            user_input['age_year'] = st.number_input("Age (in years)", min_value=0.0, max_value=200.0,step=0.1, key="age_year")
            user_input['month'] = st.number_input("Month (1-12)", min_value=1, max_value=12, key="month")
            user_input['durationofillness'] = st.number_input("Duration of Illness (in days)", min_value=1, max_value=3000, key="durationofillness")
            
            st.header("Patient Symptoms")
            any_symptom_selected = False
            for disease, symptoms in disease_groups.items():
                with st.expander(disease):
                    for symptom in symptoms:
                        user_input[symptom] = st.radio(
                            f"{symptom.replace('_', ' ').title()}",
                            ['No', 'Yes'],
                            key=symptom
                        )
                        if user_input[symptom] == 'Yes':
                            any_symptom_selected = True
            
            submitted = st.form_submit_button("Predict")
        
        if submitted:
            if not any_symptom_selected:
                st.write("No symptoms were entered. You seem healthy, or no symptoms were selected. No virus detected.")
            else:
                # Reorder inputs to match the features list
                ordered_input = {feature: user_input[feature] for feature in features}

                # Preprocess the input data
                input_df = pd.DataFrame([ordered_input])

                try:
                    # Load label encoder and trained model
                    label_encoder = joblib.load('label_encoder_xgb_N3.pkl')
                    model = joblib.load('model_xgb_N3.pkl')
                    encoded_class_labels = model.classes_  # These are numeric

                    # Convert encoded labels back to original class names
                    class_names = label_encoder.inverse_transform(encoded_class_labels)

                    # Encode categorical features
                    for col in input_df.columns:
                        if input_df[col].dtype == 'object':
                            input_df[col] = label_encoder.fit_transform(input_df[col])

                    # Get prediction probabilities
                    probabilities = model.predict_proba(input_df)[0]

                    # Combine class names and probabilities
                    predictions = list(zip(class_names, probabilities))

                    # Sort predictions by confidence in descending order
                    sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

                    # Display the top 5 predictions with actual virus names
                    st.header("Top 5 Predicted Viruses with Confidence:")
                    for i, (virus, confidence) in enumerate(sorted_predictions[:5]):
                        st.write(f"{i + 1}. **{virus}** - {confidence * 100:.2f}% confidence")

                except Exception as e:
                    st.error(f"An error occurred: {e}")
            
            st.write("\n\n")
            st.warning("This report was generated by AI. Please consult a healthcare professional for accurate diagnosis.")

if __name__ == "__main__":
    main()
