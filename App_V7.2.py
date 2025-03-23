import streamlit as st
import pandas as pd
import joblib
import numpy as np
import datetime

# Data definitions
states = [
    'Andaman And Nicobar Islands', 'Andhra Pradesh', 'Arunachal Pradesh', 'Assam',
    'Bihar', 'Chandigarh', 'Chhattisgarh', 'Delhi', 'Goa', 'Gujarat', 'Haryana',
    'Himachal Pradesh', 'Jammu And Kashmir', 'Jharkhand', 'Karnataka', 'Kerala',
    'Ladakh', 'Lakshadweep', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya',
    'Mizoram', 'Nagaland', 'Odisha', 'Puducherry', 'Punjab', 'Rajasthan', 'Sikkim',
    'Tamil Nadu', 'Telangana', 'The Dadra And Nagar Haveli And Daman And Diu',
    'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal'
]

months = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}
default_dob = datetime.date.today()

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

# Mapping for clear symptom names (display names)
symptom_display_names = {
    # Diarrheal Diseases
    'diarrhoea': "Diarrhoea",
    'dia_fever': "Fever",
    'dia_diarrhoea': "Diarrhoea",
    'dia_dysentery': "Dysentery",
    'dia_pain': "Abdominal Pain",
    'dia_vomiting': "Vomiting",
    # Respiratory Infections
    'respiratory_c': "Respiratory Contact",
    'res_sore': "Sore Throat",
    'res_cough': "Cough",
    'res_rhinorrhoe': "Runny Nose",
    'res_breath': "Breathing Difficulty",
    'res_fever': "Fever",
    # Fever and Inflammatory Responses
    'fev_fever': "Fever",
    'fev_any_loc_sym': "Local Symptoms",
    'rash_mac': "Macular Rash",
    'rash_papule': "Papule",
    'rash_mac_pop': "Maculopapular Rash",
    'rash_eschar': "Eschar",
    'rash_pustule': "Pustular Rash",
    'rash_bullae': "Bullae",
    'rash_fev': "Fever Rash",
    # Jaundice and Hepatic Issues
    'jaundice': "Jaundice",
    'jau_fever': "Fever",
    'jau_jaundice': "Jaundice",
    'jau_urine': "Dark Urine",
    'jau_hep': "Hepatic Pain",
    'jau_nausea': "Nausea",
    'jau_vomiting': "Vomiting",
    'jau_abpain': "Abdominal Pain",
    # Neurological Symptoms (Encephalitis)
    'encephalitis': "Encephalitis",
    'enc_fever': "Fever",
    'enc_seizures': "Seizures",
    'enc_rigidity': "Rigidity",
    'enc_sensorium': "Altered Sensorium",
    'enc_ment_status': "Mental Status Change",
    'enc_somnelen': "Somnolence",
    'enc_irritab': "Irritability",
    # Hemorrhagic Symptoms
    'hem_fever': "Fever",
    'hem_rigors': "Rigors",
    'hem_headache': "Headache",
    'hem_chills': "Chills",
    'hem_malaise': "Malaise",
    'hem_artharalgia': "Joint Pain",
    'hem_myalgia': "Muscle Pain",
    'hem_hemanifestat': "Hemorrhagic Manifestations",
    'hem_retro_orbital': "Retro Orbital Pain",
    # Conjunctivitis Symptoms
    'conjuctivities': "Conjunctivitis",
    'con_fever': "Fever",
    'con_redness': "Redness",
    'con_discharge': "Discharge",
    'con_scrusting': "Scrusting"
}

def initialize_defaults():
    """Return a dictionary of default values for all inputs."""
    today = datetime.date.today()
    default_age = (today - default_dob).days / 365.25
    defaults = {
        'state_patient': states[0],
        'gender': "Male",
        'dob': default_dob,
        'age_year_direct': 0,
        'age_year': round(default_age, 1),
        'month': "January",
        'durationofillness': 1,
    }
    for disease, symptoms in disease_groups.items():
        for symptom in symptoms:
            defaults[symptom] = "No"
    
    for disease in disease_groups.keys():
        defaults[f"enable_{disease}"] = False

    return defaults

def main():
    st.set_page_config(page_title="Virus Prediction App", layout="wide")
        
    # Top logos
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        st.image("Amity_logo2.png", width=300)
    with col3:
        st.image("ICMR_logo2.png", width=250)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["Home", "Prediction", "About"])
    
    # Add threshold slider in the sidebar (only used in Prediction page)
    # Slider now goes from 0 to 100
    threshold_percent = st.sidebar.slider("Set Confidence Threshold (%)", min_value=0, max_value=100, value=50, step=1)

    if page == "Home":
        st.markdown(
            "<h1 style='text-align: center;'>Personalized Recommender System for Virus Research and Diagnosis Laboratory Network</h1>",
            unsafe_allow_html=True)
        st.markdown(
            "<h2 style='text-align: center;'>Advancing Diagnostic Decision-Making through Artificial Intelligence</h2>",
            unsafe_allow_html=True)
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
        
        defaults = initialize_defaults()
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

        if st.button("Reset Selections"):
            for key, value in defaults.items():
                st.session_state[key] = value
            st.success("Selections have been reset to default values.")

        # Patient Demographics (dynamic)
        user_input = {}
        st.header("Patient Demographics")
        user_input['state_patient'] = st.selectbox("State", states, key="state_patient")
        user_input['gender'] = st.radio("Gender", ['Male', 'Female'], key="gender")
        
        st.markdown("### Age Information")
        col1, col2, col3 = st.columns([1, 0.5, 1])
        with col1:
            age_year_direct = st.number_input("Age (in years)", min_value=0.0, max_value=200.0, step=1.0, key="age_year_direct")
        with col2:
            st.markdown("<p style='text-align: center;'>**OR**</p>", unsafe_allow_html=True)
        with col3:
            today = datetime.date.today()
            min_date = datetime.date(today.year - 200, 1, 1)
            dob = st.date_input("Date of Birth (DOB)", min_value=min_date, max_value=today, format="DD/MM/YYYY", key="dob")
        calculated_age = (today - dob).days / 365.25
        user_input['age_year'] = round(calculated_age, 1) if dob != default_dob else age_year_direct
        
        st.markdown("### Month Information")
        default_month_name = "January"
        selected_month = st.selectbox("Select Month", list(months.keys()),
                                      index=list(months.keys()).index(default_month_name),
                                      key="month")
        user_input['month'] = months[selected_month]
        
        st.markdown("### Duration Information")
        user_input['durationofillness'] = st.number_input("Duration of Illness (in days)", min_value=1, max_value=3000, key="durationofillness")
        
        st.header("Patient Symptoms")
        any_symptom_selected = False

        for disease, symptoms in disease_groups.items():
            st.subheader(disease)
            toggle_key = f"enable_{disease}"
            enabled = st.checkbox(f"Enable {disease} symptoms", key=toggle_key)
            first_symptom = symptoms[0]
            display_name = symptom_display_names.get(first_symptom, first_symptom.replace('_', ' ').title())
            user_input[first_symptom] = "Yes" if enabled else "No"
            if enabled:
                any_symptom_selected = True
            remaining = symptoms[1:]
            if remaining:
                num_columns = 3
                columns = st.columns(num_columns)
                for idx, symptom in enumerate(remaining):
                    col = columns[idx % num_columns]
                    disp_name = symptom_display_names.get(symptom, symptom.replace('_', ' ').title())
                    if enabled:
                        user_input[symptom] = col.radio(
                            disp_name,
                            options=["No", "Yes"],
                            key=symptom,
                            horizontal=True
                        )
                        if user_input[symptom] == "Yes":
                            any_symptom_selected = True
                    else:
                        col.radio(
                            disp_name,
                            options=["No", "Yes"],
                            key=symptom,
                            index=0,
                            horizontal=True,
                            disabled=True
                        )
                        user_input[symptom] = "No"

        # Predict button
        if st.button("Predict"):
            if not any_symptom_selected:
                st.write("No symptoms were entered. You seem healthy, or no symptoms were selected. No virus detected.")
            else:
                # Create a base input dataframe
                ordered_input = {feature: user_input[feature] for feature in features}
                base_input_df = pd.DataFrame([ordered_input])
                print(base_input_df)
                
                # --------------------
                # First: run the binary model using its own copy of the input dataframe
                # --------------------
                try:
                    binary_input_df = base_input_df.copy()  # create a separate copy
                    # Load binary model and label encoders
                    binary_label_encoders = joblib.load('label_encoders_xgb_dengue.pkl')
                    binary_label_encoder_y = joblib.load('label_encoder_y_xgb_dengue.pkl')
                    binary_model = joblib.load('model_xgb_dengue.pkl')
                    
                    # Encode input for the binary model
                    for col in binary_input_df.columns:
                        if col in binary_label_encoders:
                            le = binary_label_encoders[col]
                            binary_input_df[col] = le.transform(binary_input_df[col])

                    binary_input_df = binary_input_df.astype(np.int32)

                    # Predict probabilities
                    binary_probabilities = binary_model.predict_proba(binary_input_df)[0]  # Get probability scores

                    # Get predicted class (0 or 1)
                    binary_prediction_index = np.argmax(binary_probabilities)  # Index with highest probability

                    # Map 0 -> Dengue, 1 -> Non-Dengue
                    binary_class_mapping = {0: "Dengue", 1: "Non-Dengue"}
                    binary_prediction = binary_class_mapping[binary_prediction_index]

                    st.info(f"Binary Model Prediction: **{binary_prediction}**")

                except Exception as e:
                    st.error(f"An error occurred in the binary model: {e}")
                # --------------------
                # Next: run the full multi-class model using its own copy of the input dataframe
                # --------------------
                try:
                    full_input_df = base_input_df.copy()  # separate copy for full model
                    label_encoders = joblib.load('label_encoders_bi_lstm_E.pkl')
                    label_encoder_y = joblib.load('label_encoder_y_bi_lstm_E.pkl')
                    model = joblib.load('model_bi_lstm_best_E.keras')

                    # Encode input for the full model
                    for col in full_input_df.columns:
                        if col in label_encoders:
                            le = label_encoders[col]
                            full_input_df[col] = le.transform(full_input_df[col])
                    full_input_df = full_input_df.astype(np.int32)

                    probabilities = model.predict(full_input_df)[0]
                    class_indices = np.argsort(probabilities)[::-1]
                    class_names = label_encoder_y.inverse_transform(class_indices)
                    sorted_probabilities = probabilities[class_indices]

                    # If the binary model predicts Non-Dengue, remove any Dengue prediction
                    if binary_prediction.lower() == "non-dengue":
                        filtered = [(name, prob) for name, prob in zip(class_names, sorted_probabilities)
                                    if name.lower() != "dengue"]
                        if filtered:
                            class_names, sorted_probabilities = zip(*filtered)
                        else:
                            class_names, sorted_probabilities = ([], [])

                    st.header("Predicted Viruses with Confidence:")
                    predictions_shown = False
                    # Filter predictions by the threshold (converting probability to percentage)
                    for i in range(len(class_names)):
                        if sorted_probabilities[i] * 100 >= threshold_percent:
                            st.write(f"{i + 1}. **{class_names[i]}** - {sorted_probabilities[i] * 100:.2f}% confidence")
                            predictions_shown = True
                    if not predictions_shown:
                        st.write("No predictions exceeded the set confidence threshold.")

                except Exception as e:
                    st.error(f"An error occurred in the full model: {e}")

            st.warning("This report was generated by AI. Please consult a healthcare professional for accurate diagnosis.")

if __name__ == "__main__":
    main()
