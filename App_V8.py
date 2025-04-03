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

    # Default for DOB enable
    defaults["enable_dob"] = False

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

    threshold_percent = st.sidebar.slider("Set Confidence Threshold (%)", min_value=0, max_value=100, value=50, step=1)
    
    st.sidebar.markdown(
        """
        <p style="font-size: 12px;">
        <strong>Confidence Threshold Setting</strong><br>
        Adjust the slider above to set the minimum confidence threshold (in percentage) for displaying predictions. 
        Only predictions with a confidence level above this threshold will be shown, ensuring more reliable results.
        </p>
        """,
        unsafe_allow_html=True
    )
    
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
        enable_dob = st.checkbox("Select Date of Birth from Calendar", key="enable_dob")
        today = datetime.date.today()
        if enable_dob:
            min_date = datetime.date(today.year - 200, 1, 1)
            dob = st.date_input("Date of Birth (DOB)", min_value=min_date, max_value=today, format="DD/MM/YYYY", key="dob")
            calculated_age = (today - dob).days / 365.25
            st.write(f"Calculated Age: **{round(calculated_age, 1)} years**")
            user_input['age_year'] = round(calculated_age, 1)
        else:
            age_year_direct = st.number_input("Age (in years)", min_value=0.0, max_value=200.0, step=1.0, key="age_year_direct")
            user_input['age_year'] = age_year_direct
        
        st.markdown("### Month of Illness Information")
        default_month_name = "January"
        selected_month = st.selectbox("Select Month", list(months.keys()),
                                      index=list(months.keys()).index(default_month_name),
                                      key="month")
        user_input['month'] = months[selected_month]
        
        st.markdown("### Duration Information")
        user_input['durationofillness'] = st.number_input("Duration of Illness (in days)", min_value=1, max_value=3000, key="durationofillness")
        
        st.header("Patient Symptoms")
        # Dictionary to track group selections and whether sub symptoms are selected.
        groups_enabled = []
        groups_without_subsymptoms = []
        
        # Process each disease group.
        for disease, symptoms in disease_groups.items():
            st.subheader(disease)
            toggle_key = f"enable_{disease}"
            enabled = st.checkbox(f"Enable {disease} symptoms", key=toggle_key)
            # The first symptom is automatically set to "Yes" if the group is enabled.
            user_input[symptoms[0]] = "Yes" if enabled else "No"
            remaining = symptoms[1:]
            # Track if any additional sub symptom is selected for this disease group.
            additional_selected = False
            if remaining:
                num_columns = 3
                columns = st.columns(num_columns)
                for idx, symptom in enumerate(remaining):
                    col = columns[idx % num_columns]
                    disp_name = symptom_display_names.get(symptom, symptom.replace('_', ' ').title())
                    if enabled:
                        selection = col.radio(
                            disp_name,
                            options=["No", "Yes"],
                            key=symptom,
                            horizontal=True
                        )
                        user_input[symptom] = selection
                        if selection == "Yes":
                            additional_selected = True
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
            # Collect which groups were enabled and if they lack additional symptoms.
            if user_input[symptoms[0]] == "Yes":
                groups_enabled.append(disease)
                if not additional_selected:
                    groups_without_subsymptoms.append(disease)

        # Predict button
        if st.button("Predict"):
            # If no disease group was enabled at all.
            if not groups_enabled:
                st.write("No symptoms were selected. You appear to be healthy; please provide some symptom details to proceed with a virus prediction.")
            # If disease groups were enabled but none have any additional symptoms.
            elif len(groups_enabled) == len(groups_without_subsymptoms):
                for disease in groups_without_subsymptoms:
                    st.write(f"For **{disease}**, you enabled the group but did not select any additional symptoms.")
                st.write("Since no additional symptom details were provided for the enabled groups, no virus prediction can be made.")
            else:
                # Display warnings for enabled groups that did not have additional symptom selections.
                if groups_without_subsymptoms:
                    for disease in groups_without_subsymptoms:
                        st.write(f"Note: For **{disease}**, you enabled the group but did not select any additional symptoms.")
                # Create a base input dataframe.
                ordered_input = {feature: user_input[feature] for feature in features}
                base_input_df = pd.DataFrame([ordered_input])
                print(base_input_df)
                
                # --------------------
                # Run the binary model.
                # --------------------
                try:
                    binary_input_df = base_input_df.copy()
                    binary_label_encoders = joblib.load('label_encoders_xgb_dengue.pkl')
                    binary_label_encoder_y = joblib.load('label_encoder_y_xgb_dengue.pkl')
                    binary_model = joblib.load('model_xgb_dengue.pkl')
                    
                    for col in binary_input_df.columns:
                        if col in binary_label_encoders:
                            le = binary_label_encoders[col]
                            binary_input_df[col] = le.transform(binary_input_df[col])
                    binary_input_df = binary_input_df.astype(np.int32)
                    
                    binary_probabilities = binary_model.predict_proba(binary_input_df)[0]
                    binary_prediction_index = np.argmax(binary_probabilities)
                    binary_class_mapping = {0: "Dengue", 1: "Non-Dengue"}
                    binary_prediction = binary_class_mapping[binary_prediction_index]
                    
                    st.info(f"Binary Model Prediction: **{binary_prediction}**")
                except Exception as e:
                    st.error(f"An error occurred in the binary model: {e}")
                
                # --------------------
                # Run the full multi-class model.
                # --------------------
                try:
                    full_input_df = base_input_df.copy()
                    label_encoders = joblib.load('label_encoders_bi_lstm_E.pkl')
                    label_encoder_y = joblib.load('label_encoder_y_bi_lstm_E.pkl')
                    model = joblib.load('model_bi_lstm_best_E.keras')
                    
                    for col in full_input_df.columns:
                        if col in label_encoders:
                            le = label_encoders[col]
                            full_input_df[col] = le.transform(full_input_df[col])
                    full_input_df = full_input_df.astype(np.int32)
                    
                    probabilities = model.predict(full_input_df)[0]
                    class_indices = np.argsort(probabilities)[::-1]
                    class_names = label_encoder_y.inverse_transform(class_indices)
                    sorted_probabilities = probabilities[class_indices]
                    
                    if binary_prediction.lower() == "non-dengue":
                        filtered = [(name, prob) for name, prob in zip(class_names, sorted_probabilities)
                                    if name.lower() != "dengue"]
                        if filtered:
                            class_names, sorted_probabilities = zip(*filtered)
                        else:
                            class_names, sorted_probabilities = ([], [])
                    
                    st.header("Predicted Viruses with Confidence:")
                    predictions_shown = False
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
