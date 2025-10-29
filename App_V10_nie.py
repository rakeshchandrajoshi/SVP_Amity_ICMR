import streamlit as st
import pandas as pd
import joblib
import numpy as np
import datetime

# =============================
# Data definitions
# =============================
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

# =============================
# Disease categories and symptoms
# =============================
disease_groups = {
    "Diarrheal Diseases": [
        'diarrhoea', 'dia_fever', 'dia_diarrhoea', 'dia_dysentery', 'dia_pain', 'dia_vomiting'
    ],
    "Respiratory Infections": [
        'respiratory_c', 'res_sore', 'res_cough', 'res_rhinorrhoe', 'res_breath', 'res_fever'
    ],
    "Fever and Inflammatory Responses": [
        'fev_fever', 'fev_any_loc_sym', 'rash_mac', 'rash_papule', 'rash_mac_pop',
        'rash_eschar', 'rash_pustule', 'rash_bullae', 'rash_fev'
    ],
    "Jaundice and Hepatic Issues": [
        'jaundice', 'jau_fever', 'jau_jaundice', 'jau_urine', 'jau_hep',
        'jau_nausea', 'jau_vomiting', 'jau_abpain'
    ],
    "Neurological Symptoms (Encephalitis)": [
        'encephalitis', 'enc_fever', 'enc_seizures', 'enc_rigidity', 'enc_sensorium',
        'enc_ment_status', 'enc_somnelen', 'enc_irritab'
    ],
    "Hemorrhagic Symptoms": [
        'hem_fever', 'hem_rigors', 'hem_headache', 'hem_chills', 'hem_malaise',
        'hem_artharalgia', 'hem_myalgia', 'hem_hemanifestat', 'hem_retro_orbital'
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

# =============================
# Symptom display names
# =============================
symptom_display_names = {s: s.replace('_', ' ').title() for g in disease_groups.values() for s in g}


def initialize_defaults():
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
        "enable_dob": False
    }
    for disease, symptoms in disease_groups.items():
        for symptom in symptoms:
            defaults[symptom] = "No"
        defaults[f"enable_{disease}"] = False
    return defaults


# =============================
# Main App
# =============================
def main():
    st.set_page_config(page_title="Virus Prediction App", layout="wide")

    # --- Header ---
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        st.image("logo_1.jpeg", width=300)
    with col2:
        st.image("logo_2.jpeg", width=250)
    with col3:
        st.image("Amity_logo2.png", width=250)

    # --- Sidebar Navigation ---
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["Home", "Prediction", "About"])

    # --- Home Page ---
    if page == "Home":
        st.markdown("<h1 style='text-align:center;'>Personalized Recommender System for Virus Diagnosis</h1>", unsafe_allow_html=True)
        st.write("""
        Welcome to the AI-powered Virus Prediction App!  
        Enter patient details and symptoms to get dynamic virus predictions.  
        """)
        st.warning("**Disclaimer:** This tool assists research and should not replace medical advice.")

    # --- About Page ---
    elif page == "About":
        st.title("About This App")
        st.write("""
        This application leverages AI to predict potential viral infections based on symptoms and demographics.  
        It integrates supervised learning models trained on real-world medical data.
        """)
        st.warning("This tool is for research support and educational use only.")

    # --- Prediction Page ---
    elif page == "Prediction":
        st.title("Symptoms-Based Virus Prediction")

        # Initialize defaults in session state
        defaults = initialize_defaults()
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

        # Reset button
        if st.button("Reset Selections"):
            for k, v in defaults.items():
                st.session_state[k] = v
            st.success("All inputs reset to default values.")

        # --- Patient Demographics ---
        st.header("Patient Demographics")
        user_input = {}
        user_input['state_patient'] = st.selectbox("State", states, key="state_patient")
        user_input['gender'] = st.radio("Gender", ["Male", "Female"], key="gender")

        st.markdown("### Age Information")
        enable_dob = st.checkbox("Select Date of Birth from Calendar", key="enable_dob")
        today = datetime.date.today()
        if enable_dob:
            dob = st.date_input("Date of Birth", value=default_dob, max_value=today, key="dob")
            calc_age = (today - dob).days / 365.25
            st.write(f"Calculated Age: **{round(calc_age, 1)} years**")
            user_input['age_year'] = round(calc_age, 1)
        else:
            user_input['age_year'] = st.number_input("Age (in years)", 0.0, 200.0, step=1.0, key="age_year_direct")

        st.markdown("### Month and Duration")
        month_name = st.selectbox("Month of Illness", list(months.keys()), key="month")
        user_input['month'] = months[month_name]
        user_input['durationofillness'] = st.number_input("Duration of Illness (days)", 1, 3000, key="durationofillness")

        # --- Symptoms Section ---
        st.header("Patient Symptoms")
        groups_enabled = []
        groups_without_subsymptoms = []

        for disease, symptoms in disease_groups.items():
            st.subheader(disease)
            toggle_key = f"enable_{disease}"
            enabled = st.checkbox(f"Enable {disease} symptoms", key=toggle_key)
            user_input[symptoms[0]] = "Yes" if enabled else "No"
            additional_selected = False

            cols = st.columns(3)
            for idx, symptom in enumerate(symptoms[1:]):
                col = cols[idx % 3]
                disp_name = symptom_display_names.get(symptom, symptom)
                if enabled:
                    sel = col.radio(disp_name, ["No", "Yes"], key=symptom, horizontal=True)
                    user_input[symptom] = sel
                    if sel == "Yes":
                        additional_selected = True
                else:
                    col.radio(disp_name, ["No", "Yes"], key=symptom, index=0, horizontal=True, disabled=True)
                    user_input[symptom] = "No"

            if enabled:
                groups_enabled.append(disease)
                if not additional_selected:
                    groups_without_subsymptoms.append(disease)

        # --- Auto Prediction (Dynamic) ---
        if True:
            if not groups_enabled:
                st.info("No symptoms selected. Please provide symptom details to get a prediction.")
                return

            ordered_input = {f: user_input[f] for f in features}
            base_df = pd.DataFrame([ordered_input])

            # --- Binary Model ---
            try:
                binary_input_df = base_df.copy()
                binary_encoders = joblib.load("label_encoders_dengue.pkl")
                binary_model = joblib.load("model_dengue.pkl")

                for col in binary_input_df.columns:
                    if col in binary_encoders:
                        le = binary_encoders[col]
                        binary_input_df[col] = le.transform(binary_input_df[col])

                binary_input_df = binary_input_df.astype(np.int32)
                binary_probs = binary_model.predict_proba(binary_input_df)[0]
                binary_pred = "Dengue" if np.argmax(binary_probs) == 0 else "Non-Dengue"
                st.info(f"Binary Model Prediction: **{binary_pred}**")
            except Exception as e:
                st.error(f"Binary model error: {e}")

            # --- Full Multi-Class Model ---
            try:
                full_input_df = base_df.copy()
                encoders = joblib.load("label_encoders_best_small_E.pkl")
                le_y = joblib.load("label_encoder_y_best_small_E.pkl")
                model = joblib.load("model_best_small_E.pkl")

                for col in full_input_df.columns:
                    if col in encoders:
                        le = encoders[col]
                        full_input_df[col] = full_input_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

                full_input_df = full_input_df.astype(float)
                probs = model.predict_proba(full_input_df)[0]
                class_names = le_y.inverse_transform(range(len(probs)))
                predictions = sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True)

                # Adaptive threshold
                mean_conf = np.mean(probs)
                std_conf = np.std(probs)
                threshold_percent = min((mean_conf + std_conf) * 100, 95)

                if binary_pred.lower() == "non-dengue":
                    predictions = [(n, p) for n, p in predictions if n.lower() != "dengue"]

                st.header("Predicted Viruses (Adaptive Confidence)")
                shown = False
                for i, (name, prob) in enumerate(predictions):
                    if prob * 100 >= threshold_percent:
                        st.success(f"{i + 1}. **{name}** — {prob * 100:.2f}% confidence")
                        shown = True

                if not shown:
                    name, prob = predictions[0]
                    st.info(f"Top prediction: **{name}** ({prob * 100:.2f}% confidence)")

                st.caption(f"(Adaptive threshold: {threshold_percent:.2f}%)")

            except Exception as e:
                st.error(f"Full model error: {e}")

            st.warning("This report was generated by AI. Please consult a healthcare professional for confirmation.")


# Run app
if __name__ == "__main__":
    main()
