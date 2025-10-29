import streamlit as st
import pandas as pd
import joblib
import datetime
import numpy as np

# ------------------------------
# Define full state list
states = [
    'Andaman And Nicobar Islands', 'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar',
    'Chandigarh', 'Chhattisgarh', 'Dadra and Nagar Haveli', 'Daman and Diu', 'Delhi',
    'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu and Kashmir', 'Jharkhand',
    'Karnataka', 'Kerala', 'Ladakh', 'Lakshadweep', 'Madhya Pradesh', 'Maharashtra',
    'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Puducherry', 'Punjab',
    'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh',
    'Uttarakhand', 'West Bengal'
]

# Month mapping
months = { "January": 1, "February": 2, "March": 3, "April": 4, "May":5, "June":6, "July":7, 
           "August":8, "September":9, "October":10, "November":11, "December":12 }

# ------------------------------
# Full disease groups & symptoms
disease_groups = {
    "Diarrheal Diseases": ['diarrhoea', 'dia_fever', 'dia_vomiting', 'dia_abdominal_pain'],
    "Respiratory Infections": ['res_sore', 'res_cough', 'res_fever', 'res_rhinorrhea'],
    "Skin Infections": ['skin_rash', 'skin_itch', 'skin_lesions'],
    "Vector Borne": ['vb_fever', 'vb_headache', 'vb_joint_pain', 'vb_rash'],
    "Other": ['other_symptom1', 'other_symptom2']
}

# Symptom display names
symptom_display_names = {
    'diarrhoea': "Diarrhoea", 'dia_fever': "Fever", 'dia_vomiting': "Vomiting", 'dia_abdominal_pain': "Abdominal Pain",
    'res_sore': "Sore Throat", 'res_cough': "Cough", 'res_fever': "Fever", 'res_rhinorrhea': "Runny Nose",
    'skin_rash': "Rash", 'skin_itch': "Itching", 'skin_lesions': "Lesions",
    'vb_fever': "Fever", 'vb_headache': "Headache", 'vb_joint_pain': "Joint Pain", 'vb_rash': "Rash",
    'other_symptom1': "Other Symptom 1", 'other_symptom2': "Other Symptom 2"
}

# ------------------------------
# Initialize default values
def initialize_defaults():
    defaults = {
        'state_patient': states[0],
        'gender': "Male",
        'dob': datetime.date.today(),
        'age_year_direct': 0,
        'age_year': 0,
        'month': "January",
        'durationofillness': 1,
    }
    for disease, symptoms in disease_groups.items():
        for symptom in symptoms:
            defaults[symptom] = "No"
        defaults[f"enable_{disease}"] = False
    defaults["enable_dob"] = False
    return defaults

# ------------------------------
# Streamlit App
def main():
    st.title("Virus Prediction App")
    
    defaults = initialize_defaults()

    # --- Patient Information ---
    st.header("Patient Information")
    state_patient = st.selectbox("State", states)
    gender = st.radio("Gender", ["Male", "Female"])
    
    enable_dob = st.checkbox("Enable Date of Birth", value=False)
    if enable_dob:
        dob = st.date_input("Date of Birth", value=datetime.date.today())
        age_year = round((datetime.date.today() - dob).days / 365.25, 1)
    else:
        age_year_direct = st.number_input("Age in Years", min_value=0, max_value=120, value=0)
        age_year = age_year_direct
    
    month = st.selectbox("Month of Symptom Onset", list(months.keys()), index=0)
    durationofillness = st.number_input("Duration of Illness (days)", min_value=1, max_value=30, value=1)

    # --- Symptoms Selection in Columns ---
    st.header("Symptoms")
    selected_symptoms = {}
    cols = st.columns(len(disease_groups))
    for col, (disease, symptoms) in zip(cols, disease_groups.items()):
        with col:
            enable_group = st.checkbox(f"{disease}", value=False)
            if enable_group:
                for symptom in symptoms:
                    selected_symptoms[symptom] = st.selectbox(symptom_display_names[symptom], ["No", "Yes"], index=0)
            else:
                for symptom in symptoms:
                    selected_symptoms[symptom] = "No"

    # --- Predict Button ---
    if st.button("Predict"):
        # Collect input data
        input_data = {
            'state_patient': state_patient,
            'gender': gender,
            'durationofillness': durationofillness,
            'age_year': age_year,
            'month': months[month]
        }
        for symptom in selected_symptoms:
            input_data[symptom] = 1 if selected_symptoms[symptom] == "Yes" else 0

        input_df = pd.DataFrame([input_data])

        # Load models
        try:
            binary_model = joblib.load("binary_model.pkl")
            multi_model = joblib.load("multi_class_model.pkl")
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return

        # --- Binary Prediction ---
        binary_prob = binary_model.predict_proba(input_df)[0][1]
        st.subheader("Binary Prediction")
        st.write(f"Probability of Viral Infection: {binary_prob*100:.2f}%")

        # --- Multi-Class Prediction ---
        multi_probs = multi_model.predict_proba(input_df)[0]
        class_names = multi_model.classes_
        sorted_probabilities = sorted(zip(class_names, multi_probs), key=lambda x: x[1], reverse=True)

        # Automatic thresholding
        probs = [prob for _, prob in sorted_probabilities]
        mean_conf = np.mean(probs)
        std_conf = np.std(probs)
        top_prob = max(probs)
        threshold_percent = min((mean_conf + std_conf) * 100, top_prob * 90, 95)

        st.subheader("Multi-Class Prediction")
        predictions_shown = False
        for i, (name, prob) in enumerate(sorted_probabilities):
            if prob*100 >= threshold_percent:
                st.write(f"{i+1}. **{name}** - {prob*100:.2f}% confidence")
                predictions_shown = True
        if not predictions_shown:
            st.write("No predictions exceeded the automatic confidence threshold.")

# ------------------------------
if __name__ == "__main__":
    main()











# import streamlit as st
# import pandas as pd
# import joblib
# import datetime
# import numpy as np

# # ------------------------------
# # Define full state list
# states = [
#     'Andaman And Nicobar Islands', 'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar',
#     'Chandigarh', 'Chhattisgarh', 'Dadra and Nagar Haveli', 'Daman and Diu', 'Delhi',
#     'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu and Kashmir', 'Jharkhand',
#     'Karnataka', 'Kerala', 'Ladakh', 'Lakshadweep', 'Madhya Pradesh', 'Maharashtra',
#     'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Puducherry', 'Punjab',
#     'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh',
#     'Uttarakhand', 'West Bengal'
# ]

# # Month mapping
# months = { "January": 1, "February": 2, "March": 3, "April": 4, "May":5, "June":6, "July":7, 
#            "August":8, "September":9, "October":10, "November":11, "December":12 }

# # ------------------------------
# # Full disease groups & symptoms
# disease_groups = {
#     "Diarrheal Diseases": ['diarrhoea', 'dia_fever', 'dia_vomiting', 'dia_abdominal_pain'],
#     "Respiratory Infections": ['res_sore', 'res_cough', 'res_fever', 'res_rhinorrhea'],
#     "Skin Infections": ['skin_rash', 'skin_itch', 'skin_lesions'],
#     "Vector Borne": ['vb_fever', 'vb_headache', 'vb_joint_pain', 'vb_rash'],
#     "Other": ['other_symptom1', 'other_symptom2']
# }

# # Symptom display names
# symptom_display_names = {
#     'diarrhoea': "Diarrhoea", 'dia_fever': "Fever", 'dia_vomiting': "Vomiting", 'dia_abdominal_pain': "Abdominal Pain",
#     'res_sore': "Sore Throat", 'res_cough': "Cough", 'res_fever': "Fever", 'res_rhinorrhea': "Runny Nose",
#     'skin_rash': "Rash", 'skin_itch': "Itching", 'skin_lesions': "Lesions",
#     'vb_fever': "Fever", 'vb_headache': "Headache", 'vb_joint_pain': "Joint Pain", 'vb_rash': "Rash",
#     'other_symptom1': "Other Symptom 1", 'other_symptom2': "Other Symptom 2"
# }

# # ------------------------------
# # Initialize default values
# def initialize_defaults():
#     defaults = {
#         'state_patient': states[0],
#         'gender': "Male",
#         'dob': datetime.date.today(),
#         'age_year_direct': 0,
#         'age_year': 0,
#         'month': "January",
#         'durationofillness': 1,
#     }
#     for disease, symptoms in disease_groups.items():
#         for symptom in symptoms:
#             defaults[symptom] = "No"
#         defaults[f"enable_{disease}"] = False
#     defaults["enable_dob"] = False
#     return defaults

# # ------------------------------
# # Streamlit App
# def main():
#     st.title("Virus Prediction App")
    
#     defaults = initialize_defaults()

#     # --- Patient Information ---
#     st.header("Patient Information")
#     state_patient = st.selectbox("State", states)
#     gender = st.radio("Gender", ["Male", "Female"])
    
#     enable_dob = st.checkbox("Enable Date of Birth", value=False)
#     if enable_dob:
#         dob = st.date_input("Date of Birth", value=datetime.date.today())
#         age_year = round((datetime.date.today() - dob).days / 365.25, 1)
#     else:
#         age_year_direct = st.number_input("Age in Years", min_value=0, max_value=120, value=0)
#         age_year = age_year_direct
    
#     month = st.selectbox("Month of Symptom Onset", list(months.keys()), index=0)
#     durationofillness = st.number_input("Duration of Illness (days)", min_value=1, max_value=30, value=1)

#     # --- Symptoms Selection ---
#     st.header("Symptoms")
#     selected_symptoms = {}
#     for disease, symptoms in disease_groups.items():
#         enable_group = st.checkbox(f"Enable {disease} Symptoms", value=False)
#         if enable_group:
#             for symptom in symptoms:
#                 selected_symptoms[symptom] = st.selectbox(symptom_display_names[symptom], ["No", "Yes"], index=0)
#         else:
#             for symptom in symptoms:
#                 selected_symptoms[symptom] = "No"

#     # --- Predict Button ---
#     if st.button("Predict"):
#         # Collect input data
#         input_data = {
#             'state_patient': state_patient,
#             'gender': gender,
#             'durationofillness': durationofillness,
#             'age_year': age_year,
#             'month': months[month]
#         }
#         for symptom in selected_symptoms:
#             input_data[symptom] = 1 if selected_symptoms[symptom] == "Yes" else 0

#         input_df = pd.DataFrame([input_data])

#         # Load models
#         try:
#             binary_model = joblib.load("binary_model.pkl")
#             multi_model = joblib.load("multi_class_model.pkl")
#         except Exception as e:
#             st.error(f"Error loading models: {e}")
#             return

#         # --- Binary Prediction ---
#         binary_prob = binary_model.predict_proba(input_df)[0][1]
#         st.subheader("Binary Prediction")
#         st.write(f"Probability of Viral Infection: {binary_prob*100:.2f}%")

#         # --- Multi-Class Prediction ---
#         multi_probs = multi_model.predict_proba(input_df)[0]
#         class_names = multi_model.classes_
#         sorted_probabilities = sorted(zip(class_names, multi_probs), key=lambda x: x[1], reverse=True)

#         # Automatic thresholding
#         probs = [prob for _, prob in sorted_probabilities]
#         mean_conf = np.mean(probs)
#         std_conf = np.std(probs)
#         top_prob = max(probs)
#         threshold_percent = min((mean_conf + std_conf) * 100, top_prob * 90, 95)

#         st.subheader("Multi-Class Prediction")
#         predictions_shown = False
#         for i, (name, prob) in enumerate(sorted_probabilities):
#             if prob*100 >= threshold_percent:
#                 st.write(f"{i+1}. **{name}** - {prob*100:.2f}% confidence")
#                 predictions_shown = True
#         if not predictions_shown:
#             st.write("No predictions exceeded the automatic confidence threshold.")

# # ------------------------------
# if __name__ == "__main__":
#     main()





# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np
# import datetime

# # Data definitions
# states = [
#     'Andaman And Nicobar Islands', 'Andhra Pradesh', 'Arunachal Pradesh', 'Assam',
#     'Bihar', 'Chandigarh', 'Chhattisgarh', 'Delhi', 'Goa', 'Gujarat', 'Haryana',
#     'Himachal Pradesh', 'Jammu And Kashmir', 'Jharkhand', 'Karnataka', 'Kerala',
#     'Ladakh', 'Lakshadweep', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya',
#     'Mizoram', 'Nagaland', 'Odisha', 'Puducherry', 'Punjab', 'Rajasthan', 'Sikkim',
#     'Tamil Nadu', 'Telangana', 'The Dadra And Nagar Haveli And Daman And Diu',
#     'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal'
# ]

# months = {
#     "January": 1, "February": 2, "March": 3, "April": 4,
#     "May": 5, "June": 6, "July": 7, "August": 8,
#     "September": 9, "October": 10, "November": 11, "December": 12
# }

# default_dob = datetime.date.today()

# disease_groups = {
#     "Diarrheal Diseases": ['diarrhoea', 'dia_fever', 'dia_diarrhoea', 'dia_dysentery', 'dia_pain', 'dia_vomiting'],
#     "Respiratory Infections": ['respiratory_c', 'res_sore', 'res_cough', 'res_rhinorrhoe', 'res_breath', 'res_fever'],
#     "Fever and Inflammatory Responses": ['fev_fever', 'fev_any_loc_sym', 'rash_mac', 'rash_papule', 'rash_mac_pop', 'rash_eschar', 'rash_pustule', 'rash_bullae', 'rash_fev'],
#     "Jaundice and Hepatic Issues": ['jaundice', 'jau_fever', 'jau_jaundice', 'jau_urine', 'jau_hep', 'jau_nausea', 'jau_vomiting', 'jau_abpain'],
#     "Neurological Symptoms (Encephalitis)": ['encephalitis', 'enc_fever', 'enc_seizures', 'enc_rigidity', 'enc_sensorium', 'enc_ment_status', 'enc_somnelen', 'enc_irritab'],
#     "Hemorrhagic Symptoms": ['hem_fever', 'hem_rigors', 'hem_headache', 'hem_chills', 'hem_malaise', 'hem_artharalgia', 'hem_myalgia', 'hem_hemanifestat', 'hem_retro_orbital'],
#     "Conjunctivitis Symptoms": ['conjuctivities', 'con_fever', 'con_redness', 'con_discharge', 'con_scrusting']
# }

# features = [
#     'state_patient', 'gender', 'durationofillness', 'diarrhoea', 'dia_fever', 'dia_diarrhoea',
#     'dia_dysentery', 'dia_pain', 'dia_vomiting', 'respiratory_c', 'res_sore', 'res_cough',
#     'res_rhinorrhoe', 'res_breath', 'res_fever', 'fev_fever', 'fev_any_loc_sym', 'rash_mac',
#     'rash_papule', 'rash_mac_pop', 'rash_eschar', 'rash_pustule', 'rash_bullae', 'rash_fev',
#     'jaundice', 'jau_fever', 'jau_jaundice', 'jau_urine', 'jau_hep', 'jau_nausea', 'jau_vomiting',
#     'jau_abpain', 'encephalitis', 'enc_fever', 'enc_seizures', 'enc_rigidity', 'enc_sensorium',
#     'enc_ment_status', 'enc_somnelen', 'enc_irritab', 'hem_fever', 'hem_rigors', 'hem_headache',
#     'hem_chills', 'hem_malaise', 'hem_artharalgia', 'hem_myalgia', 'hem_hemanifestat',
#     'hem_retro_orbital', 'conjuctivities', 'con_fever', 'con_redness', 'con_discharge',
#     'con_scrusting', 'age_year', 'month'
# ]

# symptom_display_names = {
#     'diarrhoea': "Diarrhoea", 'dia_fever': "Fever", 'dia_diarrhoea': "Diarrhoea", 'dia_dysentery': "Dysentery", 'dia_pain': "Abdominal Pain", 'dia_vomiting': "Vomiting",
#     'respiratory_c': "Respiratory Contact", 'res_sore': "Sore Throat", 'res_cough': "Cough", 'res_rhinorrhoe': "Runny Nose", 'res_breath': "Breathing Difficulty", 'res_fever': "Fever",
#     'fev_fever': "Fever", 'fev_any_loc_sym': "Local Symptoms", 'rash_mac': "Macular Rash", 'rash_papule': "Papule", 'rash_mac_pop': "Maculopapular Rash", 'rash_eschar': "Eschar",
#     'rash_pustule': "Pustular Rash", 'rash_bullae': "Bullae", 'rash_fev': "Fever Rash",
#     'jaundice': "Jaundice", 'jau_fever': "Fever", 'jau_jaundice': "Jaundice", 'jau_urine': "Dark Urine", 'jau_hep': "Hepatic Pain", 'jau_nausea': "Nausea", 'jau_vomiting': "Vomiting", 'jau_abpain': "Abdominal Pain",
#     'encephalitis': "Encephalitis", 'enc_fever': "Fever", 'enc_seizures': "Seizures", 'enc_rigidity': "Rigidity", 'enc_sensorium': "Altered Sensorium", 'enc_ment_status': "Mental Status Change", 'enc_somnelen': "Somnolence", 'enc_irritab': "Irritability",
#     'hem_fever': "Fever", 'hem_rigors': "Rigors", 'hem_headache': "Headache", 'hem_chills': "Chills", 'hem_malaise': "Malaise", 'hem_artharalgia': "Joint Pain", 'hem_myalgia': "Muscle Pain", 'hem_hemanifestat': "Hemorrhagic Manifestations", 'hem_retro_orbital': "Retro Orbital Pain",
#     'conjuctivities': "Conjunctivitis", 'con_fever': "Fever", 'con_redness': "Redness", 'con_discharge': "Discharge", 'con_scrusting': "Scrusting"
# }

# def initialize_defaults():
#     today = datetime.date.today()
#     default_age = (today - default_dob).days / 365.25
#     defaults = {
#         'state_patient': states[0],
#         'gender': "Male",
#         'dob': default_dob,
#         'age_year_direct': 0,
#         'age_year': round(default_age, 1),
#         'month': "January",
#         'durationofillness': 1,
#     }
#     for disease, symptoms in disease_groups.items():
#         for symptom in symptoms:
#             defaults[symptom] = "No"
    
#     for disease in disease_groups.keys():
#         defaults[f"enable_{disease}"] = False

#     defaults["enable_dob"] = False
#     return defaults

# def main():
#     st.set_page_config(page_title="Virus Prediction App", layout="wide")
        
#     col1, col2, col3 = st.columns([1, 3, 1])
#     with col1:
#         st.image("logo_1.jpeg", width=300)
#     with col3:
#         st.image("Amity_logo2.png", width=250)
#     with col2:
#         st.image("logo_2.jpeg", width=250)
    
#     st.sidebar.title("Navigation")
#     page = st.sidebar.radio("Go to:", ["Home", "Prediction", "About"])

#     if page == "Home":
#         st.markdown("<h1 style='text-align: center;'>Personalized Recommender System for Virus Research and Diagnosis Laboratory Network</h1>", unsafe_allow_html=True)
#         st.markdown("<h2 style='text-align: center;'>Advancing Diagnostic Decision-Making through Artificial Intelligence</h2>", unsafe_allow_html=True)
#         st.write("Welcome to the Virus Prediction App! AI-driven Personalized Recommender System for healthcare...")
#         st.warning("**Disclaimer**: This site provides general information and is not a substitute for professional medical advice.")

#     elif page == "About":
#         st.title("Virus Prediction App")
#         st.write("This application assists healthcare professionals by predicting the most probable viral infection based on patient symptoms and demographic details.")
#         st.warning("**Disclaimer**: The predictions are for research purposes only and do not replace clinical judgment.")

#     elif page == "Prediction":
#         defaults = initialize_defaults()
#         st.title("Virus Prediction")
#         col1, col2 = st.columns(2)
#         with col1:
#             state_patient = st.selectbox("State of Patient", states, index=0)
#         with col2:
#             gender = st.radio("Gender", ["Male", "Female"])
        
#         enable_dob = st.checkbox("Enable Date of Birth", value=defaults["enable_dob"])
#         if enable_dob:
#             dob = st.date_input("Date of Birth", value=default_dob)
#             age_year = round((datetime.date.today() - dob).days / 365.25, 1)
#         else:
#             age_year_direct = st.number_input("Age in Years", min_value=0, max_value=120, value=0)
#             age_year = age_year_direct

#         month = st.selectbox("Month of Symptom Onset", list(months.keys()), index=0)
#         durationofillness = st.number_input("Duration of Illness (days)", min_value=1, max_value=30, value=1)

#         selected_symptoms = {}
#         for disease, symptoms in disease_groups.items():
#             enable_group = st.checkbox(f"Enable {disease} Symptoms", value=False)
#             if enable_group:
#                 for symptom in symptoms:
#                     selected_symptoms[symptom] = st.selectbox(f"{symptom_display_names[symptom]}", ["No", "Yes"], index=0)
#             else:
#                 for symptom in symptoms:
#                     selected_symptoms[symptom] = "No"

#         if st.button("Predict"):
#             input_data = {
#                 'state_patient': state_patient, 'gender': gender, 'durationofillness': durationofillness,
#                 'age_year': age_year, 'month': months[month]
#             }
#             for symptom in selected_symptoms.keys():
#                 input_data[symptom] = 1 if selected_symptoms[symptom] == "Yes" else 0

#             input_df = pd.DataFrame([input_data])
            
#             # Load models
#             try:
#                 binary_model = joblib.load("binary_model.pkl")
#                 multi_model = joblib.load("multi_class_model.pkl")
#             except Exception as e:
#                 st.error(f"Error loading models: {e}")
#                 return
            
#             # Binary prediction
#             binary_prob = binary_model.predict_proba(input_df)[0][1]  # Assuming class 1 is virus
#             st.subheader("Binary Prediction (Virus or Not)")
#             st.write(f"Probability of Viral Infection: {binary_prob*100:.2f}%")
            
#             # Multi-class prediction
#             multi_probs = multi_model.predict_proba(input_df)[0]
#             class_names = multi_model.classes_
#             sorted_probabilities = sorted(zip(class_names, multi_probs), key=lambda x: x[1], reverse=True)

#             # Automatic threshold
#             probs = [prob for _, prob in sorted_probabilities]
#             mean_conf = np.mean(probs)
#             std_conf = np.std(probs)
#             top_prob = max(probs)
#             threshold_percent = min((mean_conf + std_conf) * 100, top_prob * 90, 95)

#             st.subheader("Multi-Class Prediction (Specific Virus)")
#             predictions_shown = False
#             for i, (name, prob) in enumerate(sorted_probabilities):
#                 if prob * 100 >= threshold_percent:
#                     st.write(f"{i + 1}. **{name}** - {prob*100:.2f}% confidence")
#                     predictions_shown = True
#             if not predictions_shown:
#                 st.write("No predictions exceeded the automatic confidence threshold.")

# if __name__ == "__main__":
#     main()








# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np
# import datetime

# # =============================
# # Data definitions
# # =============================
# states = [
#     'Andaman And Nicobar Islands', 'Andhra Pradesh', 'Arunachal Pradesh', 'Assam',
#     'Bihar', 'Chandigarh', 'Chhattisgarh', 'Delhi', 'Goa', 'Gujarat', 'Haryana',
#     'Himachal Pradesh', 'Jammu And Kashmir', 'Jharkhand', 'Karnataka', 'Kerala',
#     'Ladakh', 'Lakshadweep', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya',
#     'Mizoram', 'Nagaland', 'Odisha', 'Puducherry', 'Punjab', 'Rajasthan', 'Sikkim',
#     'Tamil Nadu', 'Telangana', 'The Dadra And Nagar Haveli And Daman And Diu',
#     'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal'
# ]

# months = {
#     "January": 1, "February": 2, "March": 3, "April": 4,
#     "May": 5, "June": 6, "July": 7, "August": 8,
#     "September": 9, "October": 10, "November": 11, "December": 12
# }
# default_dob = datetime.date.today()

# # =============================
# # Disease categories and symptoms
# # =============================
# disease_groups = {
#     "Diarrheal Diseases": [
#         'diarrhoea', 'dia_fever', 'dia_diarrhoea', 'dia_dysentery', 'dia_pain', 'dia_vomiting'
#     ],
#     "Respiratory Infections": [
#         'respiratory_c', 'res_sore', 'res_cough', 'res_rhinorrhoe', 'res_breath', 'res_fever'
#     ],
#     "Fever and Inflammatory Responses": [
#         'fev_fever', 'fev_any_loc_sym', 'rash_mac', 'rash_papule', 'rash_mac_pop',
#         'rash_eschar', 'rash_pustule', 'rash_bullae', 'rash_fev'
#     ],
#     "Jaundice and Hepatic Issues": [
#         'jaundice', 'jau_fever', 'jau_jaundice', 'jau_urine', 'jau_hep',
#         'jau_nausea', 'jau_vomiting', 'jau_abpain'
#     ],
#     "Neurological Symptoms (Encephalitis)": [
#         'encephalitis', 'enc_fever', 'enc_seizures', 'enc_rigidity', 'enc_sensorium',
#         'enc_ment_status', 'enc_somnelen', 'enc_irritab'
#     ],
#     "Hemorrhagic Symptoms": [
#         'hem_fever', 'hem_rigors', 'hem_headache', 'hem_chills', 'hem_malaise',
#         'hem_artharalgia', 'hem_myalgia', 'hem_hemanifestat', 'hem_retro_orbital'
#     ],
#     "Conjunctivitis Symptoms": [
#         'conjuctivities', 'con_fever', 'con_redness', 'con_discharge', 'con_scrusting'
#     ]
# }

# features = [
#     'state_patient', 'gender', 'durationofillness', 'diarrhoea', 'dia_fever', 'dia_diarrhoea',
#     'dia_dysentery', 'dia_pain', 'dia_vomiting', 'respiratory_c', 'res_sore', 'res_cough',
#     'res_rhinorrhoe', 'res_breath', 'res_fever', 'fev_fever', 'fev_any_loc_sym', 'rash_mac',
#     'rash_papule', 'rash_mac_pop', 'rash_eschar', 'rash_pustule', 'rash_bullae', 'rash_fev',
#     'jaundice', 'jau_fever', 'jau_jaundice', 'jau_urine', 'jau_hep', 'jau_nausea', 'jau_vomiting',
#     'jau_abpain', 'encephalitis', 'enc_fever', 'enc_seizures', 'enc_rigidity', 'enc_sensorium',
#     'enc_ment_status', 'enc_somnelen', 'enc_irritab', 'hem_fever', 'hem_rigors', 'hem_headache',
#     'hem_chills', 'hem_malaise', 'hem_artharalgia', 'hem_myalgia', 'hem_hemanifestat',
#     'hem_retro_orbital', 'conjuctivities', 'con_fever', 'con_redness', 'con_discharge',
#     'con_scrusting', 'age_year', 'month'
# ]

# # =============================
# # Symptom display names
# # =============================
# symptom_display_names = {s: s.replace('_', ' ').title() for g in disease_groups.values() for s in g}


# def initialize_defaults():
#     today = datetime.date.today()
#     default_age = (today - default_dob).days / 365.25
#     defaults = {
#         'state_patient': states[0],
#         'gender': "Male",
#         'dob': default_dob,
#         'age_year_direct': 0,
#         'age_year': round(default_age, 1),
#         'month': "January",
#         'durationofillness': 1,
#         "enable_dob": False
#     }
#     for disease, symptoms in disease_groups.items():
#         for symptom in symptoms:
#             defaults[symptom] = "No"
#         defaults[f"enable_{disease}"] = False
#     return defaults


# # =============================
# # Main App
# # =============================
# def main():
#     st.set_page_config(page_title="Virus Prediction App", layout="wide")

#     # --- Header ---
#     col1, col2, col3 = st.columns([1, 3, 1])
#     with col1:
#         st.image("logo_1.jpeg", width=300)
#     with col2:
#         st.image("logo_2.jpeg", width=250)
#     with col3:
#         st.image("Amity_logo2.png", width=250)

#     # --- Sidebar Navigation ---
#     st.sidebar.title("Navigation")
#     page = st.sidebar.radio("Go to:", ["Home", "Prediction", "About"])

#     # --- Home Page ---
#     if page == "Home":
#         st.markdown("<h1 style='text-align:center;'>Personalized Recommender System for Virus Diagnosis</h1>", unsafe_allow_html=True)
#         st.write("""
#         Welcome to the AI-powered Virus Prediction App!  
#         Enter patient details and symptoms to get dynamic virus predictions.  
#         """)
#         st.warning("**Disclaimer:** This tool assists research and should not replace medical advice.")

#     # --- About Page ---
#     elif page == "About":
#         st.title("About This App")
#         st.write("""
#         This application leverages AI to predict potential viral infections based on symptoms and demographics.  
#         It integrates supervised learning models trained on real-world medical data.
#         """)
#         st.warning("This tool is for research support and educational use only.")

#     # --- Prediction Page ---
#     elif page == "Prediction":
#         st.title("Symptoms-Based Virus Prediction")

#         # Initialize defaults in session state
#         defaults = initialize_defaults()
#         for key, val in defaults.items():
#             if key not in st.session_state:
#                 st.session_state[key] = val

#         # Reset button
#         if st.button("Reset Selections"):
#             for k, v in defaults.items():
#                 st.session_state[k] = v
#             st.success("All inputs reset to default values.")

#         # --- Patient Demographics ---
#         st.header("Patient Demographics")
#         user_input = {}
#         user_input['state_patient'] = st.selectbox("State", states, key="state_patient")
#         user_input['gender'] = st.radio("Gender", ["Male", "Female"], key="gender")

#         st.markdown("### Age Information")
#         enable_dob = st.checkbox("Select Date of Birth from Calendar", key="enable_dob")
#         today = datetime.date.today()
#         if enable_dob:
#             dob = st.date_input("Date of Birth", value=default_dob, max_value=today, key="dob")
#             calc_age = (today - dob).days / 365.25
#             st.write(f"Calculated Age: **{round(calc_age, 1)} years**")
#             user_input['age_year'] = round(calc_age, 1)
#         else:
#             user_input['age_year'] = st.number_input("Age (in years)", 0.0, 200.0, step=1.0, key="age_year_direct")

#         st.markdown("### Month and Duration")
#         month_name = st.selectbox("Month of Illness", list(months.keys()), key="month")
#         user_input['month'] = months[month_name]
#         user_input['durationofillness'] = st.number_input("Duration of Illness (days)", 1, 3000, key="durationofillness")

#         # --- Symptoms Section ---
#         st.header("Patient Symptoms")
#         groups_enabled = []
#         groups_without_subsymptoms = []

#         for disease, symptoms in disease_groups.items():
#             st.subheader(disease)
#             toggle_key = f"enable_{disease}"
#             enabled = st.checkbox(f"Enable {disease} symptoms", key=toggle_key)
#             user_input[symptoms[0]] = "Yes" if enabled else "No"
#             additional_selected = False

#             cols = st.columns(3)
#             for idx, symptom in enumerate(symptoms[1:]):
#                 col = cols[idx % 3]
#                 disp_name = symptom_display_names.get(symptom, symptom)
#                 if enabled:
#                     sel = col.radio(disp_name, ["No", "Yes"], key=symptom, horizontal=True)
#                     user_input[symptom] = sel
#                     if sel == "Yes":
#                         additional_selected = True
#                 else:
#                     col.radio(disp_name, ["No", "Yes"], key=symptom, index=0, horizontal=True, disabled=True)
#                     user_input[symptom] = "No"

#             if enabled:
#                 groups_enabled.append(disease)
#                 if not additional_selected:
#                     groups_without_subsymptoms.append(disease)

#         # --- Auto Prediction (Dynamic) ---
#         if True:
#             if not groups_enabled:
#                 st.info("No symptoms selected. Please provide symptom details to get a prediction.")
#                 return

#             ordered_input = {f: user_input[f] for f in features}
#             base_df = pd.DataFrame([ordered_input])

#             # --- Binary Model ---
#             try:
#                 binary_input_df = base_df.copy()
#                 binary_encoders = joblib.load("label_encoders_dengue.pkl")
#                 binary_model = joblib.load("model_dengue.pkl")

#                 for col in binary_input_df.columns:
#                     if col in binary_encoders:
#                         le = binary_encoders[col]
#                         binary_input_df[col] = le.transform(binary_input_df[col])

#                 binary_input_df = binary_input_df.astype(np.int32)
#                 binary_probs = binary_model.predict_proba(binary_input_df)[0]
#                 binary_pred = "Dengue" if np.argmax(binary_probs) == 0 else "Non-Dengue"
#                 st.info(f"Binary Model Prediction: **{binary_pred}**")
#             except Exception as e:
#                 st.error(f"Binary model error: {e}")

#             # --- Full Multi-Class Model ---
#             try:
#                 full_input_df = base_df.copy()
#                 encoders = joblib.load("label_encoders_best_small_E.pkl")
#                 le_y = joblib.load("label_encoder_y_best_small_E.pkl")
#                 model = joblib.load("model_best_small_E.pkl")

#                 for col in full_input_df.columns:
#                     if col in encoders:
#                         le = encoders[col]
#                         full_input_df[col] = full_input_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

#                 full_input_df = full_input_df.astype(float)
#                 probs = model.predict_proba(full_input_df)[0]
#                 class_names = le_y.inverse_transform(range(len(probs)))
#                 predictions = sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True)

#                 # Adaptive threshold
#                 mean_conf = np.mean(probs)
#                 std_conf = np.std(probs)
#                 threshold_percent = min((mean_conf + std_conf) * 100, 95)

#                 if binary_pred.lower() == "non-dengue":
#                     predictions = [(n, p) for n, p in predictions if n.lower() != "dengue"]

#                 st.header("Predicted Viruses (Adaptive Confidence)")
#                 shown = False
#                 for i, (name, prob) in enumerate(predictions):
#                     if prob * 100 >= threshold_percent:
#                         st.success(f"{i + 1}. **{name}** — {prob * 100:.2f}% confidence")
#                         shown = True

#                 if not shown:
#                     name, prob = predictions[0]
#                     st.info(f"Top prediction: **{name}** ({prob * 100:.2f}% confidence)")

#                 st.caption(f"(Adaptive threshold: {threshold_percent:.2f}%)")

#             except Exception as e:
#                 st.error(f"Full model error: {e}")

#             st.warning("This report was generated by AI. Please consult a healthcare professional for confirmation.")


# # Run app
# if __name__ == "__main__":
#     main()
