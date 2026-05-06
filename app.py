import streamlit as st
import numpy as np
import joblib


st.set_page_config(
    page_title="PCOS Prediction System",
    page_icon="🩺",
    layout="wide"
)

model = joblib.load("pcos_rf_model.pkl")
scaler = joblib.load("scaler.pkl")


st.sidebar.title("About Project")

st.sidebar.info(
    """
    Hi There 👋
    """
)



st.sidebar.write(
    "This prediction is for educational purposes only "
    "and not a medical diagnosis."
)


st.title("🩺 PCOS Prediction System")

st.markdown(
    "AI-powered healthcare prediction app"
)


st.divider()


features = [
    ' Age (yrs)', 'Weight (Kg)', 'Height(Cm) ', 'BMI',
    'Blood Group', 'Pulse rate(bpm) ', 'RR (breaths/min)',
    'Hb(g/dl)', 'Cycle(R/I)', 'Cycle length(days)',
    'Marraige Status (Yrs)', 'Pregnant(Y/N)',
    'No. of aborptions', '  I   beta-HCG(mIU/mL)',
    'FSH(mIU/mL)', 'LH(mIU/mL)', 'FSH/LH',
    'Hip(inch)', 'Waist(inch)', 'Waist:Hip Ratio',
    'TSH (mIU/L)', 'PRL(ng/mL)', 'Vit D3 (ng/mL)',
    'PRG(ng/mL)', 'RBS(mg/dl)', 'Weight gain(Y/N)',
    'hair growth(Y/N)', 'Skin darkening (Y/N)',
    'Hair loss(Y/N)', 'Pimples(Y/N)',
    'Fast food (Y/N)', 'Reg.Exercise(Y/N)',
    'BP _Systolic (mmHg)', 'BP _Diastolic (mmHg)',
    'Follicle No. (L)', 'Follicle No. (R)',
    'Avg. F size (L) (mm)', 'Avg. F size (R) (mm)',
    'Endometrium (mm)'
]


binary_features = [
    'Pregnant(Y/N)',
    'Weight gain(Y/N)',
    'hair growth(Y/N)',
    'Skin darkening (Y/N)',
    'Hair loss(Y/N)',
    'Pimples(Y/N)',
    'Fast food (Y/N)',
    'Reg.Exercise(Y/N)'
]

input_data = []


st.subheader("Patient Information")


col1, col2 = st.columns(2)

for i, feature in enumerate(features):

    current_col = col1 if i % 2 == 0 else col2

    with current_col:

       
        if feature == 'BMI':
            continue

  
        elif feature == 'Blood Group':

            blood_group = st.selectbox(
                "Blood Group",
                [
                    "Select",
                    "A+",
                    "A-",
                    "B+",
                    "B-",
                    "O+",
                    "O-",
                    "AB+",
                    "AB-"
                ]
            )

            blood_group_mapping = {
                "A+": 11,
                "A-": 12,
                "B+": 13,
                "B-": 14,
                "O+": 15,
                "O-": 16,
                "AB+": 17,
                "AB-": 18
            }

            input_data.append(
                blood_group_mapping.get(blood_group, 0)
            )

        elif feature in binary_features:

            value = st.selectbox(
                feature,
                ["Select", "No", "Yes"]
            )

            if value == "Yes":
                input_data.append(1)

            elif value == "No":
                input_data.append(0)

            else:
                input_data.append(0)

     
        else:

            value = st.text_input(
                feature,
                placeholder=f"Enter {feature}"
            )

            try:
                input_data.append(float(value))

            except:
                input_data.append(0)


try:

    weight_index = features.index('Weight (Kg)')
    height_index = features.index('Height(Cm) ')

    weight = input_data[weight_index]
    height_cm = input_data[height_index]

    height_m = height_cm / 100

    bmi = weight / (height_m ** 2)

except:
    bmi = 0

bmi_index = features.index('BMI')

input_data.insert(bmi_index, bmi)


st.info(f"Calculated BMI: {round(bmi, 2)}")

st.divider()


button_col1, button_col2 = st.columns(2)

with button_col1:
    predict_button = st.button("Predict PCOS")

with button_col2:
    clear_button = st.button("Clear Inputs")

if clear_button:
    st.rerun()


if predict_button:

    if sum(input_data) == 0:

        st.warning("Please enter patient details.")

    else:

        with st.spinner("Analyzing patient data..."):

            data = np.array([input_data])

            data_scaled = scaler.transform(data)

            prediction = model.predict(data_scaled)

            probability = model.predict_proba(data_scaled)

            confidence = round(
                np.max(probability) * 100,
                2
            )

        st.metric(
            "Prediction Confidence",
            f"{confidence}%"
        )

        if prediction[0] == 1:

            st.error(
                "High chances of PCOS detected"
            )

        else:

            st.success(
                "Low chances of PCOS detected"
            )
